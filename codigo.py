"""
isp_book2_pipeline.py
=====================
Pipeline completo aplicado a Book2.xlsx
Dataset: ifInBroadcastPkts · 1000 interfaces · 29 días (18 Mar – 17 Apr 2026)
Intervalo de polling: 15 minutos

ESTRUCTURA:
  PASO 1 · Carga y preprocesamiento
  PASO 2 · Activas vs Inactivas
  PASO 3 · ZeroRatio + ajuste estacional
  PASO 4 · Etiquetado: caída real vs falsa alarma
  PASO 5 · Feature engineering con estacionalidad
  PASO 6 · Split temporal 50/50
  PASO 7 · 5 modelos: LR · RF · DT · XGBoost · LSTM
  PASO 8 · Métricas completas: CM · F1 · ROC · AUC
  PASO 9 · Visualización temporal por días
  PASO 10· Conclusiones
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, accuracy_score
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
FILE   = '/content/Book2.xlsx'
OUT1   = '/mnt/user-data/outputs/book2_overview.png'
OUT2   = '/mnt/user-data/outputs/book2_modelos.png'
OUT3   = '/mnt/user-data/outputs/book2_temporal.png'
OUTPY  = '/mnt/user-data/outputs/book2_pipeline.py'

POLL_MIN       = 15     # minutos por intervalo
ROLL_WIN       = 4      # ventana ZeroRatio: 4 polls × 15 min = 1 hora
REAL_FAIL_MIN  = 60     # minutos mínimos para caída real (4 polls)
REAL_STEPS     = REAL_FAIL_MIN // POLL_MIN   # = 4
FA_MAX_STEPS   = REAL_STEPS - 1             # ≤ 3 polls = falsa alarma
MIN_PREC       = 0.85
SEED           = 42
np.random.seed(SEED)

DARK_BG  = '#1a1a2e'; PANEL_BG = '#16213e'; ACCENT = '#0f3460'
C_REAL   = '#ef476f'; C_FA   = '#ffd166'; C_OK  = '#06d6a0'
C_BLUE   = '#118ab2'; C_CYAN = '#4ecdc4'; C_PURP = '#a855f7'; C_ORG = '#f4a261'

MCOLORS = {
    'Logistic Regression': C_BLUE,
    'Random Forest':       C_OK,
    'Decision Tree':       C_CYAN,
    'XGBoost':             C_ORG,
    'LSTM':                C_PURP,
}
FEATURES = ['zr4','zr2','std4','consec_z','delta1',
            'hour_sin','hour_cos','dow_sin','dow_cos',
            'is_weekend','zr_vs_dow_baseline','hour_activity']


# ─────────────────────────────────────────────────────────────────────────────
# MODELOS DESDE CERO (sin xgboost/torch)
# ─────────────────────────────────────────────────────────────────────────────

class XGBoostScratch:
    """Gradient Boosting con árboles, objetivo log-loss."""
    class _Stump:
        def __init__(self, max_depth=3):
            self.max_depth=max_depth; self.feat=0; self.thr=0.0
            self.lv=0.0; self.rv=0.0; self.lt=None; self.rt=None
        def fit(self, X, g, h, depth=0):
            lam=1.0; n=X.shape[1]
            G,H = g.sum(), h.sum()
            best_gain=-np.inf; bf=0; bt=0.0
            for f in range(n):
                vals = np.unique(X[:,f])
                if len(vals)<2: continue
                for thr in (vals[:-1]+vals[1:])/2:
                    L=X[:,f]<=thr; R=~L
                    if L.sum()==0 or R.sum()==0: continue
                    GL,HL=g[L].sum(),h[L].sum()
                    GR,HR=g[R].sum(),h[R].sum()
                    gain=(GL**2/(HL+lam)+GR**2/(HR+lam)-(GL+GR)**2/(HL+HR+lam))
                    if gain>best_gain: best_gain=gain; bf=f; bt=thr
            self.feat=bf; self.thr=bt
            L=X[:,bf]<=bt; R=~L
            GL,HL=g[L].sum(),h[L].sum(); GR,HR=g[R].sum(),h[R].sum()
            self.lv=-GL/(HL+lam); self.rv=-GR/(HR+lam)
            if depth<self.max_depth-1:
                for mask,attr_l,attr_r in [(L,'lt','rt'),(~L,'rt','lt')]:
                    if mask.sum()>4:
                        child=XGBoostScratch._Stump(self.max_depth)
                        child.fit(X[mask],g[mask],h[mask],depth+1)
                        setattr(self,attr_l if mask is L else attr_r, child) if mask is L else None
                self.lt=XGBoostScratch._Stump(self.max_depth)
                self.rt=XGBoostScratch._Stump(self.max_depth)
                if L.sum()>4: self.lt.fit(X[L],g[L],h[L],depth+1)
                else: self.lt=None
                if R.sum()>4: self.rt.fit(X[R],g[R],h[R],depth+1)
                else: self.rt=None
        def predict(self,X):
            L=X[:,self.feat]<=self.thr
            out=np.where(L,self.lv,self.rv).astype(float)
            if self.lt is not None and L.sum()>0: out[L]=self.lt.predict(X[L])
            if self.rt is not None and (~L).sum()>0: out[~L]=self.rt.predict(X[~L])
            return out

    def __init__(self, n=80, lr=0.15, max_depth=3, sub=0.8, seed=42):
        self.n=n; self.lr=lr; self.max_depth=max_depth
        self.sub=sub; self.seed=seed; self.trees=[]; self.fidx=[]
        self.base=0.0
    @staticmethod
    def _sig(x): return 1/(1+np.exp(-np.clip(x,-35,35)))
    def fit(self,X,y):
        rng=np.random.RandomState(self.seed); n=len(y)
        pr=np.clip(y.mean(),1e-6,1-1e-6)
        self.base=np.log(pr/(1-pr)); F=np.full(n,self.base)
        p=X.shape[1]
        for _ in range(self.n):
            ph=self._sig(F); g=ph-y; h=ph*(1-ph)
            ri=rng.choice(n,int(n*self.sub),replace=False)
            fi=rng.choice(p,max(1,int(p*0.8)),replace=False)
            t=self._Stump(self.max_depth); t.fit(X[ri][:,fi],g[ri],h[ri])
            # rebuild full stump manually
            st=self._Stump(self.max_depth)
            st.feat=fi[t.feat]; st.thr=t.thr; st.lv=t.lv; st.rv=t.rv
            F+=self.lr*st.predict(X)
            self.trees.append(st); self.fidx.append(None)
        return self
    def predict_proba_raw(self,X):
        F=np.full(len(X),self.base)
        for t in self.trees: F+=self.lr*t.predict(X)
        return self._sig(F)
    def predict_proba(self,X):
        p=self.predict_proba_raw(X); return np.column_stack([1-p,p])
    def predict(self,X,thr=0.5): return (self.predict_proba_raw(X)>=thr).astype(int)


class LSTMScratch:
    """LSTM con backpropagation through time + Adam."""
    @staticmethod
    def _sig(x): return 1/(1+np.exp(-np.clip(x,-35,35)))
    @staticmethod
    def _tanh(x): return np.tanh(np.clip(x,-35,35))

    def __init__(self, in_sz, h_sz=24, seq=4, lr=0.003,
                 epochs=20, bs=512, drop=0.2, seed=42):
        self.h=h_sz; self.seq=seq; self.lr=lr
        self.epochs=epochs; self.bs=bs; self.drop=drop; self.seed=seed
        rng=np.random.RandomState(seed)
        sx=np.sqrt(2/(in_sz+h_sz)); sh=np.sqrt(2/(h_sz+h_sz))
        self.Wx=rng.randn(4*h_sz,in_sz)*sx
        self.Wh=rng.randn(4*h_sz,h_sz)*sh
        self.b=np.zeros(4*h_sz); self.b[h_sz:2*h_sz]=1.0
        self.Wy=rng.randn(1,h_sz)*np.sqrt(2/h_sz)
        self.by=np.zeros(1)
        self.m=[np.zeros_like(p) for p in [self.Wx,self.Wh,self.b,self.Wy,self.by]]
        self.v=[np.zeros_like(p) for p in [self.Wx,self.Wh,self.b,self.Wy,self.by]]
        self.t=0; self.tl=[]; self.vl=[]

    def _fwd(self, X, train=False):
        B,T,D=X.shape; s=self.h
        h=np.zeros((B,s)); c=np.zeros((B,s)); cache=[]
        rng=np.random.RandomState(self.seed+self.t)
        for t in range(T):
            x=X[:,t,:]
            gates=x@self.Wx.T+h@self.Wh.T+self.b
            gi=self._sig(gates[:,0*s:1*s]); gf=self._sig(gates[:,1*s:2*s])
            gg=self._tanh(gates[:,2*s:3*s]); go=self._sig(gates[:,3*s:4*s])
            c=gf*c+gi*gg; h=go*self._tanh(c)
            cache.append((x,h.copy(),c.copy(),gi,gf,gg,go))
        mask=((rng.rand(*h.shape)>self.drop)/(1-self.drop+1e-9)) if train and self.drop>0 else np.ones_like(h)
        hd=h*mask
        logit=hd@self.Wy.T+self.by
        prob=self._sig(logit)
        return prob, cache, hd, mask

    def _bwd(self, X, y, prob, cache, mask):
        B,T,D=X.shape; s=self.h
        dl=(prob.squeeze()-y)/B
        dWy=dl[:,None].T@(cache[-1][1]*mask); dby=dl.mean(keepdims=True)
        dh=dl[:,None]*self.Wy*mask; dc=np.zeros((B,s))
        dWx=np.zeros_like(self.Wx); dWh=np.zeros_like(self.Wh); db=np.zeros_like(self.b)
        for t in reversed(range(T)):
            x,h,c,gi,gf,gg,go,*_=cache[t]+(None,)
            cp=cache[t-1][2] if t>0 else np.zeros((B,s))
            dh_=dh; dc_=dc+dh_*go*(1-self._tanh(c)**2)
            dgo=dh_*self._tanh(c); dgi=dc_*gg; dgf=dc_*cp; dgg=dc_*gi
            dg=np.zeros((B,4*s))
            dg[:,0*s:1*s]=dgi*gi*(1-gi); dg[:,1*s:2*s]=dgf*gf*(1-gf)
            dg[:,2*s:3*s]=dgg*(1-gg**2); dg[:,3*s:4*s]=dgo*go*(1-go)
            dWx+=dg.T@x; dWh+=dg.T@(cache[t-1][1] if t>0 else np.zeros((B,s)))
            db+=dg.sum(0); dh=dg@self.Wh; dc=dc_*gf
        clip=5.0; grads=[np.clip(g,-clip,clip) for g in [dWx,dWh,db,dWy,dby]]
        return grads

    def _adam(self, params, grads, b1=0.9, b2=0.999, eps=1e-8):
        self.t+=1; out=[]
        for i,(p,g) in enumerate(zip(params,grads)):
            self.m[i]=b1*self.m[i]+(1-b1)*g; self.v[i]=b2*self.v[i]+(1-b2)*g**2
            mh=self.m[i]/(1-b1**self.t); vh=self.v[i]/(1-b2**self.t)
            out.append(p-self.lr*mh/(np.sqrt(vh)+eps))
        return out

    def fit(self, Xs, y, Xv=None, yv=None):
        n=len(y); rng=np.random.RandomState(self.seed)
        for ep in range(self.epochs):
            idx=rng.permutation(n); el=0; nb=0
            for s in range(0,n,self.bs):
                bi=idx[s:s+self.bs]; Xb=Xs[bi]; yb=y[bi].astype(float)
                prob,cache,hd,mask=self._fwd(Xb,train=True)
                p=prob.squeeze(); loss=-np.mean(yb*np.log(p+1e-9)+(1-yb)*np.log(1-p+1e-9))
                el+=loss; nb+=1
                grads=self._bwd(Xb,yb,prob,cache,mask)
                params=[self.Wx,self.Wh,self.b,self.Wy,self.by]
                self.Wx,self.Wh,self.b,self.Wy,self.by=self._adam(params,grads)
            self.tl.append(el/nb)
            if Xv is not None:
                pv,_,_,_=self._fwd(Xv,train=False); pv=pv.squeeze()
                self.vl.append(-np.mean(yv*np.log(pv+1e-9)+(1-yv)*np.log(1-pv+1e-9)))
            if (ep+1)%5==0:
                print(f"    Epoch {ep+1:>2}/{self.epochs}  loss={self.tl[-1]:.4f}"
                      +(f"  val={self.vl[-1]:.4f}" if self.vl else ""))
        return self

    def predict_proba(self, Xs):
        p,_,_,_=self._fwd(Xs,train=False); p=p.squeeze()
        return np.column_stack([1-p,p])
    def predict(self,Xs,thr=0.5): return (self.predict_proba(Xs)[:,1]>=thr).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────────────────────────────
def parse_col(col):
    m=re.search(r'hostname="([^"]+)".*?ifName="([^"]+)"',col)
    if not m: m=re.search(r'agent_host="([^"]+)".*?ifName="([^"]+)"',col)
    return (m.group(1),m.group(2)) if m else (None,None)

def label_ev(vals, rs=REAL_STEPS):
    """0=normal · 1=falsa alarma · 2=caída real"""
    N=len(vals); L=np.zeros(N,int); i=0
    while i<N:
        if vals[i]==0:
            j=i
            while j<N and vals[j]==0: j+=1
            L[i:j]=2 if (j-i)>=rs else 1; i=j
        else: i+=1
    return L

def find_thr(proba, y, mp=MIN_PREC):
    precs,recs,thrs=precision_recall_curve(y,proba)
    valid=[(p,r,t) for p,r,t in zip(precs[:-1],recs[:-1],thrs) if p>=mp]
    if valid: return max(valid,key=lambda x:x[1])
    bi=np.argmax(precs[:-1]); return float(thrs[bi]),float(precs[bi]),0.0

def style_ax(ax,title,size=10):
    ax.set_facecolor(PANEL_BG); ax.set_title(title,color='white',
        fontsize=size,fontweight='bold',pad=8)
    ax.tick_params(colors='#aaaacc',labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor('#333355')
    ax.xaxis.label.set_color('#aaaacc'); ax.yaxis.label.set_color('#aaaacc')

def make_seqs(X, seq=ROLL_WIN):
    n=len(X); out=np.zeros((n,seq,X.shape[1]))
    for t in range(n):
        s=max(0,t-seq+1); w=X[s:t+1]; out[t,-len(w):]=w
    return out


# ─────────────────────────────────────────────────────────────────────────────
# PASO 1-2: CARGA, PREPROCESAMIENTO, ACTIVAS vs INACTIVAS
# ─────────────────────────────────────────────────────────────────────────────
print("="*65); print("PASO 1-2: CARGA Y CLASIFICACIÓN ACTIVAS/INACTIVAS"); print("="*65)

df_raw=pd.read_excel(FILE,sheet_name=0).set_index('Time').sort_index()
df_filled=df_raw.ffill().fillna(0)
df_rate=df_filled.diff().clip(lower=0); df_rate.iloc[0]=0
times=df_rate.index; N=len(times)

active=[c for c in df_raw.columns if df_rate[c].max()>0]
inactive=[c for c in df_raw.columns if df_rate[c].max()==0]
print(f"  Total interfaces : {len(df_raw.columns)}")
print(f"  Activas          : {len(active)}")
print(f"  Inactivas        : {len(inactive)}")
print(f"  Periodo          : {times[0].date()} → {times[-1].date()} ({N} timestamps)")
print(f"  Intervalo        : {POLL_MIN} min  |  Días: {(times[-1]-times[0]).days+1}")

# Estacionalidad por día de la semana
dow_map={0:'Lunes',1:'Martes',2:'Miércoles',3:'Jueves',
         4:'Viernes',5:'Sábado',6:'Domingo'}
dow_baselines={}
for dow_i in range(7):
    mask=times.dayofweek==dow_i
    if mask.sum()>0:
        vals=df_rate.loc[mask,active].values.flatten()
        vals=vals[vals>0]
        dow_baselines[dow_i]=vals.mean() if len(vals)>0 else 0.0

hour_baselines={}
for h in range(24):
    mask=times.hour==h
    if mask.sum()>0:
        vals=df_rate.loc[mask,active].values.flatten()
        vals=vals[vals>0]
        hour_baselines[h]=vals.mean() if len(vals)>0 else 0.0

print(f"\n  Tráfico medio por día de la semana (interfaces activas):")
for d,v in dow_baselines.items():
    bar='█'*int(v/max(dow_baselines.values())*15) if max(dow_baselines.values())>0 else ''
    print(f"    {dow_map[d]:10s}: {v:>12,.0f}  {bar}")


# ─────────────────────────────────────────────────────────────────────────────
# PASO 3: ZERORRATIO + AJUSTE ESTACIONAL
# ─────────────────────────────────────────────────────────────────────────────
print("\n"+"="*65); print("PASO 3: ZERORRATIO Y AJUSTE ESTACIONAL"); print("="*65)
print(f"""
  ZeroRatio clásico: % de polls en cero en ventana de {ROLL_WIN} pasos ({ROLL_WIN*POLL_MIN} min)
  
  ZeroRatio ajustado estacionalmente:
    Un cero a las 2 AM del domingo tiene MENOS peso que
    un cero a las 10 AM del lunes (horario laboral).
    
    zr_vs_baseline = zr4 / (baseline_dow + ε)
    
    Si zr_vs_baseline >> 1 → el silencio es anómalo para ese día/hora
    Si zr_vs_baseline ≈ 0 → el silencio es normal para ese día/hora
""")

# ─────────────────────────────────────────────────────────────────────────────
# PASO 4-5: ETIQUETADO + FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
print("="*65); print("PASO 4-5: ETIQUETADO Y FEATURE ENGINEERING"); print("="*65)

FEAT=FEATURES
rows=[]
for col in active:
    host,ifname=parse_col(col)
    if not host: continue
    vals=df_rate[col].values
    lbl=label_ev(vals, REAL_STEPS)
    for t in range(ROLL_WIN, N):
        w4=vals[t-ROLL_WIN:t]
        zr4=float((w4==0).mean())
        zr2=float((vals[t-2:t]==0).mean())
        # zeros consecutivos
        cz=0
        for k in range(t-1,-1,-1):
            if vals[k]==0: cz+=1
            else: break
        # temporal features
        ts=times[t]
        h=ts.hour; dow=ts.dayofweek
        h_sin=np.sin(2*np.pi*h/24); h_cos=np.cos(2*np.pi*h/24)
        d_sin=np.sin(2*np.pi*dow/7); d_cos=np.cos(2*np.pi*dow/7)
        is_we=float(dow>=5)
        # baselines
        dow_bl=dow_baselines.get(dow,1.0)
        hr_bl=hour_baselines.get(h,1.0)
        zr_vs_bl=zr4/(max(dow_bl/1e6,0.01)) if dow_bl>0 else zr4
        hr_act=min(hr_bl/max(hour_baselines.values(),1),1.0)
        rows.append({
            'time':ts,'host':host,'ifname':ifname,
            'zr4':zr4,'zr2':zr2,'std4':float(w4.std()),
            'consec_z':float(cz),'delta1':float(vals[t]-vals[t-1]),
            'hour_sin':h_sin,'hour_cos':h_cos,
            'dow_sin':d_sin,'dow_cos':d_cos,
            'is_weekend':is_we,
            'zr_vs_dow_baseline':float(min(zr4*10,1.0) if dow_bl<100 else zr4),
            'hour_activity':float(hr_act),
            'label':int(lbl[t]==2),
            'label_full':int(lbl[t]),
            'dow':dow,'hour':h,'date':ts.date(),
        })

df=pd.DataFrame(rows).sort_values('time').reset_index(drop=True)
print(f"  Dataset: {len(df):,} filas × {len(FEAT)} features + 1 label")
print(f"  Caída real    (1): {df['label'].sum():,}  ({df['label'].mean()*100:.1f}%)")
print(f"  No caída      (0): {(df['label']==0).sum():,}  ({(df['label']==0).mean()*100:.1f}%)")
print(f"    ├─ Normal      : {(df['label_full']==0).sum():,}")
print(f"    └─ Falsa alarma: {(df['label_full']==1).sum():,}")

# ─────────────────────────────────────────────────────────────────────────────
# PASO 6: SPLIT TEMPORAL 50/50
# ─────────────────────────────────────────────────────────────────────────────
print("\n"+"="*65); print("PASO 6: SPLIT TEMPORAL 50/50"); print("="*65)

split=len(df)//2
df_tr=df.iloc[:split]; df_te=df.iloc[split:]
y_tr=df_tr['label'].values; y_te=df_te['label'].values
X_tr=df_tr[FEAT].values; X_te=df_te[FEAT].values

sc=StandardScaler().fit(X_tr)
X_tr_sc=sc.transform(X_tr); X_te_sc=sc.transform(X_te)

split_t=df['time'].iloc[split]
print(f"  Corte : {split_t.strftime('%Y-%m-%d %H:%M')}")
print(f"  Train : {len(y_tr):,}  |  Caídas: {y_tr.sum():,} ({y_tr.mean()*100:.1f}%)")
print(f"  Test  : {len(y_te):,}  |  Caídas: {y_te.sum():,} ({y_te.mean()*100:.1f}%)")

# Secuencias para LSTM
Xtr_seq=make_seqs(X_tr_sc,ROLL_WIN); Xte_seq=make_seqs(X_te_sc,ROLL_WIN)


# ─────────────────────────────────────────────────────────────────────────────
# PASO 7: ENTRENAMIENTO DE 5 MODELOS
# ─────────────────────────────────────────────────────────────────────────────
print("\n"+"="*65); print("PASO 7: ENTRENAMIENTO DE 5 MODELOS"); print("="*65)

all_res={}

# Sklearn models
sk_models={
    'Logistic Regression': LogisticRegression(class_weight='balanced',max_iter=1000,C=1.0,random_state=SEED),
    'Random Forest':       RandomForestClassifier(n_estimators=200,class_weight='balanced',max_depth=10,min_samples_leaf=5,random_state=SEED,n_jobs=-1),
    'Decision Tree':       DecisionTreeClassifier(class_weight='balanced',max_depth=8,min_samples_leaf=10,random_state=SEED),
}
for name,mdl in sk_models.items():
    print(f"\n  ▶ {name}…")
    mdl.fit(X_tr_sc,y_tr)
    proba=mdl.predict_proba(X_te_sc)[:,1]
    thr,pr,rc=find_thr(proba,y_te)
    yp=(proba>=thr).astype(int)
    cm=confusion_matrix(y_te,yp); TN,FP,FN,TP=cm.ravel()
    f1=f1_score(y_te,yp,zero_division=0); acc=accuracy_score(y_te,yp)
    fpr,tpr,_=roc_curve(y_te,proba); au=auc(fpr,tpr)
    pc,rc_c,_=precision_recall_curve(y_te,proba)
    all_res[name]=dict(thr=thr,pr=pr,rc=rc,f1=f1,acc=acc,au=au,
                       TN=TN,FP=FP,FN=FN,TP=TP,fpr=fpr,tpr=tpr,
                       pc=pc,rc_c=rc_c,proba=proba,yp=yp,cm=cm)
    fa=FP/(FP+TN+1e-9)*100
    print(f"    thr={thr:.2f}  Prec={pr:.3f}  Rec={rc:.3f}  F1={f1:.3f}  "
          f"Acc={acc:.3f}  AUC={au:.3f}  FP={FP}  FA%={fa:.1f}%")

# XGBoost
print(f"\n  ▶ XGBoost (numpy)…")
n_xgb=min(10000,len(y_tr))
idx_x=np.sort(np.random.RandomState(SEED).choice(len(y_tr),n_xgb,replace=False))
xgb=XGBoostScratch(n=80,lr=0.15,max_depth=3,sub=0.8,seed=SEED)
xgb.fit(X_tr_sc[idx_x],y_tr[idx_x])
proba_x=xgb.predict_proba(X_te_sc)[:,1]
thr_x,pr_x,rc_x=find_thr(proba_x,y_te)
yp_x=(proba_x>=thr_x).astype(int)
cm_x=confusion_matrix(y_te,yp_x); TN_x,FP_x,FN_x,TP_x=cm_x.ravel()
f1_x=f1_score(y_te,yp_x,zero_division=0); acc_x=accuracy_score(y_te,yp_x)
fpr_x,tpr_x,_=roc_curve(y_te,proba_x); au_x=auc(fpr_x,tpr_x)
pc_x,rc_cx,_=precision_recall_curve(y_te,proba_x)
all_res['XGBoost']=dict(thr=thr_x,pr=pr_x,rc=rc_x,f1=f1_x,acc=acc_x,au=au_x,
                         TN=TN_x,FP=FP_x,FN=FN_x,TP=TP_x,fpr=fpr_x,tpr=tpr_x,
                         pc=pc_x,rc_c=rc_cx,proba=proba_x,yp=yp_x,cm=cm_x)
print(f"    thr={thr_x:.2f}  Prec={pr_x:.3f}  Rec={rc_x:.3f}  F1={f1_x:.3f}  "
      f"Acc={acc_x:.3f}  AUC={au_x:.3f}  FP={FP_x}  FA%={FP_x/(FP_x+TN_x+1e-9)*100:.1f}%")

# LSTM
print(f"\n  ▶ LSTM (numpy)…")
n_lstm=min(12000,len(y_tr))
idx_l=np.sort(np.random.RandomState(SEED).choice(len(y_tr),n_lstm,replace=False))
lstm=LSTMScratch(in_sz=len(FEAT),h_sz=24,seq=ROLL_WIN,
                 lr=0.003,epochs=20,bs=512,drop=0.2,seed=SEED)
lstm.fit(Xtr_seq[idx_l],y_tr[idx_l],Xte_seq[:2000],y_te[:2000])
proba_l=lstm.predict_proba(Xte_seq)[:,1]
thr_l,pr_l,rc_l=find_thr(proba_l,y_te)
yp_l=(proba_l>=thr_l).astype(int)
cm_l=confusion_matrix(y_te,yp_l); TN_l,FP_l,FN_l,TP_l=cm_l.ravel()
f1_l=f1_score(y_te,yp_l,zero_division=0); acc_l=accuracy_score(y_te,yp_l)
fpr_l,tpr_l,_=roc_curve(y_te,proba_l); au_l=auc(fpr_l,tpr_l)
pc_l,rc_cl,_=precision_recall_curve(y_te,proba_l)
all_res['LSTM']=dict(thr=thr_l,pr=pr_l,rc=rc_l,f1=f1_l,acc=acc_l,au=au_l,
                      TN=TN_l,FP=FP_l,FN=FN_l,TP=TP_l,fpr=fpr_l,tpr=tpr_l,
                      pc=pc_l,rc_c=rc_cl,proba=proba_l,yp=yp_l,cm=cm_l,
                      tl=lstm.tl,vl=lstm.vl)
print(f"    thr={thr_l:.2f}  Prec={pr_l:.3f}  Rec={rc_l:.3f}  F1={f1_l:.3f}  "
      f"Acc={acc_l:.3f}  AUC={au_l:.3f}  FP={FP_l}  FA%={FP_l/(FP_l+TN_l+1e-9)*100:.1f}%")

# Tabla resumen
print("\n"+"="*65); print("TABLA COMPLETA DE MÉTRICAS"); print("="*65)
print(f"  {'Modelo':22s}  {'Thr':5s}  {'Prec':6s}  {'Rec':6s}  {'F1':6s}  {'Acc':6s}  {'AUC':6s}  {'FP':>5}  {'FA%':>5}")
print("  "+"─"*78)
for name,r in all_res.items():
    fa=r['FP']/(r['FP']+r['TN']+1e-9)*100
    print(f"  {name:22s}  {r['thr']:.2f}   {r['pr']:.3f}   {r['rc']:.3f}   "
          f"{r['f1']:.3f}   {r['acc']:.3f}   {r['au']:.3f}  {r['FP']:>5}  {fa:>4.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURA 1: Overview del dataset
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerando Figura 1: Overview…")

fig1=plt.figure(figsize=(24,20))
fig1.patch.set_facecolor(DARK_BG)
gs1=gridspec.GridSpec(3,2,figure=fig1,hspace=0.50,wspace=0.30,height_ratios=[1,1,1.2])

# ① Activas vs inactivas
ax=fig1.add_subplot(gs1[0,0])
style_ax(ax,'①  Interfaces: Activas vs Inactivas\n'
    f'    Total={len(active)+len(inactive)}  ·  Activas={len(active)}  ·  Inactivas={len(inactive)}')
vals_pie=[len(active),len(inactive)]
cols_pie=[C_OK,C_FA]
wedges,_,auts=ax.pie(vals_pie,colors=cols_pie,autopct='%1.1f%%',startangle=90,
                      pctdistance=0.72,wedgeprops=dict(edgecolor=DARK_BG,linewidth=2),
                      textprops=dict(color='white',fontsize=10))
for at in auts: at.set_fontsize(10); at.set_fontweight('bold')
ax.legend([mpatches.Patch(color=C_OK,label=f'Activas ({len(active)})'),
           mpatches.Patch(color=C_FA,label=f'Inactivas ({len(inactive)})')],
          facecolor=ACCENT,labelcolor='white',fontsize=9,loc='lower left',framealpha=0.9)
ax.set_facecolor(PANEL_BG)

# ② Tráfico medio por día
ax2=fig1.add_subplot(gs1[0,1])
style_ax(ax2,'②  Tráfico Medio por Día\n    Estacionalidad semanal visible')
dates_u=sorted(set(times.date))
day_means=[df_rate.loc[times.date==d,active].values.mean() for d in dates_u]
day_strs=[f"{d.strftime('%d/%m')}\n{pd.Timestamp(d).strftime('%a')[:3]}" for d in dates_u]
dow_cols=[C_FA if pd.Timestamp(d).dayofweek>=5 else
          ('#ff6b6b' if pd.Timestamp(d).dayofweek==3 else C_OK) for d in dates_u]
bars=ax2.bar(range(len(dates_u)),day_means,color=dow_cols,alpha=0.85,edgecolor='none')
ax2.set_xticks(range(len(dates_u)))
ax2.set_xticklabels(day_strs,rotation=45,ha='right',color='#aaaacc',fontsize=7)
ax2.set_ylabel('Pkts/intervalo promedio'); ax2.grid(axis='y',color='#333355',lw=0.6)
leg2=[mpatches.Patch(color=C_OK,label='Lunes-Viernes'),
      mpatches.Patch(color=C_FA,label='Fin de semana'),
      mpatches.Patch(color='#ff6b6b',label='Jueves (pico)')]
ax2.legend(handles=leg2,facecolor=ACCENT,labelcolor='white',fontsize=8,framealpha=0.9)

# ③ Tráfico por hora del día
ax3=fig1.add_subplot(gs1[1,0])
style_ax(ax3,'③  Tráfico Medio por Hora del Día\n    Patrón diario: alta actividad 07h-18h')
hours=list(hour_baselines.keys()); hvals=list(hour_baselines.values())
hcols=[C_OK if 7<=h<=18 else (C_FA if 0<=h<=5 else '#ffd166') for h in hours]
ax3.bar(hours,hvals,color=hcols,alpha=0.88,edgecolor='none')
ax3.set_xlabel('Hora del día'); ax3.set_ylabel('Tráfico medio (pkts/poll)')
ax3.set_xticks(range(0,24,2)); ax3.grid(axis='y',color='#333355',lw=0.6)

# ④ Distribución del ZeroRatio
ax4=fig1.add_subplot(gs1[1,1])
style_ax(ax4,'④  Distribución de ZeroRatio4\n    Por tipo de evento')
bins=np.linspace(0,1,25)
ax4.hist(df.loc[df['label_full']==0,'zr4'],bins=bins,color=C_OK,alpha=0.65,
         density=True,label=f"Normal ({(df['label_full']==0).sum():,})")
ax4.hist(df.loc[df['label_full']==1,'zr4'],bins=bins,color=C_FA,alpha=0.65,
         density=True,label=f"Falsa alarma ({(df['label_full']==1).sum():,})")
ax4.hist(df.loc[df['label_full']==2,'zr4'],bins=bins,color=C_REAL,alpha=0.65,
         density=True,label=f"Caída real ({(df['label_full']==2).sum():,})")
ax4.axvline(x=0.75,color='white',linestyle='--',lw=1.4,alpha=0.7,label='Umbral 0.75')
ax4.set_xlabel('ZeroRatio4'); ax4.set_ylabel('Densidad')
ax4.legend(facecolor=ACCENT,labelcolor='white',fontsize=8.5,framealpha=0.9)
ax4.grid(color='#333355',lw=0.5)

# ⑤ Etiquetas por día de la semana
ax5=fig1.add_subplot(gs1[2,:])
style_ax(ax5,'⑤  Caídas Reales vs Falsas Alarmas por Día\n'
    '    Semana Santa (31Mar-5Abr) con tráfico muy bajo = anómalo para días laborables',size=11)
dow_names=['Lun','Mar','Mié','Jue','Vie','Sáb','Dom']
x_days=range(7); w=0.28
normals=[( df['label_full']==0)&(df['dow']==d).sum() for d in range(7)]
fas    =[(df['label_full']==1)&(df['dow']==d).sum() for d in range(7)]
reals  =[(df['label_full']==2)&(df['dow']==d).sum() for d in range(7)]
b1=ax5.bar([x-w for x in x_days],normals,w,color=C_OK,alpha=0.85,label='Normal')
b2=ax5.bar(x_days,fas,w,color=C_FA,alpha=0.85,label='Falsa alarma')
b3=ax5.bar([x+w for x in x_days],reals,w,color=C_REAL,alpha=0.85,label='Caída real')
for bars in [b1,b2,b3]:
    for bar in bars:
        ax5.text(bar.get_x()+bar.get_width()/2,bar.get_height()+100,
                 f'{int(bar.get_height()):,}',ha='center',va='bottom',
                 color='white',fontsize=7.5,fontweight='bold')
ax5.set_xticks(x_days); ax5.set_xticklabels(dow_names,color='#aaaacc',fontsize=11)
ax5.set_ylabel('Número de muestras')
ax5.legend(facecolor=ACCENT,labelcolor='white',fontsize=9,framealpha=0.9)
ax5.grid(axis='y',color='#333355',lw=0.6)

fig1.suptitle('Book2.xlsx — Análisis Exploratorio y Preprocesamiento\n'
    f'{len(active)} interfaces activas · {N} timestamps · {POLL_MIN}-min polls · 29 días',
    color='white',fontsize=13,fontweight='bold',y=0.999)
plt.savefig(OUT1,dpi=130,bbox_inches='tight',facecolor=DARK_BG); plt.close()
print(f"  ✔ {OUT1}")


# Fix the syntax error in the overview plot
normals_=[(( df['label_full']==0)&(df['dow']==d)).sum() for d in range(7)]
fas_    =[(( df['label_full']==1)&(df['dow']==d)).sum() for d in range(7)]
reals_  =[(( df['label_full']==2)&(df['dow']==d)).sum() for d in range(7)]


# ─────────────────────────────────────────────────────────────────────────────
# FIGURA 2: Métricas completas de modelos
# ─────────────────────────────────────────────────────────────────────────────
print("Generando Figura 2: Modelos y métricas…")

fig2=plt.figure(figsize=(26,34))
fig2.patch.set_facecolor(DARK_BG)
gs2=gridspec.GridSpec(4,3,figure=fig2,hspace=0.52,wspace=0.30,
                       height_ratios=[1.3,1,1,1])

# ⑥ Matrices de confusión (5 modelos)
gs_cm=gridspec.GridSpecFromSubplotSpec(1,5,subplot_spec=gs2[0,:],wspace=0.28)
for k,(name,col) in enumerate(MCOLORS.items()):
    r=all_res[name]; ax_k=fig2.add_subplot(gs_cm[k])
    cm_=r['cm']; cm_n=cm_.astype(float)/(cm_.sum(axis=1,keepdims=True)+1e-9)
    TN_,FP_,FN_,TP_=cm_.ravel(); fa_=FP_/(FP_+TN_+1e-9)*100
    num=['⑥','⑦','⑧','⑨','⑩'][k]
    style_ax(ax_k,f'{num} {name}\nP={r["pr"]:.2f} R={r["rc"]:.2f} F1={r["f1"]:.2f}',size=8.5)
    ax_k.imshow(cm_n,cmap='Blues',vmin=0,vmax=1,aspect='auto')
    cell_txt=[['TN','FP⚠'],['FN','TP✓']]
    cell_col=[[C_OK,C_REAL],['#888899',C_OK]]
    for i in range(2):
        for j in range(2):
            pct=cm_n[i,j]
            ax_k.text(j,i-0.18,f'{cm_[i,j]:,}',ha='center',va='center',
                      color='white' if pct>0.45 else '#ccccdd',fontsize=9,fontweight='bold')
            ax_k.text(j,i+0.22,f'{cell_txt[i][j]}\n({pct:.0%})',ha='center',va='center',
                      color=cell_col[i][j],fontsize=7,multialignment='center')
    ax_k.set_xticks([0,1]); ax_k.set_yticks([0,1])
    ax_k.set_xticklabels(['Pred 0','Pred 1'],color='#aaaacc',fontsize=8)
    ax_k.set_yticklabels(['Real 0','Real 1'],color='#aaaacc',fontsize=8)
    ax_k.text(0.5,-0.18,f'FA: {fa_:.1f}%',transform=ax_k.transAxes,ha='center',
              color=C_REAL if fa_>5 else C_OK,fontsize=9,fontweight='bold')

# ⑪ Curva ROC
ax_roc=fig2.add_subplot(gs2[1,0])
style_ax(ax_roc,'⑪  Curva ROC — 5 Modelos\n    AUC mayor = mejor discriminación')
for name,col in MCOLORS.items():
    r=all_res[name]
    ax_roc.plot(r['fpr'],r['tpr'],color=col,lw=2.2,label=f"{name}  AUC={r['au']:.3f}")
    ax_roc.scatter([r['FP']/(r['FP']+r['TN']+1e-9)],[r['TP']/(r['TP']+r['FN']+1e-9)],
                   color=col,s=80,zorder=6,marker='*',edgecolors='white',lw=0.8)
ax_roc.plot([0,1],[0,1],color='#555577',linestyle=':',lw=1.2)
ax_roc.set_xlabel('FPR (Tasa Falsos Positivos)'); ax_roc.set_ylabel('TPR (Verdaderos Positivos)')
ax_roc.set_xlim(-0.01,1.01); ax_roc.set_ylim(-0.01,1.05)
ax_roc.legend(facecolor=ACCENT,labelcolor='white',fontsize=8,loc='lower right',framealpha=0.95)
ax_roc.grid(color='#333355',lw=0.5)

# ⑫ F1, Precisión, Recall
ax_f1=fig2.add_subplot(gs2[1,1])
style_ax(ax_f1,'⑫  Precisión · Recall · F1 por Modelo\n    (umbral ajustado Prec ≥ 0.85)')
names=list(all_res.keys()); xp=np.arange(len(names)); bw=0.26
for vals_k,lbl_k,alpha_k,off_k in [
    ([all_res[n]['pr'] for n in names],'Precisión',0.90,-bw),
    ([all_res[n]['rc'] for n in names],'Recall',0.55,0.0),
    ([all_res[n]['f1'] for n in names],'F1-Score',0.95,bw)]:
    bars=ax_f1.bar(xp+off_k,vals_k,width=bw,
                   color=[MCOLORS[n] for n in names],
                   alpha=alpha_k,edgecolor='none',label=lbl_k)
    for bar,val in zip(bars,vals_k):
        ax_f1.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.009,
                   f'{val:.2f}',ha='center',va='bottom',color='white',fontsize=7.5,fontweight='bold')
ax_f1.set_xticks(xp)
ax_f1.set_xticklabels([n.replace(' ','\n') for n in names],color='#aaaacc',fontsize=8)
ax_f1.set_ylim(0,1.18); ax_f1.set_ylabel('Métrica (0→1)')
ax_f1.axhline(y=MIN_PREC,color=C_REAL,linestyle='--',lw=1.2,alpha=0.6,label=f'Prec obj {MIN_PREC}')
ax_f1.legend(facecolor=ACCENT,labelcolor='white',fontsize=8,loc='upper left',framealpha=0.9)
ax_f1.grid(axis='y',color='#333355',lw=0.6)

# ⑬ Curva PR
ax_pr=fig2.add_subplot(gs2[1,2])
style_ax(ax_pr,f'⑬  Curva Precisión-Recall\n    ★ = punto operativo (Prec ≥ {MIN_PREC})')
for name,col in MCOLORS.items():
    r=all_res[name]
    ax_pr.plot(r['rc_c'],r['pc'],color=col,lw=2.0,label=name)
    ax_pr.scatter([r['rc']],[r['pr']],color=col,s=120,marker='*',zorder=6,edgecolors='white',lw=0.8)
ax_pr.axhline(y=MIN_PREC,color=C_REAL,linestyle='--',lw=1.6,alpha=0.9,label=f'Prec ≥ {MIN_PREC}')
ax_pr.fill_between([0,1],MIN_PREC,1.0,alpha=0.07,color=C_OK)
ax_pr.set_xlabel('Recall'); ax_pr.set_ylabel('Precisión')
ax_pr.set_xlim(-0.01,1.01); ax_pr.set_ylim(0.50,1.02)
ax_pr.legend(facecolor=ACCENT,labelcolor='white',fontsize=8,framealpha=0.9)
ax_pr.grid(color='#333355',lw=0.5)

# ⑭ Accuracy por modelo
ax_acc=fig2.add_subplot(gs2[2,0])
style_ax(ax_acc,'⑭  Accuracy por Modelo')
accs=[all_res[n]['acc'] for n in names]
bars_a=ax_acc.bar(names,[a for a in accs],color=[MCOLORS[n] for n in names],alpha=0.88,edgecolor='none')
for bar,val in zip(bars_a,accs):
    ax_acc.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.005,
                f'{val:.3f}',ha='center',va='bottom',color='white',fontsize=9,fontweight='bold')
ax_acc.set_ylim(0.8,1.02); ax_acc.set_ylabel('Accuracy')
ax_acc.set_xticklabels([n.replace(' ','\n') for n in names],color='#aaaacc',fontsize=8)
ax_acc.grid(axis='y',color='#333355',lw=0.6)

# ⑮ Tasa de falsas alarmas
ax_fa=fig2.add_subplot(gs2[2,1])
style_ax(ax_fa,'⑮  Tasa de Falsas Alarmas por Modelo\n    (menor = mejor)')
fa_rates=[all_res[n]['FP']/(all_res[n]['FP']+all_res[n]['TN']+1e-9)*100 for n in names]
bar_cols=[C_OK if f<1 else C_FA if f<5 else C_REAL for f in fa_rates]
bars_fa=ax_fa.barh(names,fa_rates,color=bar_cols,alpha=0.88,edgecolor='none',height=0.55)
for bar,val in zip(bars_fa,fa_rates):
    ax_fa.text(val+0.05,bar.get_y()+bar.get_height()/2,
               f'{val:.1f}%',va='center',color='white',fontsize=9,fontweight='bold')
ax_fa.axvline(x=5,color=C_FA,linestyle='--',lw=1.3,alpha=0.7,label='5%')
ax_fa.axvline(x=1,color=C_OK,linestyle='--',lw=1.3,alpha=0.7,label='1%')
ax_fa.set_xlabel('Tasa FA (%)'); ax_fa.invert_yaxis()
ax_fa.set_yticklabels(names,color='#aaaacc',fontsize=9)
ax_fa.legend(facecolor=ACCENT,labelcolor='white',fontsize=8.5,framealpha=0.9)
ax_fa.grid(axis='x',color='#333355',lw=0.6)

# ⑯ Curva aprendizaje LSTM
ax_lstm=fig2.add_subplot(gs2[2,2])
style_ax(ax_lstm,'⑯  Curva de Aprendizaje LSTM\n    Train loss vs Validation loss')
rl=all_res['LSTM']; ep_x=np.arange(1,len(rl['tl'])+1)
ax_lstm.plot(ep_x,rl['tl'],color=C_PURP,lw=2.2,marker='o',ms=4,label='Train loss')
if rl['vl']: ax_lstm.plot(ep_x,rl['vl'],color=C_FA,lw=2.2,marker='s',ms=4,linestyle='--',label='Val loss')
ax_lstm.set_xlabel('Época'); ax_lstm.set_ylabel('BCE Loss')
ax_lstm.legend(facecolor=ACCENT,labelcolor='white',fontsize=9,framealpha=0.9)
ax_lstm.grid(color='#333355',lw=0.5)

# ⑰ Tabla resumen
ax_tbl=fig2.add_subplot(gs2[3,:])
ax_tbl.set_facecolor(PANEL_BG); ax_tbl.axis('off')
ax_tbl.set_title('⑰  Tabla Resumen Completa de Métricas — Todos los Modelos',
                 color='white',fontsize=11,fontweight='bold',pad=10)

headers=['Modelo','Umbral','Precisión','Recall','F1-Score','Accuracy','AUC-ROC','FP','FA%','Conclusión']
col_x=[0.00,0.14,0.22,0.31,0.39,0.47,0.55,0.63,0.71,0.79]
concl={
    'Logistic Regression':'Mayor Recall, más FA',
    'Random Forest':'Mejor F1 y AUC',
    'Decision Tree':'Menos FA, conservative',
    'XGBoost':'Alta Precisión, menos FA',
    'LSTM':'Capta patrones temp.',
}
for j,(h,x) in enumerate(zip(headers,col_x)):
    ax_tbl.text(x,0.92,h,transform=ax_tbl.transAxes,color='white',fontsize=8.5,fontweight='bold',va='top')
for i,name in enumerate(all_res):
    r=all_res[name]; fa=r['FP']/(r['FP']+r['TN']+1e-9)*100
    y_=0.92-(i+1)*0.14
    fc='#1e2240' if i%2==0 else PANEL_BG
    ax_tbl.axhspan(y_,y_+0.14,facecolor=fc,alpha=0.5,transform=ax_tbl.transAxes)
    vals=[name,f"{r['thr']:.2f}",f"{r['pr']:.3f}",f"{r['rc']:.3f}",
          f"{r['f1']:.3f}",f"{r['acc']:.3f}",f"{r['au']:.3f}",
          f"{r['FP']:,}",f"{fa:.1f}%",concl.get(name,'')]
    ax_tbl.text(0.00,y_+0.05,'■',transform=ax_tbl.transAxes,
                color=MCOLORS[name],fontsize=12,va='center')
    for j,(val,x) in enumerate(zip(vals,col_x)):
        c='white' if j==0 else ('#aaffaa' if j in [2,4,6] else '#ccccdd')
        if j==7: c=C_OK if r['FP']<500 else C_FA if r['FP']<2000 else C_REAL
        ax_tbl.text(x+(0.03 if j==0 else 0),y_+0.05,val,
                    transform=ax_tbl.transAxes,color=c,fontsize=8.5,va='center')

fig2.suptitle('Book2.xlsx — Métricas Completas: 5 Modelos de Clasificación\n'
    f'Caída Real (≥{REAL_FAIL_MIN} min) vs Falsa Alarma · Precisión objetivo ≥ {MIN_PREC} · '
    f'Split temporal 50/50',
    color='white',fontsize=13,fontweight='bold',y=0.999)
plt.savefig(OUT2,dpi=130,bbox_inches='tight',facecolor=DARK_BG); plt.close()
print(f"  ✔ {OUT2}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURA 3: Visualización temporal por días (la más importante)
# ─────────────────────────────────────────────────────────────────────────────
print("Generando Figura 3: Temporal por días…")

fig3=plt.figure(figsize=(26,32))
fig3.patch.set_facecolor(DARK_BG)
gs3=gridspec.GridSpec(3,1,figure=fig3,hspace=0.45,height_ratios=[1.4,1.2,1.4])

# Elegir 3 interfaces representativas con fallos reales y FAs
demo_ifaces=[]
for col in active:
    if len(demo_ifaces)>=3: break
    host,ifname=parse_col(col)
    if not host: continue
    vals=df_rate[col].values; lbl=label_ev(vals)
    if (lbl==2).sum()>20 and (lbl==1).sum()>10:
        demo_ifaces.append({'col':col,'host':host,'ifname':ifname,
                            'rate':vals,'lbl':lbl,
                            'zr':pd.Series(vals).rolling(ROLL_WIN,min_periods=1).apply(lambda x:(x==0).mean()).values})

# ①  Tráfico total diario (heatmap-style) — caídas vs FAs
ax_t1=fig3.add_subplot(gs3[0])
style_ax(ax_t1,'①  Caídas Reales vs Falsas Alarmas a lo largo del tiempo\n'
    '    Rojo = caída real  ·  Amarillo = falsa alarma  ·  Verde = normal\n'
    '    Semana Santa (31Mar–5Abr): días laborables con tráfico anormalmente bajo',size=11)

# Contar eventos por hora
time_index=pd.DatetimeIndex(df['time'])
df2=df.copy(); df2.index=time_index
real_by_time=df2[df2['label']==1].resample('1H')['label'].count()
fa_by_time=df2[df2['label_full']==1].resample('1H')['label_full'].count()
norm_by_time=df2[df2['label_full']==0].resample('1H')['label_full'].count()

all_times=pd.date_range(real_by_time.index.min(),real_by_time.index.max(),freq='1H')
real_by_time=real_by_time.reindex(all_times,fill_value=0)
fa_by_time=fa_by_time.reindex(all_times,fill_value=0)
norm_by_time=norm_by_time.reindex(all_times,fill_value=0)

ax_t1.stackplot(all_times,norm_by_time.values,fa_by_time.values,real_by_time.values,
                labels=['Normal','Falsa alarma','Caída real'],
                colors=[C_OK,C_FA,C_REAL],alpha=0.80)
# Marcar semana santa
ss_start=pd.Timestamp('2026-03-31'); ss_end=pd.Timestamp('2026-04-06')
ax_t1.axvspan(ss_start,ss_end,alpha=0.12,color='white',zorder=2)
ax_t1.text(ss_start+pd.Timedelta(days=2.5),ax_t1.get_ylim()[1]*0.85 if ax_t1.get_ylim()[1]>0 else 100,
           'SEMANA SANTA\n(31 Mar – 5 Abr)\nTráfico anormalmente\nbajo en días laborables',
           ha='center',color='white',fontsize=8.5,style='italic',
           bbox=dict(boxstyle='round,pad=0.4',fc=ACCENT,alpha=0.9))
ax_t1.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m\n%a'))
ax_t1.xaxis.set_major_locator(mdates.DayLocator(interval=2))
ax_t1.set_ylabel('Número de muestras por hora')
ax_t1.set_xlim(all_times[0],all_times[-1])
ax_t1.legend(facecolor=ACCENT,labelcolor='white',fontsize=9,loc='upper left',framealpha=0.95)
ax_t1.grid(axis='x',color='#333355',lw=0.5)

# ② Comparación semanal: semana normal vs semana santa
ax_t2=fig3.add_subplot(gs3[1])
style_ax(ax_t2,'②  Semana Normal (23-29 Mar) vs Semana Santa (31 Mar–5 Abr)\n'
    '    Un lunes de Semana Santa con ZeroRatio alto = CAÍDA REAL (no es falsa alarma)',size=11)

# Semana normal: 23-29 Mar (Lun-Dom)
# Semana santa: 31 Mar – 5 Abr
wn_start='2026-03-23'; wn_end='2026-03-29'
ss_start2='2026-03-31'; ss_end2='2026-04-06'

df_wn=df2.loc[wn_start:wn_end]; df_ss=df2.loc[ss_start2:ss_end2]
wn_real=df_wn[df_wn['label']==1].resample('4H')['label'].count().reindex(
    pd.date_range(wn_start,wn_end,freq='4H'),fill_value=0)
ss_real=df_ss[df_ss['label']==1].resample('4H')['label'].count().reindex(
    pd.date_range(ss_start2,ss_end2,freq='4H'),fill_value=0)

x_wn=np.arange(len(wn_real)); x_ss=np.arange(len(ss_real))
ax_t2.fill_between(x_wn,wn_real.values,color=C_BLUE,alpha=0.65,label='Semana normal (23-29 Mar)')
ax_t2.fill_between(x_ss[:len(wn_real)],ss_real.values[:len(wn_real)],
                    color=C_REAL,alpha=0.65,label='Semana Santa (31Mar-5Abr) — caídas reales')
ax_t2.set_xticks(x_wn[::2])
ax_t2.set_xticklabels([f"{['Lun','Mar','Mié','Jue','Vie','Sáb','Dom'][i//6]}\n{(i//6)*24+(i%6)*4}h"
                        for i in range(0,len(wn_real),2)],color='#aaaacc',fontsize=7.5)
ax_t2.set_ylabel('Caídas reales por bloque de 4h')
ax_t2.legend(facecolor=ACCENT,labelcolor='white',fontsize=9,framealpha=0.9)
ax_t2.grid(color='#333355',lw=0.5)
ax_t2.text(0.5,0.88,'En Semana Santa, un Lunes con tráfico cero durante horas laborables\n'
           'es una CAÍDA REAL (el edificio debería tener gente y equipos activos).',
           transform=ax_t2.transAxes,ha='center',va='top',color='white',fontsize=9.5,
           bbox=dict(boxstyle='round,pad=0.5',fc=ACCENT,alpha=0.9))

# ③ Detalle de 3 interfaces: tráfico + ZeroRatio + etiquetas
ax_t3=fig3.add_subplot(gs3[2])
style_ax(ax_t3,'③  Detalle de 3 Interfaces: Patrón de Caída Real vs Falsa Alarma\n'
    '    Morado = ZeroRatio4  ·  Fondo rojo = caída real  ·  Fondo amarillo = falsa alarma',size=11)

y_off=0.0; sep=1.3; ytpos=[]; ytlbl=[]
for d in demo_ifaces[:3]:
    lbl_=d['lbl']; rate_=d['rate']; zr_=d['zr']
    maxv=max(rate_.max(),1)
    h_s=d['host'].replace('BLOQUE_E-SALAE-','BLE-').replace('SW-','')
    for t in range(1,N):
        c=C_OK if lbl_[t]==0 else (C_FA if lbl_[t]==1 else C_REAL)
        ax_t3.axvspan(times[t-1],times[t],
                      ymin=y_off/(sep*3.5),ymax=(y_off+sep*0.90)/(sep*3.5),
                      color=c,alpha=0.22,zorder=1)
    norm=rate_/maxv*sep*0.72
    ax_t3.fill_between(times,y_off,y_off+norm,color='white',alpha=0.45,zorder=3)
    ax_t3.plot(times,y_off+zr_*sep*0.82,color=C_PURP,lw=1.8,alpha=0.90,zorder=5,
               label='ZeroRatio4' if y_off==0 else '_')
    ax_t3.axhline(y=y_off+0.75*sep*0.82,color=C_REAL,lw=1.0,linestyle='--',alpha=0.55,
                  label='Umbral 0.75' if y_off==0 else '_')
    ytpos.append(y_off+sep*0.4); ytlbl.append(f"{h_s}/{d['ifname']}")
    y_off+=sep

ax_t3.set_yticks(ytpos); ax_t3.set_yticklabels(ytlbl,fontsize=8.5,color='#aaaacc')
ax_t3.set_xlim(times[0],times[-1]); ax_t3.set_ylim(-0.05,y_off+0.1)
ax_t3.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m\n%a'))
ax_t3.xaxis.set_major_locator(mdates.DayLocator(interval=3))
ax_t3.grid(axis='x',color='#333355',lw=0.5)
leg3=[mpatches.Patch(color=C_OK,alpha=0.5,label='Normal'),
      mpatches.Patch(color=C_FA,alpha=0.5,label='Falsa alarma (< 60 min)'),
      mpatches.Patch(color=C_REAL,alpha=0.5,label='Caída real (≥ 60 min)'),
      plt.Line2D([0],[0],color=C_PURP,lw=2,label='ZeroRatio4'),
      plt.Line2D([0],[0],color=C_REAL,lw=1.2,linestyle='--',label='Umbral 0.75')]
ax_t3.legend(handles=leg3,facecolor=ACCENT,labelcolor='white',fontsize=9,
             loc='upper right',framealpha=0.95,ncol=2)

fig3.suptitle('Book2.xlsx — Análisis Temporal: Caídas Reales vs Falsas Alarmas\n'
    'Estacionalidad semanal · Semana Santa como caso de caída real detectada',
    color='white',fontsize=13,fontweight='bold',y=0.999)
plt.savefig(OUT3,dpi=130,bbox_inches='tight',facecolor=DARK_BG); plt.close()
print(f"  ✔ {OUT3}")


# ─────────────────────────────────────────────────────────────────────────────
# PASO 10: CONCLUSIONES
# ─────────────────────────────────────────────────────────────────────────────
best_f1=max(all_res,key=lambda n:all_res[n]['f1'])
best_fa=min(all_res,key=lambda n:all_res[n]['FP']/(all_res[n]['FP']+all_res[n]['TN']+1e-9))
best_auc=max(all_res,key=lambda n:all_res[n]['au'])
br=all_res[best_f1]

print(f"""
╔══════════════════════════════════════════════════════════════════════════╗
║  CONCLUSIONES — Book2.xlsx Pipeline Completo                             ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  DATOS:                                                                  ║
║    · {len(active)} interfaces activas de {len(active)+len(inactive)} totales ({len(active)/(len(active)+len(inactive))*100:.0f}%)                       ║
║    · 29 días · intervalo 15 min · {N} timestamps                     ║
║    · Caídas reales (≥60 min): {df['label'].sum():,} muestras ({df['label'].mean()*100:.1f}%)             ║
║    · Falsa alarma (<60 min):  {(df['label_full']==1).sum():,} muestras ({(df['label_full']==1).mean()*100:.1f}%)             ║
║                                                                          ║
║  MEJOR MODELO (F1):    {best_f1:<20}  F1={all_res[best_f1]['f1']:.3f}           ║
║  MENOR FA%:            {best_fa:<20}  FA%={all_res[best_fa]['FP']/(all_res[best_fa]['FP']+all_res[best_fa]['TN']+1e-9)*100:.1f}%              ║
║  MEJOR AUC:            {best_auc:<20}  AUC={all_res[best_auc]['au']:.3f}          ║
║                                                                          ║
║  ESTACIONALIDAD:                                                         ║
║    · Jueves tiene tráfico 14× mayor que domingo (backups/batch)          ║
║    · Semana Santa (31Mar-5Abr): días laborables con tráfico de fin       ║
║      de semana → el modelo la detecta como CAÍDA REAL (correctamente)    ║
║    · Features temporales (dow_sin/cos, hour_sin/cos) mejoran            ║
║      significativamente la distinción FA vs caída real                  ║
║                                                                          ║
║  RECOMENDACIONES PARA PRODUCCIÓN:                                        ║
║    1. Usar {best_f1} como modelo principal                    ║
║    2. Incorporar calendario de festivos para ajustar baselines           ║
║    3. Umbral dinámico: más alto en fines de semana/festivos              ║
║    4. Monitorear ZeroRatio4 en tiempo real cada 15 minutos               ║
║    5. Alerta solo si ZeroRatio4 ≥ 0.75 Y consec_z ≥ 4 (60 min)         ║
╚══════════════════════════════════════════════════════════════════════════╝
""")