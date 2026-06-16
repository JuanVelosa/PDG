# Detección Temprana de Caídas de Red mediante ZeroRatio y Aprendizaje Automático

**Autor:** Juan Manuel Velosa Valencia  
**Programa:** Ingeniería Telemática — Universidad Icesi, Cali, Colombia  
**Directores:** Nicolás Salazar · Andres Navarro (`anavarro@icesi.edu.co`)  
**Institución objetivo:** Universidad Central del Valle del Cauca (UCEVA)

---

## Descripción

Este repositorio contiene el pipeline completo de clasificación de caídas de red en switches de distribución universitarios, desarrollado como proyecto de grado. El sistema distingue entre **fallas reales de infraestructura** y **silencios esperados por estacionalidad institucional** (festivos, fines de semana, horario nocturno) a partir de series de tiempo de tráfico recolectadas mediante SNMP.

El indicador central es el **Zero-Ratio (ZeroRatio)**: la proporción de lecturas de tráfico nulo dentro de una ventana deslizante de cuatro polls (60 minutos). Este indicador, combinado con contexto temporal y comparación contra baselines históricos, alimenta cinco clasificadores de aprendizaje automático.

---

## Estructura del repositorio

```
.
├── codigo.py                   # Pipeline principal (isp_book2_pipeline.py)
├── Book2.xlsx                  # Dataset de entrada (no incluido — ver sección Datos)
├── outputs/
│   ├── book2_overview.png      # Figura de análisis exploratorio
│   ├── book2_modelos.png       # Métricas por modelo (Precision · Recall · F1)
│   ├── book2_modelos_f1_solo.png
│   ├── book2_temporal.png      # Visualización temporal por días
│   └── book2_figure3_detalle_3interfaces.png
└── README.md
```

---

## Datos

El dataset (`Book2.xlsx`) contiene registros SNMP de la métrica `ifInBroadcastPkts` de ~1 000 interfaces de switches de distribución del campus UCEVA, muestreadas cada **15 minutos** entre el **18 de marzo y el 17 de abril de 2026** (~30 días).

> El archivo de datos no se distribuye públicamente. Para reproducir los experimentos, coloca `Book2.xlsx` en el directorio raíz junto a `codigo.py`.

---

## Instalación

**Requiere Python ≥ 3.9**

```bash
pip install numpy pandas matplotlib scikit-learn openpyxl
```

No se requieren frameworks de deep learning: el LSTM y XGBoost están implementados desde cero en NumPy.

---

## Uso

```bash
python codigo.py
```

El script ejecuta los 10 pasos del pipeline en secuencia y guarda todas las figuras en la carpeta `outputs/`.

---

## Pipeline — Pasos

| Paso | Descripción |
|------|-------------|
| 1–2 | Carga del Excel, preprocesamiento y clasificación de interfaces activas vs. inactivas |
| 3 | Cálculo del ZeroRatio (`zr4`, `zr2`) con ajuste estacional |
| 4 | Etiquetado: falla real vs. estacionalidad (**sin lookforward**) |
| 5 | Feature engineering — 12 características en 3 grupos (ver abajo) |
| 6 | Split temporal 50/50 entrenamiento / validación |
| 7 | Entrenamiento de 5 modelos de clasificación |
| 8 | Métricas: matriz de confusión, F1, ROC-AUC (umbral fijo = 0.70) |
| 9 | Visualización temporal por días |
| 10 | Impresión de conclusiones en consola |

---

## Características (Feature Engineering)

Las 12 características se organizan en tres grupos:

**Grupo 1 — Históricas de actividad**
- `zr4` — ZeroRatio en los últimos 4 polls (60 min)
- `zr2` — ZeroRatio en los últimos 2 polls (30 min)
- `std4` — Desviación estándar del tráfico en los últimos 4 polls
- `consec_z` — Ceros consecutivos acumulados hasta el timestamp actual
- `delta1` — Variación de tráfico respecto al poll anterior

**Grupo 2 — Estacionalidad temporal**
- `hour_sin`, `hour_cos` — Hora del día codificada cíclicamente
- `dow_sin`, `dow_cos` — Día de la semana codificado cíclicamente
- `is_weekend` — Indicador binario sábado/domingo

**Grupo 3 — Comparación con baselines históricos**
- `zr_vs_dow_baseline` — ZeroRatio actual vs. actividad esperada para ese día de la semana
- `hour_activity` — Nivel de actividad esperado para la hora actual (normalizado a [0, 1])

Todas las características se escalan con `StandardScaler` ajustado exclusivamente con datos de entrenamiento.

---

## Modelos evaluados

Todos los modelos se evalúan con umbral de decisión fijo de **0.70** y partición temporal 50/50.

| Modelo | Implementación | Precisión | Recall | F1 |
|--------|---------------|-----------|--------|-----|
| Regresión Logística | scikit-learn | 0.89 | 0.88 | **0.89** |
| LSTM | NumPy (BPTT + Adam) | 0.97 | 0.73 | 0.83 |
| Árbol de Decisión | scikit-learn | 0.95 | 0.61 | 0.74 |
| Random Forest | scikit-learn | 0.73 | 0.59 | 0.73 |
| XGBoost | NumPy (gradient boosting) | 1.00 | 0.41 | 0.58 ⚠️ |

> XGBoost no superó el umbral objetivo de F1 ≥ 0.70 por su bajo recall (0.41), lo que lo hace inadecuado para monitoreo operativo donde la omisión de fallos tiene consecuencias directas.

---

## Definición de Falla Real

Un timestamp se etiqueta como **falla real** si cumple **todas** las condiciones siguientes:

1. `ZeroRatio ≥ 0.75` en ventana de 4 polls
2. `ceros consecutivos ≥ 4 polls` (60 minutos de silencio acumulado)
3. El timestamp **no** es festivo, fin de semana ni fuera del horario laboral (07h–18h)

Festivos considerados (Colombia 2026, dentro del rango del dataset):
- 23 de marzo — San José (trasladado a lunes)
- 2 de abril — Jueves Santo
- 3 de abril — Viernes Santo

---

## Parámetros de configuración

Los principales parámetros se encuentran en la sección `CONFIG` del script:

```python
POLL_MIN     = 15      # Minutos por intervalo de polling
ROLL_WIN     = 4       # Ventana de ZeroRatio (4 polls = 60 min)
REAL_STEPS   = 4       # Ceros consecutivos mínimos para falla real
THRESHOLD    = 0.70    # Umbral de probabilidad para clasificación binaria
ZR_FAIL_THR  = 0.75    # Umbral de ZeroRatio
WORK_START_H = 7       # Inicio horario laboral
WORK_END_H   = 18      # Fin horario laboral
SEED         = 42
```

---

## Referencia

Si utilizas este código o la metodología en tu trabajo, por favor cita el artículo correspondiente:

> J. M. Velosa Valencia, "Detección Temprana de Caídas de Red en Switches de Distribución Universitarios Mediante Análisis de ZeroRatio y Aprendizaje Automático," Proyecto de Grado, Programa de Ingeniería Telemática, Universidad Icesi, Cali, Colombia, 2026.

---

## Licencia

Este repositorio se distribuye con fines académicos. Para cualquier uso externo, contactar al autor o a los directores del proyecto.
