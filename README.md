# TC5061 - Experimentación base con MLflow

Regresión logística para predecir **default en préstamos SBA** (Kaggle: *Should This Loan be Approved or Denied?*), convertida de notebook a script reproducible y parametrizable, con cada corrida registrada en MLflow.

**Autor:** Dan García

## Estructura

| Archivo | Para qué sirve |
|---|---|
| `train.py` | Carga, limpieza, entrenamiento, evaluación y registro en MLflow |
| `run_experiments.sh` | Corre las 10 configuraciones del reporte |
| `comparar_corridas.py` | Tabla comparativa de corridas + verificación de reproducibilidad |
| `notebooks/TC5061_EDA_GarciaDan.ipynb` | Solo exploración de datos |
| `requirements.txt` | Dependencias con versiones |

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

El dataset se descarga automáticamente de Kaggle (`kagglehub`) la primera vez y se guarda en `data/SBAnational.csv`. Si ya tienes el CSV, colócalo en esa ruta o usa `--data-path`.

## Ejecución

```bash
# Valores por defecto
python train.py

# Cambiando hiperparámetros
python train.py --C 0.01 --class-weight balanced --solver lbfgs --max-iter 1000 --seed 42

# Todos los argumentos
python train.py --help
```

| Argumento | Default | Descripción |
|---|---|---|
| `--C` | 1.0 | Inverso de la regularización |
| `--class-weight` | balanced | `balanced` o `none` |
| `--solver` | lbfgs | `lbfgs`, `liblinear`, `newton-cholesky`, `saga` |
| `--max-iter` | 1000 | Iteraciones máximas del solver |
| `--n-sample` | 100000 | Muestra estratificada (0 = dataset completo) |
| `--test-size` | 0.2 | Proporción de prueba |
| `--seed` | 42 | Semilla global |
| `--experiment-name` | TC5061_SBA_Default_LogReg | Experimento de MLflow |
| `--tracking-uri` | sqlite:///mlflow.db | Backend de tracking |

## Experimentos y comparación

```bash
bash run_experiments.sh        # 10 corridas + tabla comparativa
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Abrir http://localhost:5000 → experimento `TC5061_SBA_Default_LogReg` → seleccionar corridas → **Compare**.

## Qué registra cada corrida

- **Parámetros:** hiperparámetros, semilla, tamaño de muestra/split, número de variables y huella MD5 del dataset.
- **Métricas:** ROC-AUC, PR-AUC, recall, precision, F1 y accuracy (train y test), iteraciones del solver, convergencia y tiempo.
- **Artefactos:** matriz de confusión, curvas ROC/PR, coeficientes, `metricas.json` y bitácora de limpieza.
- **Modelo:** pipeline completo (preprocesamiento + clasificador) con firma, vía `mlflow.sklearn.log_model`.
- **Tags:** versiones de Python, scikit-learn, pandas y MLflow.

## Reproducibilidad

- `fijar_semilla()` fija `random`, `numpy` y `PYTHONHASHSEED`; además la muestra, el split y el modelo reciben `random_state=seed`.
- La limpieza es determinística y todo lo que aprende de los datos (imputación, escalado, one-hot) vive dentro del `Pipeline`, ajustado solo con train.
- `run_experiments.sh` corre la configuración por defecto dos veces y `comparar_corridas.py` confirma que las métricas son idénticas.

## Dependencias

Para regenerar `requirements.txt` con las versiones exactas del entorno:

```bash
pip freeze | grep -iE "^(mlflow|scikit-learn|pandas|numpy|matplotlib|skops|kagglehub|seaborn)==" > requirements.txt
```
