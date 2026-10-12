#!/usr/bin/env bash
# Corre la batería de experimentos del reporte. Todas quedan en el mismo experimento de MLflow.
# Uso: bash run_experiments.sh
set -e

# 1-2. Valores por defecto, dos veces (prueba de reproducibilidad)
python train.py
python train.py

# 3-5. Regularización (C)
python train.py --C 0.001
python train.py --C 0.01
python train.py --C 100

# 6. Sin balanceo de clases
python train.py --class-weight none

# 7. Pocas iteraciones (el solver no alcanza a converger)
python train.py --max-iter 10

# 8. Otro solver
python train.py --solver liblinear

# 9. Otra semilla (cambia la muestra y el split)
python train.py --seed 7

# 10. Dataset completo (~896K registros)
python train.py --n-sample 0

python comparar_corridas.py
