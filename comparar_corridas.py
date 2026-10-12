"""
Resume las corridas del experimento en una tabla y verifica la reproducibilidad
(corridas con los mismos argumentos deben tener exactamente las mismas métricas).

Uso:
    python comparar_corridas.py
"""
import argparse

import mlflow
import pandas as pd

PARAMS = ["C", "class_weight", "solver", "max_iter", "n_sample", "seed"]
METRICAS = ["test_roc_auc", "test_pr_auc", "test_recall", "test_precision", "test_f1",
            "train_roc_auc", "n_iter", "tiempo_entrenamiento_s"]


def main():
    p = argparse.ArgumentParser(description="Tabla comparativa de corridas en MLflow")
    p.add_argument("--experiment-name", default="TC5061_SBA_Default_LogReg")
    p.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    p.add_argument("--salida", default="comparacion_corridas.csv")
    args = p.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    runs = mlflow.search_runs(experiment_names=[args.experiment_name],
                              order_by=["metrics.test_pr_auc DESC"])

    cols = (["tags.mlflow.runName"] + [f"params.{c}" for c in PARAMS]
            + [f"metrics.{m}" for m in METRICAS])
    tabla = runs[cols].copy()
    tabla.columns = ["run"] + PARAMS + METRICAS
    tabla[METRICAS] = tabla[METRICAS].round(4)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    print(f"\n{len(tabla)} corridas en '{args.experiment_name}' (ordenadas por test_pr_auc):\n")
    print(tabla.to_string(index=False))
    tabla.to_csv(args.salida, index=False)
    print(f"\nTabla guardada en {args.salida}")

    # Reproducibilidad: mismas combinaciones de parámetros -> mismas métricas
    repetidas = runs.groupby([f"params.{c}" for c in PARAMS]).filter(lambda g: len(g) > 1)
    if repetidas.empty:
        print("\nNo hay corridas repetidas para verificar reproducibilidad.")
    else:
        for llaves, g in repetidas.groupby([f"params.{c}" for c in PARAMS]):
            iguales = g[[f"metrics.{m}" for m in METRICAS[:6]]].nunique().max() == 1
            print(f"\nReproducibilidad {dict(zip(PARAMS, llaves))}: "
                  f"{len(g)} corridas -> {'métricas idénticas' if iguales else 'DIFERENTES'}")


if __name__ == "__main__":
    main()
