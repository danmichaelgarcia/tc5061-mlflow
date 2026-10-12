"""
TC5061 - Código de experimentación base con MLflow
Alumno: Dan García

Entrena una regresión logística para predecir default en préstamos SBA y registra
cada corrida (parámetros, métricas, artefactos y modelo) en MLflow.

Uso:
    python train.py                                   # valores por defecto
    python train.py --C 0.01 --class-weight none --max-iter 500 --seed 7
    python train.py --help                            # lista completa de argumentos
"""
import argparse
import hashlib
import json
import os
import platform
import random
import shutil
import tempfile
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # sin ventana: las gráficas solo se guardan como archivo
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
                             accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# Aviso informativo de MLflow sobre columnas enteras en la firma; no afecta el resultado
warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer")

DATASET_ID = "mirbektoktogaraev/should-this-loan-be-approved-or-denied"
TARGET = "default"

# Grupos de variables según el tratamiento que necesitan (mismas decisiones del notebook)
LOG_COLS = ["NoEmp", "RetainedJob", "GrAppv"]                     # muy asimétricas: log + escalado
NUM_COLS = ["Term", "ApprovalFY", "sba_ratio", "es_franquicia"]   # escalado directo
CAT_COLS = ["State", "BankState", "NewExist", "UrbanRural", "RevLineCr", "LowDoc", "sector"]

# Columnas que salen del modelo y por qué
COLS_EXCLUIR = {
    "LoanNr_ChkDgt": "Identificador",
    "Name": "Identificador",
    "City": "Alta cardinalidad (~32K valores)",
    "Zip": "Alta cardinalidad (~33K valores)",
    "Bank": "Alta cardinalidad (~5.8K valores)",
    "NAICS": "Reemplazada por 'sector'",
    "FranchiseCode": "Reemplazada por 'es_franquicia'",
    "ApprovalDate": "Redundante con ApprovalFY",
    "DisbursementDate": "Ocurre después de la aprobación",
    "DisbursementGross": "Redundante con GrAppv (corr 0.97)",
    "CreateJob": "Redundante con RetainedJob (corr 0.99)",
    "BalanceGross": "Casi constante",
    "ChgOffDate": "Fuga de información",
    "ChgOffPrinGr": "Fuga de información",
    "MIS_Status": "Reemplazada por 'default'",
    "SBA_Appv": "Redundante con GrAppv; su información queda en 'sba_ratio'",
}


# ---------------------------------------------------------------------------
# 1. Argumentos de línea de comandos
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Entrena la regresión logística baseline (SBA default) y la registra en MLflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Hiperparámetros del modelo
    p.add_argument("--C", type=float, default=1.0,
                   help="Inverso de la fuerza de regularización (más chico = más regularización)")
    p.add_argument("--class-weight", choices=["balanced", "none"], default="balanced",
                   help="Peso de clases para manejar el desbalance")
    p.add_argument("--solver", choices=["lbfgs", "liblinear", "newton-cholesky", "saga"],
                   default="lbfgs", help="Algoritmo de optimización")
    p.add_argument("--max-iter", type=int, default=1000,
                   help="Iteraciones máximas del solver")
    # Datos y reproducibilidad
    p.add_argument("--n-sample", type=int, default=100_000,
                   help="Tamaño de la muestra estratificada (0 = dataset completo)")
    p.add_argument("--test-size", type=float, default=0.2, help="Proporción del set de prueba")
    p.add_argument("--seed", type=int, default=42, help="Semilla aleatoria global")
    # Rutas y MLflow
    p.add_argument("--data-path", default="data/SBAnational.csv",
                   help="CSV local; si no existe se descarga de Kaggle")
    p.add_argument("--experiment-name", default="TC5061_SBA_Default_LogReg")
    p.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    p.add_argument("--run-name", default=None,
                   help="Nombre de la corrida (por defecto se arma con los hiperparámetros)")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# 2. Reproducibilidad
# ---------------------------------------------------------------------------
def fijar_semilla(seed):
    """Fija todas las fuentes de aleatoriedad que usa el script."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    # Los splits y el modelo además reciben random_state=seed de forma explícita


def hash_archivo(path, n=12):
    """Huella MD5 del dataset: si el archivo cambia, la corrida lo deja registrado."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()[:n]


# ---------------------------------------------------------------------------
# 3. Datos
# ---------------------------------------------------------------------------
def cargar_datos(data_path):
    """Lee el CSV local; si no existe, lo descarga de Kaggle una sola vez."""
    data_path = Path(data_path)
    if not data_path.exists():
        import kagglehub  # solo se necesita si no hay copia local
        data_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path = Path(kagglehub.dataset_download(DATASET_ID))
        shutil.copy(cache_path / "SBAnational.csv", data_path)
    # low_memory=False para evitar que pandas infiera tipos distintos por bloques
    return pd.read_csv(data_path, low_memory=False), data_path


def a_numero(serie):
    """Montos con formato moneda ("$60,000.00") -> float."""
    return serie.astype(str).str.replace(r"[\$,\s]", "", regex=True).astype(float)


def agrupar_raros(serie, minimo=1000):
    """Conserva códigos con al menos `minimo` registros; el resto pasa a 'Otro'."""
    serie = serie.fillna("Desconocido").astype(str).str.strip()
    frecuencia = serie.map(serie.value_counts())
    return serie.where(frecuencia >= minimo, "Otro")


def limpiar_datos(df_raw):
    """
    Limpieza determinística (no aprende nada de los datos), igual que el Paso 3 del notebook.
    Regresa el dataset limpio y la bitácora de decisiones.
    """
    bitacora = []

    def registrar(paso, decision, d):
        bitacora.append({"paso": paso, "decision": decision,
                         "filas": len(d), "columnas": d.shape[1]})

    df = df_raw.copy()
    registrar("0. Inicio", "Dataset original sin cambios", df)

    # Sin variable objetivo no se puede entrenar
    n_sin_target = int(df["MIS_Status"].isna().sum())
    df = df.dropna(subset=["MIS_Status"])
    df[TARGET] = (df["MIS_Status"] == "CHGOFF").astype(int)
    registrar("1. Variable objetivo", f"Se eliminan {n_sin_target:,} registros sin MIS_Status", df)

    # Duplicados ignorando el ID (mismo préstamo con distinto número)
    dup = df.drop(columns="LoanNr_ChkDgt").duplicated()
    df = df[~dup]
    registrar("2. Duplicados", f"Se eliminan {int(dup.sum()):,} duplicados (ignorando el ID)", df)

    # Tipos: montos y año fiscal
    for col in ["GrAppv", "SBA_Appv"]:
        df[col] = a_numero(df[col])
    df["ApprovalFY"] = pd.to_numeric(df["ApprovalFY"].astype(str).str.extract(r"(\d{4})")[0])
    registrar("3. Tipos", "Montos de texto a float y ApprovalFY a número", df)

    # Códigos: sector, categorías 'No definido', franquicia y códigos raros
    df["sector"] = np.where(df["NAICS"] == 0, "No registrado", df["NAICS"].astype(str).str[:2])
    df["NewExist"] = df["NewExist"].map({1: "Existente", 2: "Nueva"}).fillna("No definido")
    df["UrbanRural"] = df["UrbanRural"].map({1: "Urbano", 2: "Rural", 0: "No definido"})
    df["es_franquicia"] = (df["FranchiseCode"] > 1).astype(int)
    for col in ["RevLineCr", "LowDoc"]:
        df[col] = agrupar_raros(df[col])
    for col in ["State", "BankState"]:
        df[col] = df[col].fillna("Desconocido")
    registrar("4. Códigos", "Sector, códigos 0 como 'No definido', franquicia, códigos raros", df)

    # Plazo de 0 meses no es posible
    n_term0 = int((df["Term"] == 0).sum())
    df = df[df["Term"] > 0]
    registrar("5. Outliers", f"Se eliminan {n_term0:,} préstamos con plazo 0", df)

    # Proporción garantizada por la SBA y selección final de columnas
    df["sba_ratio"] = df["SBA_Appv"] / df["GrAppv"]
    df = df.drop(columns=list(COLS_EXCLUIR))
    registrar("6. Selección", f"Se excluyen {len(COLS_EXCLUIR)} columnas y se crea 'sba_ratio'", df)

    return df, bitacora


def preparar_splits(df_clean, n_sample, test_size, seed):
    """Muestra estratificada (opcional) y split train/test estratificado."""
    if n_sample and n_sample < len(df_clean):
        df_modelo, _ = train_test_split(df_clean, train_size=n_sample,
                                        stratify=df_clean[TARGET], random_state=seed)
    else:
        df_modelo = df_clean

    X = df_modelo.drop(columns=TARGET)
    y = df_modelo[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


# ---------------------------------------------------------------------------
# 4. Modelo
# ---------------------------------------------------------------------------
def construir_pipeline(C, class_weight, solver, max_iter, seed):
    """Preprocesamiento + regresión logística en un solo Pipeline (se ajusta solo con train)."""
    log_pipe = Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("escalar", StandardScaler()),
    ])
    num_pipe = Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("escalar", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputar", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocesador = ColumnTransformer([
        ("log", log_pipe, LOG_COLS),
        ("num", num_pipe, NUM_COLS),
        ("cat", cat_pipe, CAT_COLS),
    ])
    clasificador = LogisticRegression(
        C=C,
        class_weight=None if class_weight == "none" else class_weight,
        solver=solver,
        max_iter=max_iter,
        random_state=seed,
    )
    return Pipeline([("preprocesamiento", preprocesador), ("clasificador", clasificador)])


def entrenar(modelo, X_train, y_train):
    """Entrena y regresa el tiempo y las iteraciones que usó el solver."""
    inicio = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # la no convergencia se registra como métrica, no como aviso
        modelo.fit(X_train, y_train)
    n_iter = int(np.max(modelo.named_steps["clasificador"].n_iter_))
    return time.time() - inicio, n_iter


# ---------------------------------------------------------------------------
# 5. Evaluación
# ---------------------------------------------------------------------------
def evaluar(modelo, X, y, prefijo):
    """Métricas principales para un set de datos, con prefijo train_/test_."""
    pred = modelo.predict(X)
    proba = modelo.predict_proba(X)[:, 1]
    m = {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
    }
    return {f"{prefijo}_{k}": float(v) for k, v in m.items()}


def guardar_graficas(modelo, X_test, y_test, out_dir):
    """Matriz de confusión, curvas ROC/PR y coeficientes. Regresa las rutas generadas."""
    out_dir = Path(out_dir)
    rutas = []

    # Matriz de confusión
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(modelo, X_test, y_test, display_labels=["Pagado", "Default"],
                                          cmap="Blues", values_format=",", ax=ax)
    ax.set_title("Matriz de confusión (test)")
    rutas.append(out_dir / "matriz_confusion.png")
    fig.savefig(rutas[-1], dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Curvas ROC y Precision-Recall
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    RocCurveDisplay.from_estimator(modelo, X_test, y_test, name="LogReg", ax=axes[0])
    axes[0].plot([0, 1], [0, 1], "r--", label="Azar")
    axes[0].set_title("Curva ROC")
    axes[0].legend()
    PrecisionRecallDisplay.from_estimator(modelo, X_test, y_test, name="LogReg", ax=axes[1])
    axes[1].axhline(y_test.mean(), color="red", ls="--", label=f"Piso ({y_test.mean():.3f})")
    axes[1].set_title("Curva Precision-Recall")
    axes[1].legend()
    rutas.append(out_dir / "curvas_roc_pr.png")
    fig.savefig(rutas[-1], dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Coeficientes: positivos aumentan el riesgo, negativos lo reducen
    nombres = modelo.named_steps["preprocesamiento"].get_feature_names_out()
    coefs = pd.Series(modelo.named_steps["clasificador"].coef_[0], index=nombres)
    top = pd.concat([coefs.nlargest(10), coefs.nsmallest(10)]).sort_values()
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top.index, top.values, color=np.where(top.values > 0, "#c0392b", "#2e86c1"))
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Top 10 variables que aumentan (rojo) y reducen (azul) el riesgo")
    rutas.append(out_dir / "coeficientes.png")
    fig.savefig(rutas[-1], dpi=120, bbox_inches="tight")
    plt.close(fig)

    return rutas


# ---------------------------------------------------------------------------
# 6. Orquestación + MLflow
# ---------------------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)
    fijar_semilla(args.seed)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    # Datos
    df_raw, data_path = cargar_datos(args.data_path)
    df_clean, bitacora = limpiar_datos(df_raw)
    X_train, X_test, y_train, y_test = preparar_splits(df_clean, args.n_sample, args.test_size, args.seed)

    # Modelo
    modelo = construir_pipeline(args.C, args.class_weight, args.solver, args.max_iter, args.seed)
    tiempo, n_iter = entrenar(modelo, X_train, y_train)

    # Métricas
    metricas = {
        **evaluar(modelo, X_test, y_test, "test"),
        **evaluar(modelo, X_train, y_train, "train"),
        "n_iter": n_iter,
        "convergio": int(n_iter < args.max_iter),
        "tiempo_entrenamiento_s": tiempo,
    }

    params = {
        "modelo": "LogisticRegression",
        "C": args.C,
        "class_weight": args.class_weight,
        "solver": args.solver,
        "max_iter": args.max_iter,
        "n_sample": args.n_sample,
        "test_size": args.test_size,
        "seed": args.seed,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": int(modelo.named_steps["preprocesamiento"].get_feature_names_out().shape[0]),
        "dataset": DATASET_ID,
        "dataset_md5": hash_archivo(data_path),
    }

    run_name = args.run_name or f"logreg_C{args.C}_{args.class_weight}_{args.solver}_it{args.max_iter}_s{args.seed}_n{args.n_sample}"

    with mlflow.start_run(run_name=run_name) as run, tempfile.TemporaryDirectory() as tmp:
        mlflow.set_tags({
            "curso": "TC5061", "autor": "Dan García", "etapa": "experimentacion",
            "python": platform.python_version(), "sklearn": sklearn.__version__,
            "pandas": pd.__version__, "mlflow": mlflow.__version__,
        })

        # Parámetros y métricas
        for k, v in params.items():
            mlflow.log_param(k, v)
        for k, v in metricas.items():
            mlflow.log_metric(k, v)

        # Artefactos: gráficas, métricas en JSON y bitácora de limpieza
        for ruta in guardar_graficas(modelo, X_test, y_test, tmp):
            mlflow.log_artifact(str(ruta), artifact_path="graficas")

        ruta_json = Path(tmp) / "metricas.json"
        ruta_json.write_text(json.dumps({"params": params, "metricas": metricas}, indent=2, ensure_ascii=False))
        mlflow.log_artifact(str(ruta_json), artifact_path="resultados")

        ruta_bitacora = Path(tmp) / "bitacora_limpieza.csv"
        pd.DataFrame(bitacora).to_csv(ruta_bitacora, index=False)
        mlflow.log_artifact(str(ruta_bitacora), artifact_path="documentacion")

        # Modelo: pipeline completo (preprocesamiento + clasificador) con firma
        # skops (formato de MLflow 3.x) requiere declarar numpy.dtype como tipo confiable
        firma = infer_signature(X_train, modelo.predict(X_train))
        mlflow.sklearn.log_model(sk_model=modelo, name="modelo", signature=firma,
                                 input_example=X_train.head(5),
                                 skops_trusted_types=["numpy.dtype"])

        run_id = run.info.run_id

    print(f"\nCorrida: {run_name} | run_id: {run_id}")
    print(f"test_roc_auc={metricas['test_roc_auc']:.4f} | test_pr_auc={metricas['test_pr_auc']:.4f} | "
          f"test_recall={metricas['test_recall']:.4f} | test_precision={metricas['test_precision']:.4f} | "
          f"n_iter={n_iter} | {tiempo:.1f}s")


if __name__ == "__main__":
    main()
