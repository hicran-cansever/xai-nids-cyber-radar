import pandas as pd
import numpy as np
import time
import warnings
import joblib
import os
import logging
from typing import Tuple, Dict, Any

import optuna
from optuna.samplers import TPESampler

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import IoTDataLoader
from utils import load_config, setup_logging, get_project_path, save_model_with_hash

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Logger
logger = logging.getLogger("ModelTrainer")


# Veri hazırlama

def _filter_rare_classes(X: pd.DataFrame, y: pd.Series, test_size: float) -> Tuple[pd.DataFrame, pd.Series]:
    """Stratify split'te temsil edilemeyecek kadar az örneği olan sınıfları filtreler.

    Bir sınıfın test ve eğitim setinin her ikisinde de yer alabilmesi için
    gereken minimum örnek sayısı ``ceil(1 / test_size) + 1`` formülüyle
    dinamik olarak hesaplanır.  Sabit bir eşik (örn. ``< 2``) kullanmak,
    küçük sınıfların yalnızca eğitim ya da yalnızca test setine düşmesine
    ve ``LabelEncoder.transform`` sırasında ``ValueError`` alınmasına
    yol açar.

    Args:
        X: Özellik matrisi.
        y: Hedef serisi.
        test_size: Train/test bölme oranı (örn. 0.2).

    Returns:
        Nadir sınıflar çıkarılmış ``(X, y)`` çifti.
    """
    import math
    min_required = math.ceil(1 / test_size) + 1
    class_counts = y.value_counts()
    rare_classes = class_counts[class_counts < min_required].index

    if len(rare_classes) > 0:
        logger.warning(
            f"[FİLTRE] {len(rare_classes)} nadir sınıf çıkarılıyor "
            f"(min_required={min_required}): {rare_classes.tolist()}"
        )
        valid_mask = ~y.isin(rare_classes)
        X = X[valid_mask].reset_index(drop=True)
        y = y[valid_mask].reset_index(drop=True)

    return X, y


def prepare_and_clean_data(
    config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, LabelEncoder]:
    """Veriyi yükler, temizler, nadir sınıfları filtreler, böler ve LabelEncoder uygular.

    Veri Sızıntısını (Data Leakage) önlemek için LabelEncoder yalnızca
    eğitim verisi (y_train) üzerinde eğitilir (fit); test verisi (y_test)
    üzerinde ise yalnızca dönüştürme (transform) yapılır.

    Args:
        config: Proje konfigürasyon sözlüğü.

    Returns:
        ``(X_train, X_test, y_train_encoded, y_test_encoded, label_encoder)``
        beşlisi.
    """
    loader = IoTDataLoader()
    X, y = loader.load_data(
        max_files=config["data"]["max_files_train"],
        random_select=True,
    )

    logger.info("Veri ön işleme ve temizlik yapılıyor...")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    X = X.select_dtypes(include=[np.number])

    # Dinamik nadir sınıf filtresi (split öncesi)
    X, y = _filter_rare_classes(X, y, test_size=config["data"]["test_size"])

    # Veri Sızıntısı Çözümü: önce böl, sonra encode et
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y,
    )

    # LabelEncoder SADECE y_train üzerinde fit edilir
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    logger.info(
        f"[GÜVENLİK] LabelEncoder SADECE y_train ile eğitildi. "
        f"Sınıflar: {le.classes_}"
    )

    # LabelEncoder kaydı — SHA-256 imzasıyla güvenli serileştirme
    models_dir = get_project_path("models")
    os.makedirs(models_dir, exist_ok=True)
    le_path = get_project_path("models", "label_encoder.pkl")
    save_model_with_hash(le, le_path, compress=0)
    logger.info("[GÜVENLİK] LabelEncoder imzalı biçimde kaydedildi.")

    return X_train, X_test, y_train_encoded, y_test_encoded, le


# Hiperparametre optimizasyonu

def _build_objective(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    random_state: int,
) -> Any:
    """Optuna için objective fonksiyonunu closure olarak üretir.

    Her trial, RandomForestClassifier'ı farklı hiperparametrelerle eğitir
    ve 3-katlı çapraz doğrulama (CV) ile Macro F1 skorunu döndürür.
    CV, test setini kirletmeden güvenilir bir tahmin sağlar.

    Args:
        X_train: Eğitim özellik matrisi.
        y_train: Kodlanmış eğitim etiket dizisi.
        random_state: Tekrarlanabilirlik için seed değeri.

    Returns:
        Optuna trial nesnesini kabul eden ve Macro F1 döndüren callable.
    """
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
            "max_depth": trial.suggest_categorical("max_depth", [10, 20, 30, None]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "class_weight": "balanced",
            "random_state": random_state,
            "n_jobs": -1,
        }
        model = RandomForestClassifier(**params)
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=3, scoring="f1_macro", n_jobs=-1
        )
        return cv_scores.mean()

    return objective


def optimize_hyperparameters(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Optuna TPE Sampler ile en iyi RandomForest hiperparametrelerini arar.

    Config dosyasında ``model.optuna_trials`` anahtarı varsa o kadar
    deneme yapılır; yoksa varsayılan olarak 20 trial çalıştırılır.

    Args:
        X_train: Eğitim özellik matrisi.
        y_train: Kodlanmış eğitim etiket dizisi.
        config: Proje konfigürasyon sözlüğü.

    Returns:
        En yüksek CV F1 skorunu veren hiperparametre sözlüğü.
    """
    n_trials = config.get("model", {}).get("optuna_trials", 20)
    random_state = config["data"]["random_state"]

    logger.info(
        f"[OPTUNA] Hiperparametre optimizasyonu başlatılıyor "
        f"({n_trials} trial, TPE Sampler)..."
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=random_state),
        study_name="nids_rf_optimization",
    )
    study.optimize(
        _build_objective(X_train, y_train, random_state),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best_params = study.best_params
    best_params["class_weight"] = "balanced"
    best_params["random_state"] = random_state
    best_params["n_jobs"] = -1

    logger.info(
        f"[OPTUNA] Optimizasyon tamamlandı. "
        f"En iyi CV F1: {study.best_value:.4f} | "
        f"Parametreler: {best_params}"
    )
    return best_params


# Performans raporlama

def _log_full_report(
    y_test: np.ndarray,
    preds: np.ndarray,
    y_test_proba: np.ndarray,
    le: LabelEncoder,
    elapsed_time: float,
) -> None:
    """Test seti üzerinde kapsamlı performans metriklerini loglar ve kaydeder.

    Hesaplanan metrikler:
    - Macro / Weighted F1, Precision, Recall
    - Sınıf başına (per-class) Precision, Recall, F1
    - Macro-average One-vs-Rest ROC-AUC
    - Confusion Matrix (PNG olarak diske yazılır)

    Args:
        y_test: Gerçek test etiketleri (kodlanmış).
        preds: Modelin tahminleri (kodlanmış).
        y_test_proba: Modelin sınıf olasılıkları (predict_proba çıktısı).
        le: Eğitilmiş LabelEncoder (sınıf adlarını çözmek için).
        elapsed_time: Eğitim süresi (saniye).
    """
    class_names = le.classes_

    # --- Özet Metrikler ---
    macro_f1 = f1_score(y_test, preds, average="macro")
    weighted_f1 = f1_score(y_test, preds, average="weighted")
    macro_precision = precision_score(y_test, preds, average="macro", zero_division=0)
    macro_recall = recall_score(y_test, preds, average="macro", zero_division=0)

    logger.info("=" * 60)
    logger.info("          KAPSAMLI PERFORMANS RAPORU")
    logger.info("=" * 60)
    logger.info(f"  Eğitim Süresi       : {elapsed_time:.2f} saniye")
    logger.info(f"  Macro    F1 Skoru   : {macro_f1:.4f}")
    logger.info(f"  Weighted F1 Skoru   : {weighted_f1:.4f}")
    logger.info(f"  Macro Precision     : {macro_precision:.4f}")
    logger.info(f"  Macro Recall        : {macro_recall:.4f}")

    # --- ROC-AUC (One-vs-Rest, Macro) ---
    try:
        y_test_bin = label_binarize(y_test, classes=np.arange(len(class_names)))
        if y_test_bin.shape[1] > 1:
            roc_auc = roc_auc_score(
                y_test_bin, y_test_proba, average="macro", multi_class="ovr"
            )
            logger.info(f"  Macro ROC-AUC (OvR) : {roc_auc:.4f}")
        else:
            logger.info("  ROC-AUC hesaplanamadı (tek sınıf tespit edildi).")
    except Exception as exc:
        logger.warning(f"  ROC-AUC hesaplanamadı: {exc}")

    # --- Sınıf Başına Detaylı Rapor ---
    logger.info("-" * 60)
    logger.info("  Sınıf Başına Detaylı Rapor (Per-Class):")
    logger.info("-" * 60)
    report = classification_report(
        y_test, preds, target_names=class_names, zero_division=0
    )
    for line in report.splitlines():
        logger.info(f"  {line}")
    logger.info("=" * 60)

    # --- Confusion Matrix (PNG) ---
    _save_confusion_matrix(y_test, preds, class_names)


def _save_confusion_matrix(
    y_test: np.ndarray,
    preds: np.ndarray,
    class_names: np.ndarray,
) -> None:
    """Confusion Matrix'i normalize ederek PNG dosyasına kaydeder.

    Normalize edilmiş matris (oransal), sınıf boyutları arasındaki
    dengesizliği görsel olarak daha iyi yansıtır.

    Args:
        y_test: Gerçek etiketler (kodlanmış).
        preds: Tahmin edilen etiketler (kodlanmış).
        class_names: İnsan okunabilir sınıf adları dizisi.
    """
    cm = confusion_matrix(y_test, preds)
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig_size = max(10, len(class_names))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("Normalize Edilmiş Confusion Matrix", fontsize=14, pad=15)
    ax.set_ylabel("Gerçek Sınıf", fontsize=12)
    ax.set_xlabel("Tahmin Edilen Sınıf", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    output_path = get_project_path("data", "confusion_matrix.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[RAPOR] Confusion Matrix kaydedildi: {output_path}")


# Ana eğitim akışı

def train_and_evaluate() -> None:
    """Modeli eğitir, değerlendirir ve kaydeder.

    Akış:
    1. Veri hazırlama ve bölme (``prepare_and_clean_data``).
    2. Optuna ile hiperparametre optimizasyonu (``optimize_hyperparameters``).
    3. En iyi parametrelerle nihai model eğitimi.
    4. Kapsamlı performans raporlama (``_log_full_report``).
    5. Modelin diske kaydedilmesi.

    Config dosyasında ``model.optuna_trials: 0`` ayarlanırsa
    optimizasyon atlanır ve config'deki sabit parametreler kullanılır.
    """
    config = load_config()
    logger.info("Model eğitimi başlatılıyor.")

    X_train, X_test, y_train, y_test, le = prepare_and_clean_data(config)

    # Hiperparametre optimizasyonu
    n_trials = config.get("model", {}).get("optuna_trials", 20)

    if n_trials > 0:
        best_params = optimize_hyperparameters(X_train, y_train, config)
    else:
        # Optuna devre dışı: config'deki sabit değerleri kullan
        logger.info("[OPTUNA] Devre dışı (optuna_trials=0). Config parametreleri kullanılıyor.")
        best_params = {
            "n_estimators": config["model"]["n_estimators"],
            "max_depth": config["model"]["max_depth"],
            "class_weight": config["model"]["class_weight"],
            "random_state": config["data"]["random_state"],
            "n_jobs": config["model"]["n_jobs"],
        }

    # Model eğitimi
    logger.info(f"Model eğitiliyor. Parametreler: {best_params}")
    start_time = time.time()

    best_rf = RandomForestClassifier(**best_params)
    best_rf.fit(X_train, y_train)

    elapsed_time = time.time() - start_time

    # Tahmin
    preds = best_rf.predict(X_test)
    y_test_proba = best_rf.predict_proba(X_test)

    # Raporlama
    _log_full_report(y_test, preds, y_test_proba, le, elapsed_time)

    # Model kaydı
    models_dir = get_project_path("models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = get_project_path("models", "nids_champion_model.pkl")
    save_model_with_hash(best_rf, model_path, compress=3)
    logger.info(f"Model kaydedildi: {model_path}")


if __name__ == "__main__":
    setup_logging()
    train_and_evaluate()