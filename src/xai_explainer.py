import warnings
import logging

# Headless ortam için GUI backend devre dışı
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import shap

from data_loader import IoTDataLoader
from utils import load_config, setup_logging, get_project_path, verify_and_load_model

warnings.filterwarnings("ignore")

# Logger
logger = logging.getLogger("XAI_Pipeline")


# Veri hazırlama

def _load_and_prepare_data(config: dict):
    loader = IoTDataLoader()
    max_files = config["data"].get("max_files_xai", 2)
    logger.info(f"{max_files} dosya yükleniyor...")

    X, y = loader.load_data(max_files=max_files, random_select=True)

    logger.info("Veri ön işleme yapılıyor...")
    X = X.replace([float("inf"), float("-inf")], float("nan")).fillna(0)
    X = X.select_dtypes(include=[float, int])

    le_path = get_project_path("models", "label_encoder.pkl")
    le = verify_and_load_model(le_path)

    # Label drift kontrolü: eğitimde görülmeyen etiketleri filtrele
    known_labels = set(le.classes_)
    unknown_mask = ~y.isin(known_labels)
    n_unknown = unknown_mask.sum()

    if n_unknown > 0:
        unknown_classes = y[unknown_mask].unique().tolist()
        logger.warning(
            f"[LABEL_DRIFT] {n_unknown} satır, modelin eğitimde görmediği "
            f"{len(unknown_classes)} bilinmeyen etiket içeriyor. "
            f"Filtre uygulanıyor → {unknown_classes}"
        )
        valid_mask = ~unknown_mask
        X = X[valid_mask].reset_index(drop=True)
        y = y[valid_mask].reset_index(drop=True)
        logger.info(
            f"[LABEL_DRIFT] Filtreleme sonrası kalan satır sayısı: {len(y)}"
        )

    if len(y) == 0:
        raise ValueError(
            "[LABEL_DRIFT] Kritik Hata: Tüm etiketler bilinmeyen sınıflardan oluşuyor. "
            "Veri seti ile model arasında tam etiket uyumsuzluğu var. "
            "Modeli güncel veri setiyle yeniden eğitin."
        )

    y_encoded = le.transform(y)

    return X, y_encoded, le


# SHAP hesaplama

def _compute_shap_values(model, X_sample, random_state: int) -> tuple:
    logger.info("SHAP TreeExplainer başlatılıyor...")
    explainer = shap.TreeExplainer(model)

    logger.info(
        f"[COMPUTATION] SHAP matrisi hesaplanıyor ({len(X_sample)} örnek, "
        "çok sınıflı dağılım ayrıştırılıyor)..."
    )
    shap_values = explainer.shap_values(X_sample, check_additivity=False)
    return shap_values, explainer


# Görselleştirme

def _normalize_shap_values(shap_values) -> list:
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        return [shap_values[:, :, i] for i in range(shap_values.shape[2])]
    if isinstance(shap_values, list):
        return shap_values
    return shap_values


def _render_and_save_plot(shap_values_list, X_sample, class_names, max_disp: int, output_path: str) -> None:
    logger.info("SHAP grafiği oluşturuluyor...")

    plt.figure(figsize=(22, 12))
    shap.summary_plot(
        shap_values_list,
        X_sample,
        class_names=class_names,
        plot_type="bar",
        show=False,
        plot_size=(22, 12),
        max_display=max_disp,
    )
    plt.title(
        f"NIDS Özellik Önem Sıralaması (İlk {max_disp})",
        fontsize=20,
        pad=40,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"[ARTIFACT] SHAP grafiği kaydedildi: {output_path}")


# Ana akış

def generate_shap_summary() -> None:
    config = load_config()
    logger.info("XAI modülü başlatıldı.")

    # Model yükleme
    model_path = get_project_path("models", "nids_champion_model.pkl")
    logger.info(f"Model yükleniyor: {model_path}")
    trained_model = verify_and_load_model(model_path)
    logger.info(
        f"Model yüklendi. Tip: {type(trained_model).__name__}, "
        f"Ağaç sayısı: {trained_model.n_estimators}"
    )

    # Veri hazırlama
    X, _y_encoded, le = _load_and_prepare_data(config)

    # Şema hizalama
    logger.info("[SCHEMA] Veri matrisi, modelin orijinal eğitim şemasıyla hizalanıyor...")
    if hasattr(trained_model, 'feature_names_in_'):
        expected_cols = list(trained_model.feature_names_in_)
        logger.info(f"[SCHEMA] Beklenen özellik (Feature) sayısı: {len(expected_cols)}")
        
        missing_cols = set(expected_cols) - set(X.columns)
        if missing_cols:
            logger.warning(f"[IMPUTATION] Veri setinde eksik olan {len(missing_cols)} özellik '0' ile dolduruluyor.")
            for col in missing_cols:
                X[col] = 0
                
        # Sütunları eğitim sırasına diz
        X = X[expected_cols]
        logger.info("Şema hizalama tamamlandı.")
    else:
        # Eski sürüm güvenliği
        expected_features = trained_model.n_features_in_
        if X.shape[1] != expected_features:
            raise ValueError(
                f"[ERROR] Özellik sayısı uyumsuz: model {expected_features} bekliyor, "
                f"veri {X.shape[1]} sütun içeriyor."
            )

    # Örneklem seçimi
    sample_size = config["xai"]["sample_size"]
    random_state = config["data"]["random_state"]
    X_sample = X.sample(
        min(sample_size, len(X)),
        random_state=random_state,
    )
    logger.info(f"[SAMPLING] SHAP için {len(X_sample)} satırlık örneklem seçildi.")

    # SHAP hesaplama
    shap_values, _explainer = _compute_shap_values(trained_model, X_sample, random_state)
    shap_values_list = _normalize_shap_values(shap_values)

    # Grafik oluşturma
    max_disp = config["xai"]["max_display_features"]
    output_path = get_project_path("data", "shap_summary_plot.png")
    _render_and_save_plot(shap_values_list, X_sample, le.classes_, max_disp, output_path)
    logger.info("XAI analizi tamamlandı.")


if __name__ == "__main__":
    setup_logging()
    generate_shap_summary()