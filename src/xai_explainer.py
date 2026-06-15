import warnings
import logging

# CRITICAL: Grafik çizilirken arka planda pencere açılmasını (VS Code kilitlenmesini) önler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import shap

from data_loader import IoTDataLoader
from utils import load_config, setup_logging, get_project_path, verify_and_load_model

warnings.filterwarnings("ignore")

# --- KURUMSAL LOGLAMA KÖPRÜSÜ ---
logger = logging.getLogger("XAI_Pipeline")


# ---------------------------------------------------------------------------
# KATMAN 1: Veri Hazırlama
# ---------------------------------------------------------------------------

def _load_and_prepare_data(config: dict):
    loader = IoTDataLoader()
    max_files = config["data"].get("max_files_xai", 2)
    logger.info(f"[MEMORY] RAM koruması için {max_files} dosyalık veri havuzu yükleniyor...")

    X, y = loader.load_data(max_files=max_files, random_select=True)

    logger.info("[PROCESS] Veri ön işleme ve matris dönüşümleri yapılıyor...")
    X = X.replace([float("inf"), float("-inf")], float("nan")).fillna(0)
    X = X.select_dtypes(include=[float, int])

    le_path = get_project_path("models", "label_encoder.pkl")
    le = verify_and_load_model(le_path)

    # --- KRİTİK: ETİKET KAYMA ZIRHI (Label Drift Guard) ---
    # LabelEncoder yalnızca eğitim verisindeki sınıfları tanır.
    # Yeni bir CSV'de modelin hiç görmediği bir saldırı etiketi (örn. "Zero-Day-Attack")
    # varsa le.transform(y) → ValueError fırlatır ve XAI motoru çöker.
    # Bu blok, bilinmeyen etiketleri transform öncesinde acımadan filtreler.
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


# ---------------------------------------------------------------------------
# KATMAN 2: SHAP Hesaplama
# ---------------------------------------------------------------------------

def _compute_shap_values(model, X_sample, random_state: int) -> tuple:
    logger.info("[COMPUTATION] SHAP TreeExplainer başlatılıyor (Champion model üzerinde)...")
    explainer = shap.TreeExplainer(model)

    logger.info(
        f"[COMPUTATION] SHAP matrisi hesaplanıyor ({len(X_sample)} örnek, "
        "çok sınıflı dağılım ayrıştırılıyor)..."
    )
    shap_values = explainer.shap_values(X_sample, check_additivity=False)
    return shap_values, explainer


# ---------------------------------------------------------------------------
# KATMAN 3: Görselleştirme
# ---------------------------------------------------------------------------

def _normalize_shap_values(shap_values) -> list:
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        return [shap_values[:, :, i] for i in range(shap_values.shape[2])]
    if isinstance(shap_values, list):
        return shap_values
    return shap_values


def _render_and_save_plot(shap_values_list, X_sample, class_names, max_disp: int, output_path: str) -> None:
    logger.info("[VISUALIZATION] Grafik çizimi diske yazılıyor (Yerleşim ve Estetik Düzeltmeler)...")

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
        f"Nihai NIDS Karar Parametreleri — Champion Model "
        f"(Gerçek Ağ İmzaları, İlk {max_disp})",
        fontsize=20,
        pad=40,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"[ARTIFACT] SHAP grafiği kaydedildi: {output_path}")


# ---------------------------------------------------------------------------
# KATMAN 4: Orkestrasyon
# ---------------------------------------------------------------------------

def generate_shap_summary() -> None:
    config = load_config()
    logger.info("[INIT] XAI Modülü (Champion Model Tabanlı, Sızıntı Korumalı) Başlatıldı...")

    # --- 1. Champion Modeli Yükle ---
    model_path = get_project_path("models", "nids_champion_model.pkl")
    logger.info(f"[ARTIFACT] Champion model yükleniyor ve imzası doğrulanıyor: {model_path}")
    champion_model = verify_and_load_model(model_path)
    logger.info(
        f"[LOAD] Champion model başarıyla yüklendi. "
        f"Tip: {type(champion_model).__name__}, "
        f"Ağaç sayısı: {champion_model.n_estimators}"
    )

    # --- 2. Veriyi Hazırla ---
    X, _y_encoded, le = _load_and_prepare_data(config)

    # --- 3. KRİTİK YAMA: ŞEMA UYUMU VE KİLİDİ (SCHEMA ALIGNMENT) ---
    logger.info("[SCHEMA] Veri matrisi, modelin orijinal eğitim şemasıyla hizalanıyor...")
    if hasattr(champion_model, 'feature_names_in_'):
        expected_cols = list(champion_model.feature_names_in_)
        logger.info(f"[SCHEMA] Beklenen özellik (Feature) sayısı: {len(expected_cols)}")
        
        missing_cols = set(expected_cols) - set(X.columns)
        if missing_cols:
            logger.warning(f"[IMPUTATION] Veri setinde eksik olan {len(missing_cols)} özellik '0' ile dolduruluyor.")
            for col in missing_cols:
                X[col] = 0
                
        # Sütunları modelin ezberlediği sıraya TAM OLARAK diz ve fazlalıkları at
        X = X[expected_cols]
        logger.info("[SCHEMA] Özellik matrisi, model şemasına %100 uyumlu hale getirildi.")
    else:
        # Eski sürüm güvenliği
        expected_features = champion_model.n_features_in_
        if X.shape[1] != expected_features:
            raise ValueError(
                f"[ERROR] Özellik sayısı uyumsuz: model {expected_features} bekliyor, "
                f"veri {X.shape[1]} sütun içeriyor."
            )

    # --- 4. Örneklem Seç ---
    sample_size = config["xai"]["sample_size"]
    random_state = config["data"]["random_state"]
    X_sample = X.sample(
        min(sample_size, len(X)),
        random_state=random_state,
    )
    logger.info(f"[SAMPLING] SHAP için {len(X_sample)} satırlık örneklem seçildi.")

    # --- 5. SHAP Hesapla ---
    shap_values, _explainer = _compute_shap_values(champion_model, X_sample, random_state)
    shap_values_list = _normalize_shap_values(shap_values)

    # --- 6. Grafik Oluştur ve Kaydet ---
    max_disp = config["xai"]["max_display_features"]
    output_path = get_project_path("data", "shap_summary_plot.png")
    _render_and_save_plot(shap_values_list, X_sample, le.classes_, max_disp, output_path)
    logger.info("[TERMINATION] XAI analizi başarıyla tamamlandı.")


if __name__ == "__main__":
    setup_logging()
    generate_shap_summary()