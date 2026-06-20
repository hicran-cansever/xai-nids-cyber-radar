import streamlit as st
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from typing import Tuple, Any, Dict, List
from data_loader import IoTDataLoader
from utils import (
    load_config,
    get_project_path,
    check_file_exists,
    verify_and_load_model,
    SecurityError,
)

warnings.filterwarnings("ignore")

# Sayfa ayarları
st.set_page_config(page_title="NIDS CYBER RADAR", page_icon="⚡", layout="wide")

# Tema (CSS)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    .stApp { background-color: #0f172a; color: #f8fafc; font-family: 'Inter', sans-serif; }
    h1, h2, h3, h4, h5, h6, p, div { font-family: 'Inter', sans-serif !important; }
    h1 {
        background: -webkit-linear-gradient(45deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
        padding-bottom: 0.5rem;
    }
    [data-testid="stMetricValue"] { color: #38bdf8 !important; font-weight: 800 !important; font-size: 2rem !important; }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.05em; font-size: 0.75rem !important; }
    [data-testid="stSidebar"] { background-color: #1e293b; border-right: 1px solid #334155; }
    hr { border-color: #334155 !important; }
    .stButton>button {
        background: linear-gradient(135deg, #2563eb, #4f46e5) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        padding: 0.5rem 1rem !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1), 0 4px 6px -2px rgba(0,0,0,0.05) !important;
        background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
    }
    .stAlert { border-radius: 8px !important; border: none !important;
               box-shadow: 0 1px 3px 0 rgba(0,0,0,0.1), 0 1px 2px 0 rgba(0,0,0,0.06) !important; }
</style>
""", unsafe_allow_html=True)

# Sabitler

PREFERRED_DISPLAY_COLS: Dict[str, str] = {
    "Header_Length": "HEADER_LEN",
    "TCP": "TCP_FLAGS",
    "syn_count": "SYN_COUNT",
    "Tot size": "TOTAL_SIZE",
}

_config = load_config()
BENIGN_CLASS_KEYWORD: str = _config.get("labels", {}).get("benign_keyword", "BENIGN")
XAI_TOP_N_FEATURES: int = _config.get("xai", {}).get("top_n_local_features", 12)


# Kaynak yükleme

@st.cache_resource(show_spinner="Model yükleniyor...")
def load_models() -> Tuple[Any, Any]:
    model_path = get_project_path("models", "nids_champion_model.pkl")
    le_path = get_project_path("models", "label_encoder.pkl")
    try:
        model = verify_and_load_model(model_path)
        le = verify_and_load_model(le_path)
        return model, le
    except SecurityError as exc:
        st.error(f"⚠️ **GÜVENLİK İHLALİ**\n\n{exc}")
        st.stop()
    except Exception as exc:
        st.error(f"Model yüklenirken hata: {exc}")
        st.stop()


@st.cache_resource(show_spinner="SHAP başlatılıyor...")
def load_shap_explainer(_model: Any) -> shap.TreeExplainer:
    return shap.TreeExplainer(_model)


@st.cache_data(show_spinner="Trafik verisi okunuyor...")
def load_sample_traffic() -> Tuple[pd.DataFrame, pd.Series]:
    config = load_config()
    max_files = config.get("data", {}).get("max_files_xai", 1)
    loader = IoTDataLoader()
    X, y = loader.load_data(max_files=max_files, random_select=True)
    X = X.select_dtypes(include=["number"]).fillna(0)
    return X, y


# XAI fonksiyonları

def _align_schema(live_packet: pd.DataFrame, model: Any) -> pd.DataFrame:
    """Canlı ağ paketini, modelin eğitimde öğrendiği 38 sütunluk şablona uydurur."""
    aligned_packet = live_packet.copy()
    if hasattr(model, 'feature_names_in_'):
        expected_cols = list(model.feature_names_in_)
        missing_cols = set(expected_cols) - set(aligned_packet.columns)
        for col in missing_cols:
            aligned_packet[col] = 0
        aligned_packet = aligned_packet[expected_cols]
    return aligned_packet

def explain_prediction(
    explainer: shap.TreeExplainer,
    aligned_packet: pd.DataFrame,
    predicted_class_idx: int,
    top_n: int = 12,
) -> Tuple[List[Tuple[str, float]], np.ndarray]:
    """Tek paket için SHAP değerlerini hesaplar."""
    shap_values = explainer.shap_values(aligned_packet, check_additivity=False)
    
    # Çoklu sınıf ayrıştırma
    if isinstance(shap_values, list):
        class_shap = shap_values[predicted_class_idx][0]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        class_shap = shap_values[0, :, predicted_class_idx]
    else:
        class_shap = np.asarray(shap_values).flatten()

    feature_names = aligned_packet.columns.tolist()
    paired = list(zip(feature_names, class_shap))
    sorted_pairs = sorted(paired, key=lambda x: abs(x[1]), reverse=True)
    return sorted_pairs[:top_n], class_shap


def render_shap_chart(top_features: List[Tuple[str, float]], prediction_label: str) -> None:
    if not top_features:
        st.warning("SHAP değerleri hesaplanamadı.")
        return

    names = [f[0] for f in reversed(top_features)]
    values = [f[1] for f in reversed(top_features)]
    
    # Pozitif = kararı destekler, negatif = zayıflatır
    colors = ["#ef4444" if v > 0 else "#3b82f6" for v in values]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.45)))
    fig.patch.set_facecolor("#1e293b")
    ax.set_facecolor("#1e293b")

    bars = ax.barh(names, values, color=colors, edgecolor="none", height=0.6)

    for bar, val in zip(bars, values):
        ax.text(
            val + (0.005 if val >= 0 else -0.005),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=10,
            color="#f8fafc",
            fontweight='bold'
        )

    ax.axvline(0, color="#64748b", linewidth=1.5, linestyle="--")
    ax.set_xlabel(f"SHAP Değeri (Modelin '{prediction_label}' Kararına Etkisi)", color="#cbd5e1", fontsize=10, fontweight='bold')
    ax.set_title("Paket Parametrelerinin Karar Ağırlıkları (XAI)", color="#f8fafc", fontsize=13, fontweight="bold", pad=15)
    ax.tick_params(colors="#cbd5e1", labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# Yardımcı fonksiyonlar

def get_system_logs(tail_lines: int = 10) -> str:
    log_path = get_project_path("logs", "system.log")
    if not check_file_exists(log_path):
        return "[SYS_LOG_EMPTY] Sunucu logları henüz başlatılmadı..."
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return "".join(lines[-tail_lines:])

def get_display_columns(df: pd.DataFrame) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for col, label in PREFERRED_DISPLAY_COLS.items():
        if col in df.columns:
            result[col] = label
    if not result:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        for col in numeric_cols[:4]:
            result[col] = col.upper().replace(" ", "_")
    return result

def _pick_random_packet(X_traffic: pd.DataFrame, rng: np.random.Generator) -> Tuple[pd.DataFrame, int]:
    idx = int(rng.integers(0, len(X_traffic)))
    return X_traffic.iloc[[idx]], idx


# Ana arayüz

st.markdown("<h1>⚡ NIDS_CYBER_RADAR_v2.0 ⚡</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #94a3b8;'>"
    "Gerçek zamanlı ağ paketi inceleme ve XAI karar gerekçesi."
    "</p>",
    unsafe_allow_html=True,
)
st.divider()

try:
    model, le = load_models()
    X_traffic, y_traffic = load_sample_traffic()
    explainer = load_shap_explainer(model)
except Exception as exc:
    st.error(f"[!] FATAL_ERROR: Sistem başlatılamadı. Detay: {exc}")
    st.stop()

if "rng_seed" not in st.session_state:
    st.session_state["rng_seed"] = int(np.random.randint(0, 2**31))

rng = np.random.default_rng(st.session_state["rng_seed"])

with st.sidebar:
    st.markdown("<h2 style='text-align:center; color: #f8fafc;'>SYS_CONTROL</h2>", unsafe_allow_html=True)
    st.divider()
    st.markdown("> **STATUS:** `ONLINE_🟢`")
    st.markdown("> **ENGINE:** `RandomForest`")
    st.markdown(f"> **PACKET_POOL:** `{len(X_traffic):,}`")
    st.markdown(f"> **XAI_ENGINE:** `SHAP_TreeExplainer_🟢`")
    st.divider()
    scan_button = st.button(">> INJECT_PACKET_AND_SCAN <<", use_container_width=True)
    if scan_button:
        st.session_state["rng_seed"] = int(np.random.randint(0, 2**31))
        rng = np.random.default_rng(st.session_state["rng_seed"])

if scan_button:
    with st.spinner("[*] TRAFIK_ANALIZ_EDILIYOR..."):
        time.sleep(0.5)

        # 1. Paket Seçimi
        live_packet, random_idx = _pick_random_packet(X_traffic, rng)
        actual_label = y_traffic.iloc[random_idx]

        st.markdown("### 📡 [1] YAKALANAN AĞ PAKETİ PARAMETRELERİ")
        display_cols = get_display_columns(live_packet)
        if display_cols:
            cols = st.columns(len(display_cols))
            for i, (col_name, col_label) in enumerate(display_cols.items()):
                cols[i].metric(label=col_label, value=f"{live_packet[col_name].values[0]:.1f}")

        # 2. Şema Uyumu ve Tahmin
        aligned_packet = _align_schema(live_packet, model)
        prediction_encoded = model.predict(aligned_packet)
        predicted_class_idx: int = int(prediction_encoded[0])
        prediction_label: str = le.inverse_transform(prediction_encoded)[0]
        prediction_proba: np.ndarray = model.predict_proba(aligned_packet)[0]
        confidence: float = float(prediction_proba[predicted_class_idx]) * 100

        st.divider()
        st.markdown("### 🧠 [2] YAPAY ZEKA KARAR MOTORU")

        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.info(f"**[ GERÇEK ETİKET (Ground Truth) ]**\n\n> {actual_label}")
        with res_col2:
            is_benign = BENIGN_CLASS_KEYWORD in prediction_label.upper()
            if is_benign:
                st.success(f"**[ YZ TAHMİNİ ] (Zararsız)**\n\n> {prediction_label}")
            else:
                st.error(f"**[ YZ TAHMİNİ ] (⚠️ TEHLİKE)**\n\n> {prediction_label}")
        with res_col3:
            st.metric(label="MODEL GÜVENİ (Confidence)", value=f"%{confidence:.1f}")

        if prediction_label == actual_label:
            st.success(">> EŞLEŞME BAŞARILI : Model doğru tespit yaptı. [SİSTEM GÜVENDE]")
        else:
            st.warning(">> TESPİT UYUŞMAZLIĞI : Model farklı bir saldırı türüne veya False Positive karara yöneldi.")

        # 3. XAI - Kara Kutu Açıklaması
        st.divider()
        st.markdown("### 🔬 [3] XAI - MODEL BU KARARI NEDEN VERDİ?")

        icon = "🟢" if is_benign else "🔴"
        verdict_text = "zararsız" if is_benign else "**TEHDİT (Saldırı)**"

        st.markdown(
            f"{icon} Yapay zeka bu paketi **{prediction_label}** olarak etiketledi. "
            f"Aşağıdaki grafik, modelin bu teşhisi koyarken **hangi ağ özelliklerine bakarak** ikna olduğunu gösterir.\n\n"
            f"> 🔴 **Kırmızı (Pozitif) Çubuklar** → Modelin **'{prediction_label}'** kararını GÜÇLENDİREN (Destekleyen) özelliklerdir.\n\n"
            f"> 🔵 **Mavi (Negatif) Çubuklar** → Modelin aklını karıştıran veya diğer saldırı türlerine yönlendiren özelliklerdir."
        )

        with st.spinner("[*] SHAP_DEGERLERI_HESAPLANIYOR..."):
            try:
                top_features, all_shap = explain_prediction(explainer, aligned_packet, predicted_class_idx, top_n=XAI_TOP_N_FEATURES)
                render_shap_chart(top_features, prediction_label)

                st.markdown("**📋 Kararı Belirleyen En Kritik 3 Ağ Özelliği (Kanıtlar):**")
                top3_cols = st.columns(3)
                for i, (feat_name, shap_val) in enumerate(top_features[:3]):
                    direction = "🔴 Kararı Güçlendiriyor" if shap_val > 0 else "🔵 Kararı Zayıflatıyor"
                    raw_value = live_packet[feat_name].values[0] if feat_name in live_packet.columns else "N/A"
                    top3_cols[i].metric(
                        label=feat_name,
                        value=f"{raw_value:.2f}" if isinstance(raw_value, float) else str(raw_value),
                        delta=f"Etki Skoru: {shap_val:+.4f}",
                        delta_color="inverse" if shap_val < 0 else "normal",
                    )
                    top3_cols[i].caption(direction)

            except Exception as exc:
                st.warning(f"[!] XAI açıklaması üretilemedi: {exc}")

st.divider()
st.markdown("### 🖥️ [4] SERVER BACKEND LOGS")
system_logs = get_system_logs(tail_lines=10)
st.code(system_logs, language="bash")