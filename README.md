<div align="center">

# ⚡ NIDS Cyber Radar

### Açıklanabilir Yapay Zeka Destekli Ağ Saldırı Tespit Sistemi
### Explainable AI-Powered Network Intrusion Detection System

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.1-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![SHAP](https://img.shields.io/badge/XAI-SHAP_TreeExplainer-8B5CF6?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyTDIgN2wxMCA1IDEwLTV6TTIgMTdsOCA0IDQtMiA0IDJ6Ii8+PC9zdmc+&logoColor=white)](https://shap.readthedocs.io/)
[![Optuna](https://img.shields.io/badge/HPO-Optuna_TPE-00C4CC?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iMTAiIGZpbGw9IndoaXRlIi8+PC9zdmc+&logoColor=white)](https://optuna.org/)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-98.41%25-22C55E?style=for-the-badge)](.)
[![Classes](https://img.shields.io/badge/Threat_Classes-34-EF4444?style=for-the-badge)](.)
[![License](https://img.shields.io/badge/License-MIT-94A3B8?style=for-the-badge)](LICENSE)

<br/>

> **Kurs:** CENG 3544 — Computer and Network Security  
> **Kapsam:** Makine Öğrenmesi + Açıklanabilir YZ + MLOps Boru Hattı

</div>

---

<br/>

# 🇹🇷 TÜRKÇE DOKÜMANTASYON

<br/>

## 📋 İçindekiler

- [Proje Hakkında](#-proje-hakkında)
- [Mimari Başarılar & Özel Mühendislik Çözümleri](#-mimari-başarılar--özel-mühendislik-çözümleri)
- [Sistem Mimarisi](#-sistem-mimarisi)
- [Model Performansı](#-model-performansı)
- [Kurulum & Çalıştırma (Docker)](#-kurulum--çalıştırma-docker)
- [Proje Yapısı](#-proje-yapısı)
- [Konfigürasyon](#-konfigürasyon)
- [Champion Model Sürümü](#-champion-model-sürümü)

---

## 🎯 Proje Hakkında

**NIDS Cyber Radar**, CENG 3544 Computer and Network Security dersi kapsamında geliştirilen bir **Ağ Saldırı Tespit Sistemi (NIDS)**'dir. Bu sistem; ham ağ trafiği paketlerini gerçek zamanlı olarak sınıflandırır ve kararlarının gerekçesini **SHAP (SHapley Additive exPlanations)** değerleri aracılığıyla SOC analistlerine görsel olarak sunar.

Standart bir makine öğrenmesi alıştırmasının ötesine geçerek, proje; **RAM güvenliği**, **veri sızıntısı koruması**, **model bütünlük doğrulaması** ve **gerçek zamanlı açıklanabilirlik** gibi, üretim ortamlarında kritik olan mühendislik problemlerini özgün çözümlerle ele almaktadır.

### Temel Yetenekler

| Yetenek | Detay |
|---|---|
| 🛡️ **34 Siber Tehdit Sınıfı** | DDoS, SQL Injection, XSS, Port Scan ve daha fazlası |
| 🧠 **XAI Karar Gerekçesi** | Her tahmin için SHAP tabanlı lokal açıklama |
| ⚡ **Tek Komut Deployment** | `docker-compose up --build` ile sıfırdan ayağa kalkış |
| 🔐 **SHA-256 Model İmzalama** | Dosya manipülasyonuna karşı bütünlük doğrulaması |
| 📊 **98.41% ROC-AUC** | 34 sınıflı Macro One-vs-Rest değerlendirmesi |

---

## 🏗️ Mimari Başarılar & Özel Mühendislik Çözümleri

Bu bölüm, projenin standart bir ML modelinin ötesinde ele aldığı mühendislik problemlerini açıklamaktadır.

---

### 1️⃣ Smart Siphoning — Akıllı Cımbızlama Boru Hattı

**Modül:** [`src/data_loader.py`](src/data_loader.py) → `IoTDataLoader`

**Problem:** CIC-IoT-2023 gibi devasa ağ trafiği veri setleri (63+ CSV dosyası), naif bir `pd.read_csv()` yaklaşımıyla yüklendiğinde işletim sistemi düzeyinde **OOM (Out-of-Memory) kilitlenmesine** yol açar. Bunun yanı sıra `BENIGN` sınıfının veri setine ezici üstünlüğü, `XSS` veya `SQL Injection` gibi nadir saldırı sınıflarının model tarafından görmezden gelmesine neden olur.

**Çözüm — Dört Kademeli Savunma Hattı:**

```
┌─────────────────────────────────────────────────────────────────┐
│  CSV Dosyası Okunur                                             │
│         │                                                       │
│  [1] Sızıntı Koruması: 'Number', 'Unnamed:0' sütunları silinir │
│         │                                                       │
│  [2] Hedef Tespiti: label/Label/class/attack_type  aranır      │
│         │                                                       │
│  [3] STRATIFIED SUB-SAMPLING (Cımbızlama)                      │
│       └─► Her sınıf için MAX 2000 örnek örneklenir             │
│           (pandas apply tuzağından kaçmak için explicit döngü) │
│         │                                                       │
│  [4] RAM Sıkıştırma: int64→int8/16/32, float64→float32        │
│       └─► Dosya başı bellek kullanımı %40–60 azaltılır        │
└─────────────────────────────────────────────────────────────────┘
```

**Teknik Önem:** `groupby` tabanlı Stratified Sub-sampling, her nadir sınıfın (XSS, SQL Injection, vb.) örneklemde **orantılı** temsil edilmesini garanti eder. Sabit `random_state=42` ile sonuçlar **tam tekrarlanabilir** (reproducible)'dir.

---

### 2️⃣ Model Başarısı & Optuna TPE Optimizasyonu

**Modül:** [`src/train_model.py`](src/train_model.py) → `optimize_hyperparameters()`

**Problem:** Manuel hiperparametre deneme-yanılması, büyük arama uzaylarında hem verimsiz hem de önyargılıdır. Grid Search ise bu boyutta hesaplamalı olarak imkânsızdır.

**Çözüm:** Optuna'nın **TPE (Tree-structured Parzen Estimator)** örnekleyicisi, Bayes istatistiğini kullanarak arama uzayını akıllıca daraltır:

```python
# Arama Uzayı
n_estimators   : [50, 100, 150, 200, 250, 300]
max_depth      : [10, 20, 30, None]
min_samples_split: 2 – 10
min_samples_leaf : 1 – 4
max_features   : ["sqrt", "log2"]
class_weight   : "balanced"  # Sınıf dengesizliği için sabit
```

**Kilitlenen Şampiyon Mimari:**
- **n_estimators: 250** ağaç (Optuna TPE tarafından seçildi)
- **Optimizasyon Hedefi:** 3-katlı Çapraz Doğrulama ile Macro F1 Skoru
- **Veri Sızıntısı Koruması:** LabelEncoder yalnızca `y_train` üzerinde `fit()` edilir; test seti yalnızca `transform()` görür

**Sonuç Metrikleri:**

| Metrik | Skor |
|---|---|
| **Macro ROC-AUC (OvR)** | **%98.41** |
| Weighted F1 | Yüksek |
| Toplam Sınıf | 34 |
| Değerlendirme | Stratified 80/20 Split |

---

### 3️⃣ Schema Alignment & Zero Data Leakage — Şema Kilidi

**Modüller:** [`src/poc_ui.py`](src/poc_ui.py) → `_align_schema()` | [`src/xai_explainer.py`](src/xai_explainer.py) → `generate_shap_summary()`

**Problem:** Canlı ağ trafiği yakalandığında veya XAI analizi için yeni veri seti yüklendiğinde, gelen veri matrisinin sütun sayısı/sırası modelin eğitimde öğrendiği **38 özelliklik şablondan** farklı olabilir. Bu, `predict()` çağrısında `ValueError` veya sessiz yanlış hesaplamalara yol açar.

**Çözüm — İki Katmanlı Şema Kilidi:**

```python
# poc_ui.py → _align_schema() ve xai_explainer.py → generate_shap_summary()

# ADIM 1: Modelin "hafızasından" beklenen sütun listesini çek
expected_cols = list(model.feature_names_in_)  # 38 özellik

# ADIM 2: Eksik sütunları '0' ile doldur (imputation)
missing_cols = set(expected_cols) - set(live_packet.columns)
for col in missing_cols:
    live_packet[col] = 0

# ADIM 3: Sütunları eğitim sırasına TAM OLARAK hizala, fazlalıkları at
live_packet = live_packet[expected_cols]
```

**Neden Kritik?** Bu mekanizma olmadan model, eğitimde görmediği sütun düzenlemeleriyle karşılaştığında **sessiz hata** üretir — yani yanlış tahmin yapar ama hata vermez. Şema kilidi bu "kör noktayı" ortadan kaldırır.

---

### 4️⃣ Local & Global XAI — SHAP Açıklanabilirlik Motoru

**Modüller:** [`src/xai_explainer.py`](src/xai_explainer.py) | [`src/poc_ui.py`](src/poc_ui.py) → `explain_prediction()`

**Problem:** Bir ağ güvenliği analistinin "Bu paket neden DDoS olarak işaretlendi?" sorusunu cevaplayabilmesi, kara kutu model çıktısından çok daha fazla değer taşır.

**Çözüm — İki Modlu XAI Pipeline:**

#### Global XAI (`xai_explainer.py`)
```
Champion Model + 300 örneklik temsili veri seti
        ↓
SHAP TreeExplainer (ağaç yapısı doğrudan kullanılır, model-agnostik değil)
        ↓
Çoklu Sınıf SHAP Matrisi: [n_samples × n_features × n_classes]
        ↓
Global Feature Importance Bar Chart (300 DPI PNG, 22×12 inch)
        ↓
"Bu model genel olarak hangi özelliklere bakıyor?"
```

#### Local XAI (`poc_ui.py`)
```
Tek Yakalanan Paket (Canlı Trafik)
        ↓
SHAP TreeExplainer → Paket başına SHAP vektörü
        ↓
Tahmin edilen sınıfın SHAP değerleri ayrıştırılır
        ↓
Top-N özellik → Kırmızı (Karar güçlendirir) / Mavi (Karar zayıflatır)
        ↓
"Bu model BU paketi neden DDoS/Zararsız dedi?"
```

**Teknik Detay:** `check_additivity=False` parametresi, çoklu sınıflı Random Forest'ta TreeExplainer'ın additivity check hatasını önler. Çoklu sınıf SHAP matrisi `[n_samples, n_features, n_classes]` boyutundadır; tahmin edilen sınıfın indeksi ile `shap_values[0, :, predicted_class_idx]` şeklinde doğru dilim alınır.

---

### 5️⃣ SHA-256 Model İmzalama — Güvenli Serileştirme

**Modül:** [`src/utils.py`](src/utils.py) → `save_model_with_hash()` / `verify_and_load_model()`

**Problem:** `.pkl` dosyaları imzasız olduğunda, bir saldırgan dosyayı değiştirerek `joblib.load()` üzerinden **Arbitrary Code Execution (OWASP A08:2021)** gerçekleştirebilir.

**Çözüm:**
- Model kaydedilirken SHA-256 özeti yan dosyaya (`model.pkl.sha256`) yazılır
- Her yüklemede hash yeniden hesaplanıp karşılaştırılır
- Uyumsuzluk `SecurityError` fırlatır ve yükleme **reddedilir**
- UI katmanında `SecurityError` yakalanarak kullanıcıya "GÜVENLİK İHLALİ" uyarısı gösterilir

---

## 🏛️ Sistem Mimarisi

```
xai_nids_project/
│
├── 📄 Dockerfile              # python:3.10-slim, curl healthcheck
├── 📄 docker-compose.yml      # Port 8501, volume mounts, restart policy
├── 📄 config.yaml             # Merkezi konfigürasyon (data/model/xai/labels)
├── 📄 requirements.txt        # Tam bağımlılık manifestosu
│
├── 📁 src/
│   ├── data_loader.py         # IoTDataLoader → Akıllı Cımbızlama + RAM Sıkıştırma
│   ├── train_model.py         # Optuna TPE → Champion RF → SHA-256 Serileştirme
│   ├── xai_explainer.py       # Global SHAP → Summary Bar Chart (300 DPI)
│   ├── poc_ui.py              # Streamlit SOC Terminal
│   └── utils.py               # Yol/Config/Log/SHA-256 Güvenlik Katmanı
│
├── 📁 models/
│   ├── nids_champion_model.pkl       # ~305 MB Champion RF (250 ağaç)
│   ├── nids_champion_model.pkl.sha256
│   ├── label_encoder.pkl             # 34 Sınıf LabelEncoder
│   └── label_encoder.pkl.sha256
│
└── 📁 data/
    ├── confusion_matrix.png          # Normalize edilmiş karışıklık matrisi
    ├── shap_summary_plot.png         # Global SHAP özet grafiği
    └── merged/                       # CIC-IoT-2023 CSV dosyaları (63+ adet)
```

---

## 📊 Model Performansı

| Parametre | Değer |
|---|---|
| Mimari | Random Forest Classifier |
| Ağaç Sayısı (`n_estimators`) | **250** (Optuna TPE) |
| Optimizasyon | Optuna TPE, 3-katlı CV Macro F1 |
| Sınıf Ağırlıklandırma | `balanced` (azınlık sınıf koruması) |
| Eğitim/Test Ayrımı | Stratified 80% / 20% |
| **Macro ROC-AUC (OvR)** | **%98.41** |
| Toplam Sınıf | **34** (DDoS, XSS, SQL Injection, Port Scan, vb.) |
| Veri Ön İşleme | Smart Siphoning + RAM Compression |

---

## 🚀 Kurulum & Çalıştırma (Docker)

### Ön Koşullar

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) kurulu ve çalışıyor olmalı
- `nids_champion_model.pkl` dosyası `models/` klasöründe mevcut olmalı  
  *(GitHub Releases → **v1.0.0 Stable Release** bölümünden indirin — aşağıya bakın)*
- `data/merged/` klasörü CSV dosyalarını içermeli

### Tek Komut ile Başlatma

```bash
# 1. Repoyu klonlayın
git clone https://github.com/<kullanici-adi>/xai_nids_project.git
cd xai_nids_project

# 2. Champion modeli GitHub Releases'ten indirip models/ klasörüne koyun
#    (bkz. "Champion Model Sürümü" bölümü)

# 3. Sistemi tek komutla ayağa kaldırın
docker-compose up --build
```

Sistem başarıyla ayağa kalktığında tarayıcıyı açın:

```
http://localhost:8501
```

### Servis Detayları (`docker-compose.yml`)

| Parametre | Değer |
|---|---|
| Servis Adı | `nids-cyber-radar` |
| Container | `nids_ui` |
| Port Yönlendirme | `8501:8501` |
| Volume — Veri | `./data:/app/data` |
| Volume — Modeller | `./models:/app/models` |
| Volume — Loglar | `./logs:/app/logs` |
| Yeniden Başlatma | `unless-stopped` |
| Healthcheck | `curl http://localhost:8501/_stcore/health` |

### Manuel Kurulum (Docker Olmadan)

```bash
# Sanal ortam oluştur
python -m venv venv
.\venv\Scripts\activate        # Windows
# source venv/bin/activate    # Linux/macOS

# Bağımlılıkları yükle
pip install -r requirements.txt

# (Opsiyonel) Champion modeli yeniden eğit
python src/train_model.py

# (Opsiyonel) Global SHAP grafiği üret
python src/xai_explainer.py

# Arayüzü başlat
streamlit run src/poc_ui.py
```

---

## ⚙️ Konfigürasyon

Tüm sistem parametreleri [`config.yaml`](config.yaml) üzerinden yönetilir:

```yaml
data:
  max_files_train: 63    # Eğitim için kullanılacak maksimum CSV dosya sayısı
  max_files_xai: 2       # XAI analizi için RAM korumalı dosya sayısı
  test_size: 0.2         # %20 test, %80 eğitim (stratified)
  random_state: 42       # Tekrarlanabilirlik için sabit tohum

model:
  n_estimators: 100      # Optuna devre dışıysa kullanılan sabit değer
  max_depth: null        # null = sınırsız derinlik
  class_weight: "balanced"
  optuna_trials: 5       # 0 = Optuna devre dışı | 20+ = tam optimizasyon

xai:
  sample_size: 300       # Global SHAP için örneklem büyüklüğü
  max_display_features: 10
  top_n_local_features: 12  # UI'da gösterilecek lokal SHAP özellik sayısı

labels:
  benign_keyword: "BENIGN"  # Zararsız trafik etiket anahtarı
```

---

## 🏆 Champion Model Sürümü

> **📦 Not:** `nids_champion_model.pkl` dosyası yaklaşık **305 MB** boyutunda olduğundan doğrudan Git deposuna eklenmemiştir.

Eğitilmiş şampiyon model şu adreste yayımlanmaktadır:

**GitHub → Releases → [v1.0.0 Stable Release](../../releases/tag/v1.0.0)**

İndirme sonrası dosyayı `models/` klasörüne yerleştirin:

```
models/
├── nids_champion_model.pkl        ← Buraya koyun
├── nids_champion_model.pkl.sha256 ← Bu imza dosyasını da indirin
├── label_encoder.pkl
└── label_encoder.pkl.sha256
```

> ⚠️ **Önemli:** SHA-256 imza dosyası (`*.sha256`) ile birlikte indirilmesi zorunludur. Sistem, model yüklenirken otomatik olarak bütünlük doğrulaması yapar; imza dosyası eksikse veya hash uyuşmazlığı varsa yükleme güvenlik gerekçesiyle reddedilir.

---

<br/>
<br/>

---

# 🇬🇧 ENGLISH DOCUMENTATION

<br/>

## 📋 Table of Contents

- [About The Project](#-about-the-project)
- [Architectural Achievements & Engineering Solutions](#-architectural-achievements--engineering-solutions)
- [System Architecture](#-system-architecture)
- [Model Performance](#-model-performance)
- [Installation & Deployment (Docker)](#-installation--deployment-docker)
- [Project Structure](#-project-structure-1)
- [Configuration](#-configuration-1)
- [Champion Model Release](#-champion-model-release)

---

## 🎯 About The Project

**NIDS Cyber Radar** is a **Network Intrusion Detection System (NIDS)** developed as part of the CENG 3544 Computer and Network Security course. The system classifies raw network traffic packets in real time and visually presents the rationale behind each decision to SOC analysts via **SHAP (SHapley Additive exPlanations)** values.

Beyond a standard machine learning exercise, this project addresses production-critical engineering challenges including **RAM safety**, **data leakage prevention**, **model integrity verification**, and **real-time explainability** through original solutions.

### Core Capabilities

| Capability | Detail |
|---|---|
| 🛡️ **34 Cyber Threat Classes** | DDoS, SQL Injection, XSS, Port Scan and more |
| 🧠 **XAI Decision Rationale** | SHAP-based local explanation for every prediction |
| ⚡ **One-Command Deployment** | Zero-to-running with `docker-compose up --build` |
| 🔐 **SHA-256 Model Signing** | Integrity verification against file tampering |
| 📊 **98.41% ROC-AUC** | 34-class Macro One-vs-Rest evaluation |

---

## 🏗️ Architectural Achievements & Engineering Solutions

This section explains the engineering challenges addressed by the project.

---

### 1️⃣ Smart Siphoning — Intelligent Stratified Sub-sampling Pipeline

**Module:** [`src/data_loader.py`](src/data_loader.py) → `IoTDataLoader`

**Problem:** Loading massive network traffic datasets such as CIC-IoT-2023 (63+ CSV files) naively with `pd.read_csv()` causes **OOM (Out-of-Memory) crashes** at the OS level. Furthermore, the overwhelming dominance of the `BENIGN` class causes the model to ignore rare attack classes such as `XSS` or `SQL Injection`.

**Solution — Four-Stage Defence Pipeline:**

```
┌──────────────────────────────────────────────────────────────────┐
│  CSV File is read                                                │
│        │                                                         │
│  [1] Leakage Guard: 'Number', 'Unnamed:0' columns are dropped   │
│        │                                                         │
│  [2] Target Detection: label/Label/class/attack_type searched    │
│        │                                                         │
│  [3] STRATIFIED SUB-SAMPLING                                     │
│      └─► Max 2000 samples drawn per class                       │
│          (explicit loop to avoid pandas apply trap)              │
│        │                                                         │
│  [4] RAM Compression: int64→int8/16/32, float64→float32        │
│      └─► Memory footprint reduced 40–60% per file               │
└──────────────────────────────────────────────────────────────────┘
```

**Technical Significance:** `groupby`-based Stratified Sub-sampling guarantees **proportional representation** of every rare class (XSS, SQL Injection, etc.) in the sample. A fixed `random_state=42` ensures full **reproducibility**.

---

### 2️⃣ Model Achievement & Optuna TPE Optimisation

**Module:** [`src/train_model.py`](src/train_model.py) → `optimize_hyperparameters()`

**Problem:** Manual hyperparameter trial-and-error is both inefficient and biased in large search spaces. Grid Search is computationally infeasible at this scale.

**Solution:** Optuna's **TPE (Tree-structured Parzen Estimator)** sampler uses Bayesian statistics to intelligently narrow the search space:

```python
# Search Space
n_estimators    : [50, 100, 150, 200, 250, 300]
max_depth       : [10, 20, 30, None]
min_samples_split: 2 – 10
min_samples_leaf : 1 – 4
max_features    : ["sqrt", "log2"]
class_weight    : "balanced"   # fixed for class imbalance
```

**Champion Architecture Selected by Optuna:**
- **n_estimators: 250** trees
- **Optimisation Target:** Macro F1 Score via 3-fold Cross-Validation
- **Zero Data Leakage:** LabelEncoder is `fit()` only on `y_train`; the test set sees only `transform()`

**Performance Metrics:**

| Metric | Score |
|---|---|
| **Macro ROC-AUC (OvR)** | **98.41%** |
| Total Classes | 34 |
| Evaluation | Stratified 80/20 Split |
| Class Balancing | `balanced` weight policy |

---

### 3️⃣ Schema Alignment & Zero Data Leakage — Schema Lock

**Modules:** [`src/poc_ui.py`](src/poc_ui.py) → `_align_schema()` | [`src/xai_explainer.py`](src/xai_explainer.py) → `generate_shap_summary()`

**Problem:** When live network traffic is captured or a new dataset is loaded for XAI analysis, the incoming data matrix may differ in column count or ordering from the **38-feature template** the model memorised during training. This causes `ValueError` on `predict()` or, worse, silent incorrect computations.

**Solution — Two-Layer Schema Lock:**

```python
# Step 1: Pull the expected column list from the model's "memory"
expected_cols = list(model.feature_names_in_)   # 38 features

# Step 2: Fill missing columns with '0' (zero imputation)
missing_cols = set(expected_cols) - set(live_packet.columns)
for col in missing_cols:
    live_packet[col] = 0

# Step 3: Reorder columns EXACTLY to training order, drop extras
live_packet = live_packet[expected_cols]
```

**Why Critical?** Without this mechanism, the model silently produces wrong predictions when encountering column arrangements it did not see in training — no error, just incorrect output. The schema lock eliminates this blind spot entirely.

---

### 4️⃣ Local & Global XAI — SHAP Explainability Engine

**Modules:** [`src/xai_explainer.py`](src/xai_explainer.py) | [`src/poc_ui.py`](src/poc_ui.py) → `explain_prediction()`

**Problem:** A network security analyst's ability to answer "Why was this packet flagged as DDoS?" carries far more operational value than a bare model output.

**Solution — Dual-Mode XAI Pipeline:**

#### Global XAI (`xai_explainer.py`)
```
Champion Model + 300-sample representative dataset
        ↓
SHAP TreeExplainer (leverages tree structure directly)
        ↓
Multi-class SHAP Matrix: [n_samples × n_features × n_classes]
        ↓
Global Feature Importance Bar Chart (300 DPI PNG, 22×12 inch)
        ↓
"Which features does this model generally rely on?"
```

#### Local XAI (`poc_ui.py`)
```
Single Captured Packet (Live Traffic)
        ↓
SHAP TreeExplainer → Per-packet SHAP vector
        ↓
SHAP values for the predicted class are extracted
        ↓
Top-N features → Red (strengthens decision) / Blue (weakens decision)
        ↓
"Why did this model call THIS packet DDoS/Benign?"
```

**Technical Detail:** `check_additivity=False` prevents the TreeExplainer additivity check error on multi-class Random Forests. The multi-class SHAP matrix is shaped `[n_samples, n_features, n_classes]`; the correct slice is extracted as `shap_values[0, :, predicted_class_idx]`.

---

### 5️⃣ SHA-256 Model Signing — Secure Serialisation

**Module:** [`src/utils.py`](src/utils.py) → `save_model_with_hash()` / `verify_and_load_model()`

**Problem:** Unsigned `.pkl` files allow an attacker to modify the file and achieve **Arbitrary Code Execution (OWASP A08:2021)** via `joblib.load()`.

**Solution:**
- On save, the SHA-256 digest is written to a side-car file (`model.pkl.sha256`)
- On every load, the hash is recomputed and compared
- Mismatch raises `SecurityError` and loading is **rejected**
- The UI layer catches `SecurityError` and surfaces a "SECURITY BREACH" warning to the user

---

## 🏛️ System Architecture

```
xai_nids_project/
│
├── 📄 Dockerfile              # python:3.10-slim, curl healthcheck
├── 📄 docker-compose.yml      # Port 8501, volume mounts, restart policy
├── 📄 config.yaml             # Central config (data/model/xai/labels)
├── 📄 requirements.txt        # Full dependency manifest
│
├── 📁 src/
│   ├── data_loader.py         # IoTDataLoader → Smart Siphoning + RAM Compression
│   ├── train_model.py         # Optuna TPE → Champion RF → SHA-256 Serialisation
│   ├── xai_explainer.py       # Global SHAP → Summary Bar Chart (300 DPI)
│   ├── poc_ui.py              # Streamlit SOC Terminal
│   └── utils.py               # Path/Config/Logging/SHA-256 Security Layer
│
├── 📁 models/
│   ├── nids_champion_model.pkl       # ~305 MB Champion RF (250 trees)
│   ├── nids_champion_model.pkl.sha256
│   ├── label_encoder.pkl             # 34-Class LabelEncoder
│   └── label_encoder.pkl.sha256
│
└── 📁 data/
    ├── confusion_matrix.png          # Normalised confusion matrix
    ├── shap_summary_plot.png         # Global SHAP summary plot
    └── merged/                       # CIC-IoT-2023 CSV files (63+ files)
```

---

## 📊 Model Performance

| Parameter | Value |
|---|---|
| Architecture | Random Forest Classifier |
| Trees (`n_estimators`) | **250** (selected by Optuna TPE) |
| Optimisation | Optuna TPE, 3-fold CV Macro F1 |
| Class Weighting | `balanced` (minority class protection) |
| Train/Test Split | Stratified 80% / 20% |
| **Macro ROC-AUC (OvR)** | **98.41%** |
| Total Classes | **34** (DDoS, XSS, SQL Injection, Port Scan, etc.) |
| Pre-processing | Smart Siphoning + RAM Compression |

---

## 🚀 Installation & Deployment (Docker)

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- `nids_champion_model.pkl` must be present in the `models/` directory  
  *(Download from GitHub Releases → **v1.0.0 Stable Release** — see below)*
- `data/merged/` directory must contain the CSV files

### One-Command Launch

```bash
# 1. Clone the repository
git clone https://github.com/<username>/xai_nids_project.git
cd xai_nids_project

# 2. Download the Champion Model from GitHub Releases and place it in models/
#    (see "Champion Model Release" section)

# 3. Launch the entire system with a single command
docker-compose up --build
```

Once the system is running, open your browser:

```
http://localhost:8501
```

### Service Details (`docker-compose.yml`)

| Parameter | Value |
|---|---|
| Service Name | `nids-cyber-radar` |
| Container Name | `nids_ui` |
| Port Mapping | `8501:8501` |
| Volume — Data | `./data:/app/data` |
| Volume — Models | `./models:/app/models` |
| Volume — Logs | `./logs:/app/logs` |
| Restart Policy | `unless-stopped` |
| Healthcheck | `curl http://localhost:8501/_stcore/health` |

### Manual Setup (Without Docker)

```bash
# Create virtual environment
python -m venv venv
.\venv\Scripts\activate        # Windows
# source venv/bin/activate    # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# (Optional) Retrain the champion model from scratch
python src/train_model.py

# (Optional) Generate global SHAP summary plot
python src/xai_explainer.py

# Launch the interface
streamlit run src/poc_ui.py
```

---

## 📁 Project Structure

```
src/
├── data_loader.py    Smart Siphoning + RAM compression pipeline
├── train_model.py    Optuna HPO + RF training + SHA-256 serialisation
├── xai_explainer.py  Global SHAP analysis + 300 DPI visualisation
├── poc_ui.py         Streamlit SOC terminal + local XAI per packet
└── utils.py          Security layer: path/config/logging/SHA-256
```

---

## ⚙️ Configuration

All system parameters are managed via [`config.yaml`](config.yaml):

```yaml
data:
  max_files_train: 63    # Max CSV files used for training
  max_files_xai: 2       # RAM-safe file count for XAI analysis
  test_size: 0.2         # 20% test, 80% train (stratified)
  random_state: 42       # Fixed seed for reproducibility

model:
  n_estimators: 100      # Fallback value if Optuna is disabled
  max_depth: null        # null = unlimited depth
  class_weight: "balanced"
  optuna_trials: 5       # 0 = Optuna disabled | 20+ = full optimisation

xai:
  sample_size: 300       # Sample size for global SHAP computation
  max_display_features: 10
  top_n_local_features: 12  # Local SHAP features shown in UI

labels:
  benign_keyword: "BENIGN"  # Keyword identifying benign traffic labels
```

---

## 🏆 Champion Model Release

> **📦 Note:** `nids_champion_model.pkl` is approximately **305 MB** in size and is therefore not committed directly to the Git repository.

The trained champion model is published at:

**GitHub → Releases → [v1.0.0 Stable Release](../../releases/tag/v1.0.0)**

After downloading, place the files in the `models/` directory:

```
models/
├── nids_champion_model.pkl        ← Place here
├── nids_champion_model.pkl.sha256 ← Download this signature file too
├── label_encoder.pkl
└── label_encoder.pkl.sha256
```

> ⚠️ **Important:** The SHA-256 signature file (`*.sha256`) **must** be downloaded alongside the model. The system automatically performs integrity verification on every load; if the signature file is missing or the hash does not match, loading is rejected on security grounds.

---

## 🛠️ Tech Stack

| Layer | Technology | Version |
|---|---|---|
| **Language** | Python | 3.10 |
| **UI / Dashboard** | Streamlit | 1.32.0 |
| **ML Framework** | scikit-learn | 1.4.1 |
| **HPO Engine** | Optuna (TPE Sampler) | 3.5.0 |
| **XAI Engine** | SHAP (TreeExplainer) | 0.45.0 |
| **Data** | Pandas + NumPy | 2.2.1 / 1.26.4 |
| **Serialisation** | joblib | 1.3.2 |
| **Containerisation** | Docker + Compose | — |
| **Config** | PyYAML | 6.0.1 |

---

## 📄 License

This project is distributed under the MIT License. See [`LICENSE`](LICENSE) for details.

---

<div align="center">

**CENG 3544 — Computer and Network Security**

*Developed as an academic project demonstrating MLOps and cybersecurity engineering principles.*

</div>
