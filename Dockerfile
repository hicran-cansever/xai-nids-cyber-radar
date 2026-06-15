# 1. Hafif ve güvenli bir Python tabanı seç (Debian Slim)
FROM python:3.10-slim

# 2. İşletim sistemi güncellemeleri, C derleyicisi ve Healthcheck için curl
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 3. Çalışma dizinini ayarla
WORKDIR /app

# 4. Sadece requirements dosyasını kopyala
COPY requirements.txt .

# 5. Kütüphaneleri kur
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Projenin geri kalan tüm kodlarını kopyala
COPY . .

# 7. Streamlit'in varsayılan portunu dışarı aç
EXPOSE 8501

# 8. Konteynerin sağlığını kontrol et (Healthcheck)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 9. Arayüzü başlat (Sistemin Kalbi)
ENTRYPOINT ["streamlit", "run", "src/poc_ui.py", "--server.port=8501", "--server.address=0.0.0.0"]