"""Yol yönetimi, konfigürasyon, loglama ve model serileştirme yardımcıları."""

import os
import hashlib
import logging
import yaml
import joblib
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List


# Yol yönetimi

def get_project_path(sub_folder: str, filename: str = "") -> str:
    """Proje dizinindeki bir dosya veya klasörün mutlak yolunu döndürür.

    Args:
        sub_folder: Proje ana dizini altındaki alt klasör adı (örn: 'data', 'models').
        filename: Alt klasör içindeki dosya adı. Varsayılan boş string.

    Returns:
        Dosya veya klasörün mutlak yolu.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if filename:
        return os.path.join(base_dir, sub_folder, filename)
    return os.path.join(base_dir, sub_folder)


def check_file_exists(path: str) -> bool:
    """Belirtilen yolun (dosya veya klasör) mevcut olup olmadığını kontrol eder.

    Args:
        path: Kontrol edilecek yol.

    Returns:
        Yol mevcutsa True, aksi halde False.
    """
    return os.path.exists(path)


def get_basename(path: str) -> str:
    """Belirtilen yolun dosya adını (basename) döndürür.

    Args:
        path: Dosya yolu.

    Returns:
        Dosya adı.
    """
    return os.path.basename(path)


def list_files_in_dir(dir_path: str, extension: str = "") -> List[str]:
    """Belirtilen dizindeki dosyaların isimlerini liste olarak döndürür.

    Args:
        dir_path: Dosyaların listeleneceği dizin.
        extension: Belirli bir uzantıya sahip dosyaları filtrelemek için (örn: '.csv').

    Returns:
        Dosya adlarından oluşan liste. Dizin bulunamazsa boş liste döner.
    """
    if not os.path.exists(dir_path):
        return []
    files = os.listdir(dir_path)
    if extension:
        return [f for f in files if f.endswith(extension)]
    return files


# Konfigürasyon yönetimi

# Her bölüm için zorunlu anahtarlar.
# Yeni bir config anahtarı eklendiğinde buraya da eklenmelidir.
_REQUIRED_CONFIG_SCHEMA: Dict[str, List[str]] = {
    "data": ["max_files_train", "max_files_xai", "test_size", "random_state"],
    "model": ["n_estimators", "max_depth", "n_jobs", "class_weight", "optuna_trials"],
    "xai": ["sample_size", "max_display_features", "top_n_local_features"],
    "labels": ["benign_keyword"],
}


def _validate_config(config: Any) -> None:
    """Yüklenen config sözlüğünü zorunlu şemaya karşı doğrular.

    ``yaml.safe_load`` boş bir dosya için ``None`` döndürebilir. Eksik
    bölüm ya da anahtar, tüketim noktasında anlaşılmaz ``TypeError`` /
    ``KeyError`` üretmek yerine burada erken ve açıklayıcı biçimde
    raporlanır.

    Args:
        config: ``yaml.safe_load`` çıktısı. ``None`` veya ``dict`` olabilir.

    Raises:
        ValueError: config ``None`` ya da dict değilse.
        KeyError: Zorunlu bir bölüm veya anahtar eksikse.
    """
    if config is None or not isinstance(config, dict):
        raise ValueError(
            "config.yaml boş veya geçersiz formatta. "
            "Dosyanın doğru YAML sözdizimi içerdiğinden emin olun."
        )

    for section, keys in _REQUIRED_CONFIG_SCHEMA.items():
        if section not in config:
            raise KeyError(
                f"config.yaml şema hatası: '{section}' bölümü eksik. "
                f"Beklenen anahtarlar: {keys}"
            )
        for key in keys:
            if key not in config[section]:
                raise KeyError(
                    f"config.yaml şema hatası: '{section}.{key}' anahtarı eksik. "
                    f"'{section}' bölümünün tüm zorunlu anahtarları: {keys}"
                )


def load_config() -> Dict[str, Any]:
    """Merkezi config.yaml dosyasını okur, şemasını doğrular ve dict olarak döndürür.

    Doğrulama, tüketim noktasında oluşabilecek ``KeyError`` / ``TypeError``
    hatalarını erken ve açıklayıcı biçimde yakalar. Şema değişikliklerini
    yönetmek için ``_REQUIRED_CONFIG_SCHEMA`` sabitini güncelleyin.

    Returns:
        Doğrulanmış konfigürasyon sözlüğü.

    Raises:
        FileNotFoundError: config.yaml dosyası bulunamazsa.
        yaml.YAMLError: YAML dosyası parse edilemezse.
        ValueError: config.yaml boş veya dict değilse.
        KeyError: Zorunlu bir bölüm veya anahtar eksikse.
    """
    config_path = get_project_path("", "config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"config.yaml bulunamadı: '{config_path}'. "
            "Dosyanın proje kök dizininde mevcut olduğundan emin olun."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    _validate_config(config)
    return config


# Model serileştirme

def _compute_sha256(file_path: str) -> str:
    """Belirtilen dosyanın SHA-256 özetini hesaplar.

    Büyük model dosyaları için bellek taşmasını önlemek amacıyla
    dosya 8 KB'lık parçalar halinde okunur.

    Args:
        file_path: Hash'i hesaplanacak dosyanın tam yolu.

    Returns:
        64 karakterlik onaltılık SHA-256 özet string'i.
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def save_model_with_hash(model: Any, model_path: str, compress: int = 3) -> str:
    """Modeli diske kaydeder ve SHA-256 bütünlük imzasını yan dosyaya yazar.

    Kaydedilen imza dosyası ``<model_path>.sha256`` adını taşır.
    ``verify_and_load_model`` bu imzayı yükleme öncesinde doğrular;
    eşleşmezse yükleme reddedilir.

    Neden gerekli?
        ``joblib`` / ``pickle`` tabanlı ``.pkl`` dosyaları imzasız
        olduğunda, bir saldırgan dosyayı değiştirerek ``joblib.load()``
        çağrısında rastgele Python kodu çalıştırabilir
        (Arbitrary Code Execution — OWASP A08:2021).

    Args:
        model: Kaydedilecek Python nesnesi (genellikle sklearn model).
        model_path: Modelin yazılacağı tam dosya yolu.
        compress: joblib sıkıştırma seviyesi (0–9). Varsayılan 3.

    Returns:
        Hesaplanan SHA-256 özeti string'i.
    """
    joblib.dump(model, model_path, compress=compress)
    sha256_hash = _compute_sha256(model_path)
    hash_path = model_path + ".sha256"

    with open(hash_path, "w", encoding="utf-8") as f:
        f.write(sha256_hash)

    logger = logging.getLogger("utils")
    logger.info(
        f"[GÜVENLİK] Model kaydedildi ve imzalandı: {model_path} "
        f"| SHA-256: {sha256_hash[:16]}..."
    )
    return sha256_hash


def verify_and_load_model(model_path: str) -> Any:
    """Model dosyasının SHA-256 bütünlüğünü doğrular ve güvenli biçimde yükler.

    Yükleme öncesinde ``<model_path>.sha256`` dosyasındaki kayıtlı özet
    ile mevcut dosyanın özeti karşılaştırılır. Eşleşmezse yükleme
    ``SecurityError`` ile reddedilir; bu durum dosyanın izinsiz
    değiştirildiğine işaret eder.

    Hash dosyası hiç yoksa (örn. eski sürümden kalan model) güvenli
    olmayan yükleme yapılmaz; bunun yerine açıklayıcı bir hata fırlatılır.

    Args:
        model_path: Yüklenecek ``.pkl`` dosyasının tam yolu.

    Returns:
        Doğrulanmış ve yüklenmiş Python nesnesi.

    Raises:
        FileNotFoundError: Model dosyası veya ``.sha256`` imza dosyası eksikse.
        SecurityError: SHA-256 özeti eşleşmiyorsa (dosya manipülasyonu).
    """
    logger = logging.getLogger("utils")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model dosyası bulunamadı: '{model_path}'. "
            "Lütfen önce train_model.py'yi çalıştırın."
        )

    hash_path = model_path + ".sha256"
    if not os.path.exists(hash_path):
        raise FileNotFoundError(
            f"Model imza dosyası bulunamadı: '{hash_path}'. "
            "Model, save_model_with_hash() ile kaydedilmemiş olabilir. "
            "Güvenlik nedeniyle yükleme reddedildi. "
            "Modeli yeniden eğiterek imzalı biçimde kaydedin."
        )

    with open(hash_path, "r", encoding="utf-8") as f:
        expected_hash = f.read().strip()

    actual_hash = _compute_sha256(model_path)

    if actual_hash != expected_hash:
        raise SecurityError(
            f"[KRİTİK GÜVENLİK İHLALİ] Model bütünlük doğrulaması başarısız: "
            f"'{model_path}'\n"
            f"  Beklenen SHA-256 : {expected_hash}\n"
            f"  Mevcut  SHA-256  : {actual_hash}\n"
            "Bu dosya izinsiz değiştirilmiş olabilir. Yükleme iptal edildi."
        )

    logger.info(
        f"[GÜVENLİK] Bütünlük doğrulaması geçti: {os.path.basename(model_path)} "
        f"| SHA-256: {actual_hash[:16]}..."
    )
    return joblib.load(model_path)


class SecurityError(Exception):
    """Model bütünlük doğrulaması başarısız olduğunda fırlatılır.

    ``Exception``'dan türetilerek standart hata yakalama mekanizmalarıyla
    uyumlu tutulmuştur. Ayrı bir istisnai tip kullanmak, çağıran kodun
    güvenlik ihlallerini genel hatalardan ayırt etmesini sağlar.
    """


# Loglama

_MAX_LOG_BYTES: int = 10 * 1024 * 1024   # 10 MB — tek log dosyası üst sınırı
_BACKUP_LOG_COUNT: int = 5               # system.log + system.log.1 ... .5


def setup_logging() -> None:
    """Proje geneli loglama altyapısını kurar.

    Bu fonksiyon yalnızca ana giriş noktalarında
    (``if __name__ == '__main__'``) çağrılmalıdır; kütüphane modüllerinde
    çağrılmamalıdır.

    Neden RotatingFileHandler?
        ``logging.FileHandler`` ile log dosyası production ortamında
        sınırsız büyür. Bir SOC sisteminde dakikada yüzlerce log satırı
        yazılabilir; bu durum disk dolumuna ve Denial-of-Service'e
        yol açar. ``RotatingFileHandler`` dosyayı ``_MAX_LOG_BYTES``
        sınırında döndürür ve en fazla ``_BACKUP_LOG_COUNT`` yedek tutar.

    Çıktı:
        - ``logs/system.log``           → aktif log
        - ``logs/system.log.1`` ... ``.5`` → yedek rotasyonlar
        - Konsol (StreamHandler)
    """
    log_dir = get_project_path("logs")
    os.makedirs(log_dir, exist_ok=True)

    log_format = logging.Formatter(
        "%(asctime)s - [%(levelname)s] - %(name)s - %(message)s"
    )

    rotating_handler = RotatingFileHandler(
        filename=get_project_path("logs", "system.log"),
        maxBytes=_MAX_LOG_BYTES,
        backupCount=_BACKUP_LOG_COUNT,
        encoding="utf-8",
    )
    rotating_handler.setFormatter(log_format)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[rotating_handler, console_handler],
    )
