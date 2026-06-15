import os
import pandas as pd
import numpy as np
import random
import logging
from typing import Tuple, List, Optional

from utils import load_config, get_project_path, check_file_exists, get_basename, list_files_in_dir

# --- Logger ---
logger = logging.getLogger("DataLoader")

class IoTDataLoader:
    def __init__(self) -> None:
        self.data_dir = get_project_path("data", "merged")
        self.config = load_config()
        logger.info("IoTDataLoader başlatıldı ve config.yaml başarıyla okundu.")

    def load_data(self, max_files: Optional[int] = None, random_select: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        if max_files is None:
            max_files = self.config['data']['max_files_train']
            
        logger.info(f"Aranan tam dizin: {self.data_dir}")
        
        if not check_file_exists(self.data_dir):
            raise FileNotFoundError(f"HATA: '{self.data_dir}' dizini bulunamadı!")
            
        csv_files = list_files_in_dir(self.data_dir, extension='.csv')
        all_files: List[str] = sorted([get_project_path("data/merged", f) for f in csv_files])
        
        rng = random.Random(self.config['data']['random_state'])
        
        max_limit = min(max_files, len(all_files)) if max_files > 0 else len(all_files)
        if random_select:
            selected_files = rng.sample(all_files, max_limit)
            logger.info(f"{len(all_files)} dosya arasından {len(selected_files)} tanesinde NADİR SALDIRI AVI başlatıldı.")
        else:
            selected_files = all_files[:max_limit]

        df_list: List[pd.DataFrame] = []
        MAX_SAMPLES_PER_CLASS = 2000 
        
        for file in selected_files:
            try:
                df = pd.read_csv(file)
            except Exception as e:
                logger.error(f"Hata: {file} okunamadı. Atlanıyor. Detay: {e}")
                continue
            
            # 1. Sızıntı Koruması
            leakage_cols = ['Number', 'Unnamed: 0']
            for col in leakage_cols:
                if col in df.columns:
                    df = df.drop(columns=[col])
                    
            # 2. Erken Hedef (Target) Tespiti
            possible_target_names = ['label', 'Label', 'class', 'Class', 'attack_type', 'Attack_type']
            target_col = next((name for name in possible_target_names if name in df.columns), None)
            
            if target_col is None:
                continue
                
            # 3. AKILLI CIMBIZLAMA STRATEJİSİ (Pandas Çökme Korumalı)
            pre_siphon_rows = len(df)
            
            # Pandas apply tuzağına düşmemek için Explicit (Açık) Döngü kullanıyoruz
            sampled_chunks = []
            for _, group in df.groupby(target_col):
                n_samples = min(len(group), MAX_SAMPLES_PER_CLASS)
                sampled_chunks.append(group.sample(n=n_samples, random_state=self.config['data']['random_state']))
                
            df = pd.concat(sampled_chunks, ignore_index=True)
            post_siphon_rows = len(df)
                
            # 4. RAM Sıkıştırma
            start_mem = df.memory_usage().sum() / 1024**2
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    c_min = df[col].min()
                    c_max = df[col].max()
                    col_type = str(df[col].dtype)
                    
                    if col_type.startswith('int'):
                        if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                            df[col] = df[col].astype(np.int8)
                        elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                            df[col] = df[col].astype(np.int16)
                        elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                            df[col] = df[col].astype(np.int32)
                    elif col_type.startswith('float'):
                        if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                            df[col] = df[col].astype(np.float32) 
                            
            end_mem = df.memory_usage().sum() / 1024**2
            logger.info(
                f"[{get_basename(file)}] "
                f"Cımbızlanan Satır: {pre_siphon_rows} -> {post_siphon_rows} | "
                f"RAM: {start_mem:.2f} MB -> {end_mem:.2f} MB"
            )
            
            df_list.append(df)

        if not df_list:
            raise ValueError("Okunabilen ve cımbızlanan hiçbir veri bulunamadı!")

        # Tüm dosyaları güvenle birleştir
        final_df = pd.concat(df_list, ignore_index=True)
        
        target_col = next((name for name in possible_target_names if name in final_df.columns), None)
        if target_col is None:
            raise KeyError("Kritik Hata: Birleştirme sonrasında Hedef (Target) sütunu kayboldu!")
            
        logger.info(f"Hedef (Target) Sütunu tespit edildi: '{target_col}'")
        
        y: pd.Series = final_df[target_col]
        X: pd.DataFrame = final_df.drop(target_col, axis=1)
        
        # --- KRİTİK: Label sütunundaki NaN satırları temizle ---
        nan_mask = y.notna()
        nan_count = (~nan_mask).sum()
        if nan_count > 0:
            logger.warning(f"[TEMİZLİK] Label sütununda {nan_count} adet NaN satır tespit edildi ve çıkarıldı.")
            X = X[nan_mask].reset_index(drop=True)
            y = y[nan_mask].reset_index(drop=True)
        
        logger.info(f"Nihai Akıllı Veri Seti Yüklendi. Toplam Avlanan Satır: {len(X)}, Parametre: {len(X.columns)}")
        return X, y