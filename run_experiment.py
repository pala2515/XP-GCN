# src/run_experiment.py

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import time
import os
import warnings
from rdkit import RDLogger

# Sklearn ve Keras/Tensorflow importları
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, cohen_kappa_score, matthews_corrcoef, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Kendi modüllerimizden importlar
import config
from preprocessing import load_data, augment_and_balance_data, calculate_molecular_descriptors, molecule_to_graph
from models import EnhancedGCNModelAblation, ELMModel
from visualization import generate_visualizations

# Uyarıları ve RDKit loglarını bastır
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

def main():
    """Ana deney iş akışını çalıştırır."""
    start_time = time.time()
    
    # Sonuç klasörlerinin var olduğundan emin ol
    os.makedirs(config.RESULTS_TABLES_PATH, exist_ok=True)
    os.makedirs(config.RESULTS_PLOTS_PATH, exist_ok=True)

    # 1. Veri Yükleme ve Augmentasyon
    # ... (Orijinal koddaki Ana İş Akışı bölümü buraya uyarlanacak)
    
    # ...
    # ... Kodun geri kalanı buraya gelecek
    # ...
    
    # Önemli Değişiklikler:
    # 1. Sabit değerler yerine 'config' modülünden değişkenler kullanılacak.
    #    Örn: N_SPLITS -> config.N_SPLITS
    # 2. Fonksiyon çağrıları modül adıyla yapılacak.
    #    Örn: load_data() -> preprocessing.load_data()
    # 3. Model sınıfları 'models' modülünden çağrılacak.
    #    Örn: ELMModel -> models.ELMModel
    # 4. Sonuç dosyalarının yolları config'den alınacak.
    #    Örn: "ablation_kfold_performance_results.csv" -> os.path.join(config.RESULTS_TABLES_PATH, "ablation_kfold_performance_results.csv")
    # 5. Görselleştirme bölümü 'visualization.generate_visualizations(results_df)' çağrısıyla değiştirilecek.

    # ... Orijinal koddaki "Ana İş Akışı" bölümünün tamamını buraya taşıyın ve yukarıdaki değişiklikleri uygulayın ...
    
    end_time = time.time()
    print(f"\nToplam Çalışma Süresi: {(end_time - start_time) / 60:.2f} dakika")
    print("\n--- İşlem Tamamlandı ---")


if __name__ == '__main__':
    main()
