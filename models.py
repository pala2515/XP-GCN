# src/models.py

import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Concatenate, Dropout
from spektral.layers import GCNConv, GlobalSumPool

# --- 5. GCN Modeli (Ablasyon için Güncellendi) ---
class EnhancedGCNModelAblation(keras.Model):
    # ... (Orijinal koddaki sınıfın içeriği buraya aynen kopyalanacak)
    def __init__(self, node_feature_dim, hidden_dim1, hidden_dim2, hidden_dim3, dropout_rate=0.2):
        super(EnhancedGCNModelAblation, self).__init__()
        # ... (içerik)

# 6. ELM Modeli
class ELMModel:
    # ... (Orijinal koddaki sınıfın içeriği buraya aynen kopyalanacak)
    def __init__(self, input_dim, hidden_dim, activation_function='relu', lambda_reg=0.9, random_state=None):
        # ... (içerik)

# Not: Sınıf içeriklerini orijinal koddan kopyalayıp yapıştırın.
# Yukarıdaki koda tam içerikleri eklemedim, sadece iskeleti verdim.
# Orijinal kodunuzdaki EnhancedGCNModelAblation ve ELMModel sınıflarını buraya taşıyın.
