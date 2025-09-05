# src/config.py

from sklearn.feature_selection import SelectKBest, f_classif, chi2

# --- Deney Yapılandırması ---
N_SPLITS = 5  # K-Fold için kat sayısı
RANDOM_STATE = 16
N_AUGMENTS_PER_SMILES = 2
NUM_FEATURES_TO_SELECT = 100  # Özellik seçimi sayısı
PCA_N_COMPONENTS = 100  # PCA bileşen sayısı

# --- Dosya Yolları ---
INPUT_DATA_PATH = "data/combined_smiles_data.csv"
RESULTS_TABLES_PATH = "results/tables/"
RESULTS_PLOTS_PATH = "results/plots/"

# --- Hiperparametreler ---
# hidden_neuron_sizes = [50, 100, 200, 500, 1000] # Daha hızlı çalıştırma için
# activation_functions = ['relu', 'tanh'] # Daha hızlı çalıştırma için
HIDDEN_NEURON_SIZES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000]
ACTIVATION_FUNCTIONS = ['relu', 'sigmoid', 'tanh', 'mish']
FEATURE_SELECTORS_CONFIG = {
    "ANOVA": SelectKBest(f_classif, k=NUM_FEATURES_TO_SELECT),
    "Chi2": SelectKBest(chi2, k=NUM_FEATURES_TO_SELECT)
}

# --- Ablasyon Senaryoları ---
ABLATION_CONFIGS = {
    "gcn_block1": {'gcn': ['block1'], 'desc': False},
    "gcn_block2": {'gcn': ['block2'], 'desc': False},
    "gcn_block3": {'gcn': ['block3'], 'desc': False},
    "gcn_block12": {'gcn': ['concat_12'], 'desc': False},
    "gcn_block123": {'gcn': ['concat_123'], 'desc': False},
    "descriptors_only": {'gcn': [], 'desc': True},
    "gcn_block1_desc": {'gcn': ['block1'], 'desc': True},
    "gcn_block2_desc": {'gcn': ['block2'], 'desc': True},
    "gcn_block3_desc": {'gcn': ['block3'], 'desc': True},
    "gcn_block12_desc": {'gcn': ['concat_12'], 'desc': True},
    "gcn_block123_desc": {'gcn': ['concat_123'], 'desc': True},  # Orijinal tam model
}

# --- GCN Model Parametreleri ---
GCN_HIDDEN_DIM1 = 256
GCN_HIDDEN_DIM2 = 128
GCN_HIDDEN_DIM3 = 64
GCN_DROPOUT_RATE = 0.3
