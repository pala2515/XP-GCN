# src/config.py

from sklearn.feature_selection import SelectKBest, f_classif, chi2

# --- Experiment Configuration ---
N_SPLITS = 10
RANDOM_STATE = 16
N_AUGMENTS_PER_SMILES = 10

# --- Feature Engineering Parameters ---
NUM_FEATURES_TO_SELECT = 100
PCA_N_COMPONENTS = 100
MAX_SMILES_LEN = 400

# --- Feature Extraction Parameters ---
FP_MORGAN_RADIUS = 2
FP_MORGAN_NBITS = 2048

# --- GCN Model Parameters (As per XP-GCN architecture) ---
GCN_HIDDEN_DIM1 = 256
GCN_HIDDEN_DIM2 = 128
GCN_HIDDEN_DIM3 = 64
GCN_DROPOUT_RATE = 0.3

# --- ELM Hyperparameters for Ablation Study (As per Table 2) ---
HIDDEN_NEURON_SIZES = [10, 25, 50, 100, 200, 500, 1000, 1500, 2000]
ACTIVATION_FUNCTIONS = ['relu', 'sigmoid', 'tanh', 'mish']
ELM_LAMBDA_REG = 0.9 # L2 regularization parameter

# --- Feature Selector Configuration ---
FEATURE_SELECTORS_CONFIG = {
    "ANOVA": SelectKBest(f_classif, k=NUM_FEATURES_TO_SELECT),
    "Chi2": SelectKBest(chi2, k=NUM_FEATURES_TO_SELECT)
}

# --- Ablation Study Scenarios ---
# These define which feature sets are fed into the ELM classifier
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
    "gcn_block123_desc": {'gcn': ['concat_123'], 'desc': True}, # Full XP-GCN Model
}

# --- File Paths (Example) ---
DATASET_PATHS = [
    "data/B3DB.csv",
    "data/MoleculeNet.csv",
    "data/LightBBB.csv",
    "data/DeePred-BBB.csv"
]
COLUMN_MAP = {
    'smiles': 'SMILES',
    'SMILES': 'SMILES',
    'Label': 'Label',
    'p_np': 'Label'
}
RESULTS_TABLES_PATH = "results/tables/"
RESULTS_PLOTS_PATH = "results/plots/"
