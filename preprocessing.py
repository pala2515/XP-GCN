# src/preprocessing.py

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, SanitizeMol, Descriptors, rdMolDescriptors as rdescriptors
from rdkit.Chem import MACCSkeys
from sklearn.utils import resample

def load_data(filepath):
    """Veri setini belirtilen yoldan yükler."""
    return pd.read_csv(filepath)

def randomize_smiles(smiles):
    """Verilen bir SMILES dizesini rastgele hale getirir."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    try:
        atom_indices = np.random.permutation(mol.GetNumAtoms()).tolist()
        randomized_mol = Chem.RenumberAtoms(mol, atom_indices)
        return Chem.MolToSmiles(randomized_mol, canonical=False)
    except Exception: return None

def augment_and_balance_data(smiles_data, labels, n_augments=2, random_state=42):
    """SMILES verisini augmentasyon ve yeniden örnekleme ile dengeler."""
    # ... (Orijinal koddaki fonksiyonun içeriği buraya aynen kopyalanacak)
    smiles_0 = [s for s, l in zip(smiles_data, labels) if l == 0]
    smiles_1 = [s for s, l in zip(smiles_data, labels) if l == 1]
    n_samples = max(len(smiles_0), len(smiles_1))

    final_smiles = []
    final_labels = []

    for label_val, smiles_list in zip([0, 1], [smiles_0, smiles_1]):
        augmented_for_class = []
        for smiles in smiles_list:
            augmented_for_class.append(smiles) # Orijinali ekle
            for _ in range(n_augments):
                aug_smiles = randomize_smiles(smiles)
                if aug_smiles: augmented_for_class.append(aug_smiles)

        current_samples = len(augmented_for_class)
        if current_samples == 0: continue

        resample_replace = current_samples < n_samples
        resampled_smiles = resample(augmented_for_class,
                                     replace=resample_replace,
                                     n_samples=n_samples,
                                     random_state=random_state)

        final_smiles.extend(resampled_smiles)
        final_labels.extend([label_val] * n_samples)

    return final_smiles, final_labels

def calculate_molecular_descriptors(smiles):
    """Bir SMILES dizesinden moleküler tanımlayıcıları ve parmak izlerini hesaplar."""
    # ... (Orijinal koddaki fonksiyonun içeriği buraya aynen kopyalanacak)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    try:
        descriptors_val = []
        desc_list = [
            Descriptors.MolWt, Descriptors.MolLogP, Descriptors.NumHDonors, Descriptors.NumHAcceptors,
            Descriptors.TPSA, Descriptors.HeavyAtomCount, Descriptors.NHOHCount, Descriptors.NOCount,
            Descriptors.NumAromaticRings, Descriptors.NumSaturatedRings, Descriptors.NumAliphaticRings,
            Descriptors.RingCount, rdescriptors.CalcFractionCSP3, rdescriptors.CalcNumHBA,
            rdescriptors.CalcNumHBD, rdescriptors.CalcNumRotatableBonds, rdescriptors.CalcNumRings,
            rdescriptors.CalcNumHeteroatoms
        ]
        for desc_func in desc_list: descriptors_val.append(desc_func(mol))
        fp_morgan = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        fp_maccs = MACCSkeys.GenMACCSKeys(mol)
        descriptors_val += list(fp_morgan) + list(fp_maccs)
        if np.any(np.isnan(descriptors_val)) or np.any(np.isinf(descriptors_val)): return None
        return descriptors_val
    except Exception: return None


def molecule_to_graph(smiles):
    """Bir SMILES dizesini grafik gösterimine (düğüm özellikleri ve bitişiklik matrisi) dönüştürür."""
    # ... (Orijinal koddaki fonksiyonun içeriği buraya aynen kopyalanacak)
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None: SanitizeMol(mol)
        if mol is None: return None, None
        node_features_val = []
        for atom in mol.GetAtoms():
            features = [atom.GetAtomicNum(), 1 if atom.GetIsAromatic() else 0,
                        atom.GetTotalNumHs(), atom.GetImplicitValence()]
            node_features_val.append(features)
        adj_matrix_val = np.zeros((mol.GetNumAtoms(), mol.GetNumAtoms()), dtype=np.float32)
        for bond in mol.GetBonds():
            begin_idx, end_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            adj_matrix_val[begin_idx, end_idx] = 1
            adj_matrix_val[end_idx, begin_idx] = 1
        if not node_features_val: return None, None
        return np.array(node_features_val, dtype=np.float32), adj_matrix_val
    except Exception: return None, None
