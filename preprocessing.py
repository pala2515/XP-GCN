# src/preprocessing.py

import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, SanitizeMol, Descriptors, rdMolDescriptors as rdescriptors
from rdkit.Chem import MACCSkeys
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Import configuration constants
from config import FP_MORGAN_RADIUS, FP_MORGAN_NBITS

# Suppress RDKit console output
RDLogger.DisableLog("rdApp.*")


def load_and_integrate_datasets(dataset_paths: list, column_map: dict) -> pd.DataFrame:
    """
    Loads, standardizes, and integrates multiple datasets into a single DataFrame.
    """
    print(f"Loading and integrating {len(dataset_paths)} datasets...")
    all_dfs = []
    for path in dataset_paths:
        try:
            df = pd.read_csv(path)
            df.rename(columns=column_map, inplace=True)
            if 'SMILES' in df.columns and 'Label' in df.columns:
                all_dfs.append(df[['SMILES', 'Label']])
            else:
                print(f"Warning: {path} is missing required columns after mapping. Skipping.")
        except FileNotFoundError:
            print(f"Warning: Dataset file not found at {path}. Skipping.")

    if not all_dfs:
        raise ValueError("No datasets were loaded. Please check file paths.")

    integrated_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Successfully integrated datasets. Total entries: {len(integrated_df)}")
    return integrated_df


def preprocess_integrated_data(df: pd.DataFrame, max_smiles_len: int) -> pd.DataFrame:
    """
    Applies pre-processing steps: removes nulls, filters by length, and deduplicates.
    """
    print("Starting data pre-processing and cleaning...")
    print(f"Initial dataset size: {len(df)} molecules.")

    df.dropna(subset=['SMILES'], inplace=True)
    df = df[df['SMILES'].str.strip() != '']
    print(f"Size after removing empty SMILES: {len(df)} molecules.")

    df = df[df['SMILES'].str.len() <= max_smiles_len]
    print(f"Size after filtering SMILES longer than {max_smiles_len} characters: {len(df)} molecules.")

    df.drop_duplicates(subset=['SMILES'], keep='first', inplace=True)
    print(f"Final size after deduplication: {len(df)} unique molecules.")

    return df


def augment_smiles_data(df: pd.DataFrame, n_augmentations: int) -> pd.DataFrame:
    """
    Expands the dataset by generating non-canonical SMILES for each molecule.
    """
    print(f"Starting SMILES augmentation to generate {n_augmentations} variants per molecule...")
    augmented_smiles_list = []
    augmented_labels_list = []

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Augmenting SMILES"):
        original_smiles = row['SMILES']
        label = row['Label']
        mol = Chem.MolFromSmiles(original_smiles)

        if mol is None:
            continue

        canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        generated_smiles = {canonical_smiles}

        max_attempts = n_augmentations * 5
        attempts = 0
        while len(generated_smiles) < n_augmentations + 1 and attempts < max_attempts:
            random_mol = Chem.MolFromSmiles(canonical_smiles)
            if random_mol:
                atom_indices = np.random.permutation(random_mol.GetNumAtoms()).tolist()
                randomized_mol = Chem.RenumberAtoms(random_mol, atom_indices)
                random_s = Chem.MolToSmiles(randomized_mol, canonical=False, isomericSmiles=True)
                if random_s and random_s not in generated_smiles:
                    generated_smiles.add(random_s)
            attempts += 1

        for smi in generated_smiles:
            augmented_smiles_list.append(smi)
            augmented_labels_list.append(label)

    augmented_df = pd.DataFrame({
        'SMILES': augmented_smiles_list,
        'Label': augmented_labels_list
    })
    print(f"Augmentation complete. Final dataset size: {len(augmented_df)} SMILES strings.")
    return augmented_df


def calculate_molecular_descriptors(smiles: str) -> list:
    """
    Calculates RDKit descriptors, ECFP (Morgan), and MACCS fingerprints.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        # 1. RDKit physicochemical descriptors
        descriptors_val = []
        desc_list = [
            Descriptors.MolWt, Descriptors.MolLogP, Descriptors.NumHDonors,
            Descriptors.NumHAcceptors, Descriptors.TPSA, Descriptors.HeavyAtomCount,
            Descriptors.NHOHCount, Descriptors.NOCount, Descriptors.NumAromaticRings,
            Descriptors.NumSaturatedRings, Descriptors.NumAliphaticRings,
            Descriptors.RingCount, rdescriptors.CalcFractionCSP3, rdescriptors.CalcNumHBA,
            rdescriptors.CalcNumHBD, rdescriptors.CalcNumRotatableBonds,
            rdescriptors.CalcNumRings, rdescriptors.CalcNumHeteroatoms
        ]
        for desc_func in desc_list:
            descriptors_val.append(desc_func(mol))

        # 2. ECFP (Morgan) Fingerprints (2048-bit)
        fp_morgan = AllChem.GetMorganFingerprintAsBitVect(mol, FP_MORGAN_RADIUS, nBits=FP_MORGAN_NBITS)

        # 3. MACCS Keys (166-bit)
        fp_maccs = MACCSkeys.GenMACCSKeys(mol)

        # Combine all features
        descriptors_val.extend(list(fp_morgan))
        descriptors_val.extend(list(fp_maccs)[1:]) # [1:] to get 166 keys

        if np.any(np.isnan(descriptors_val)) or np.any(np.isinf(descriptors_val)):
            return None
        return descriptors_val
    except Exception:
        return None


def molecule_to_graph(smiles: str) -> tuple:
    """
    Converts a SMILES string into a graph representation (node features and adjacency matrix).
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            SanitizeMol(mol)
        if mol is None:
            return None, None

        # Node features based on atom properties
        node_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum(),
                1 if atom.GetIsAromatic() else 0,
                atom.GetTotalNumHs(),
                atom.GetImplicitValence(),
                atom.GetFormalCharge(),
            ]
            node_features.append(features)

        # Adjacency matrix based on bonds
        adj_matrix = np.zeros((mol.GetNumAtoms(), mol.GetNumAtoms()), dtype=np.float32)
        for bond in mol.GetBonds():
            begin_idx, end_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            adj_matrix[begin_idx, end_idx] = 1
            adj_matrix[end_idx, begin_idx] = 1

        if not node_features:
            return None, None

        return np.array(node_features, dtype=np.float32), adj_matrix
    except Exception:
        return None, None
