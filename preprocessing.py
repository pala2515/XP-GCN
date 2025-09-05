import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, SanitizeMol, Descriptors, rdMolDescriptors as rdescriptors
from rdkit.Chem import MACCSkeys
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Suppress RDKit console output
RDLogger.DisableLog("rdApp.*")


def load_and_integrate_datasets(dataset_paths: list, column_map: dict) -> pd.DataFrame:
    """
    Loads multiple datasets from specified paths, standardizes column names,
    and integrates them into a single DataFrame.

    Args:
        dataset_paths (list): A list of file paths to the CSV datasets.
        column_map (dict): A dictionary to map original column names to
                           standard names ('SMILES', 'Label').

    Returns:
        pd.DataFrame: A single DataFrame containing the combined data.
    """
    print(f"Loading and integrating {len(dataset_paths)} datasets...")
    all_dfs = []
    for path in dataset_paths:
        try:
            df = pd.read_csv(path)
            # Standardize column names using the provided map
            df.rename(columns=column_map, inplace=True)
            # Ensure only required columns are kept
            if 'SMILES' in df.columns and 'Label' in df.columns:
                all_dfs.append(df[['SMILES', 'Label']])
            else:
                print(f"Warning: {path} is missing 'SMILES' or 'Label' column after mapping. Skipping.")
        except FileNotFoundError:
            print(f"Warning: Dataset file not found at {path}. Skipping.")

    if not all_dfs:
        raise ValueError("No datasets were loaded. Please check file paths.")

    integrated_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Successfully integrated datasets. Total entries: {len(integrated_df)}")
    return integrated_df


def preprocess_integrated_data(df: pd.DataFrame, max_smiles_len: int = 400) -> pd.DataFrame:
    """
    Applies a series of pre-processing steps to the integrated dataset.

    Steps:
    1. Removes rows with empty or incomplete SMILES strings.
    2. Filters out SMILES strings that exceed a maximum character length.
    3. Performs deduplication to ensure each unique chemical structure is represented only once.

    Args:
        df (pd.DataFrame): The integrated DataFrame with 'SMILES' and 'Label' columns.
        max_smiles_len (int): The maximum allowed character length for a SMILES string.

    Returns:
        pd.DataFrame: A cleaned and deduplicated DataFrame.
    """
    print("Starting data pre-processing and cleaning...")
    print(f"Initial dataset size: {len(df)} molecules.")

    # 1. Remove empty and incomplete SMILES
    df.dropna(subset=['SMILES'], inplace=True)
    df = df[df['SMILES'].str.strip() != '']
    print(f"Size after removing empty SMILES: {len(df)} molecules.")

    # 2. Filter by SMILES length
    df = df[df['SMILES'].str.len() <= max_smiles_len]
    print(f"Size after filtering SMILES longer than {max_smiles_len} characters: {len(df)} molecules.")

    # 3. Deduplication based on SMILES string
    df.drop_duplicates(subset=['SMILES'], keep='first', inplace=True)
    print(f"Final size after deduplication: {len(df)} unique molecules.")

    return df


def randomize_smiles(smiles: str) -> str:
    """
    Generates a non-canonical, randomized SMILES representation for a given molecule.

    Args:
        smiles (str): A valid SMILES string.

    Returns:
        str: A randomized SMILES string, or None if the process fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        atom_indices = np.random.permutation(mol.GetNumAtoms()).tolist()
        randomized_mol = Chem.RenumberAtoms(mol, atom_indices)
        return Chem.MolToSmiles(randomized_mol, canonical=False, isomericSmiles=True)
    except Exception:
        return None


def augment_smiles_data(df: pd.DataFrame, n_augmentations: int = 10) -> pd.DataFrame:
    """
    Expands the dataset by generating multiple non-canonical SMILES for each molecule.
    For each molecule, the final set includes its canonical SMILES plus n_augmentations
    unique, non-canonical variants.

    Args:
        df (pd.DataFrame): The pre-processed DataFrame of unique molecules.
        n_augmentations (int): The number of non-canonical versions to generate per molecule.

    Returns:
        pd.DataFrame: An augmented DataFrame with (original_size * (n_augmentations + 1)) rows.
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

        # 1. Add the canonical SMILES representation
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        generated_smiles = {canonical_smiles}

        # 2. Generate non-canonical SMILES representations
        max_attempts = n_augmentations * 5  # Set a reasonable attempt limit
        attempts = 0
        while len(generated_smiles) < n_augmentations + 1 and attempts < max_attempts:
            random_s = randomize_smiles(canonical_smiles)
            if random_s and random_s not in generated_smiles:
                generated_smiles.add(random_s)
            attempts += 1

        # Add all generated unique SMILES to the final list
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
    Calculates a set of molecular descriptors and fingerprints for a given SMILES string.

    Args:
        smiles (str): A valid SMILES string.

    Returns:
        list: A list of numerical features, or None if calculation fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
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

        fp_morgan = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        fp_maccs = MACCSkeys.GenMACCSKeys(mol)
        descriptors_val.extend(list(fp_morgan))
        descriptors_val.extend(list(fp_maccs))

        if np.any(np.isnan(descriptors_val)) or np.any(np.isinf(descriptors_val)):
            return None
        return descriptors_val
    except Exception:
        return None


def molecule_to_graph(smiles: str) -> tuple:
    """
    Converts a SMILES string into a graph representation (node features and adjacency matrix).

    Args:
        smiles (str): A valid SMILES string.

    Returns:
        tuple: A tuple containing the node features array and the adjacency matrix.
               Returns (None, None) on failure.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            SanitizeMol(mol)
        if mol is None:
            return None, None

        node_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum(),
                1 if atom.GetIsAromatic() else 0,
                atom.GetTotalNumHs(),
                atom.GetImplicitValence()
            ]
            node_features.append(features)

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
