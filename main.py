def randomize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles);
    if mol is None: return None
    try: atom_indices=np.random.permutation(mol.GetNumAtoms()).tolist(); randomized_mol=Chem.RenumberAtoms(mol,atom_indices); return Chem.MolToSmiles(randomized_mol,canonical=False)
    except Exception: return None
def augment_smiles(smiles, n_augments=N_AUGMENTS):
    augmented_smiles = [smiles]; count = 0; attempts = 0; max_attempts = n_augments * 5
    while count < n_augments and attempts < max_attempts:
        random_smiles = randomize_smiles(smiles)
        if random_smiles and random_smiles != smiles: augmented_smiles.append(random_smiles); count += 1
        attempts += 1
    return augmented_smiles
def calculate_molecular_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles);
    if mol is None: return None
    try:
        descriptors_list = [ Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol), Descriptors.TPSA(mol), Descriptors.HeavyAtomCount(mol), Descriptors.NHOHCount(mol), Descriptors.NOCount(mol), Descriptors.NumAromaticRings(mol), Descriptors.NumSaturatedRings(mol), Descriptors.NumAliphaticRings(mol), Descriptors.RingCount(mol), rdescriptors.CalcFractionCSP3(mol), rdescriptors.CalcNumHBA(mol), rdescriptors.CalcNumHBD(mol), rdescriptors.CalcNumRotatableBonds(mol), rdescriptors.CalcNumRings(mol), rdescriptors.CalcNumHeteroatoms(mol) ]
        fp_morgan = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024); fp_maccs = MACCSkeys.GenMACCSKeys(mol)
        descriptors_list.extend([float(bit) for bit in fp_morgan.ToBitString()]); descriptors_list.extend([float(bit) for bit in fp_maccs.ToBitString()])
        return descriptors_list
    except Exception: return None
def molecule_to_graph(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles);
        if mol is not None: SanitizeMol(mol)
        if mol is None: return None, None
        node_features = []
        for atom in mol.GetAtoms(): features = [ atom.GetAtomicNum(), 1 if atom.GetIsAromatic() else 0, 1 if atom.GetHybridization() == Chem.HybridizationType.SP else 0, 1 if atom.GetHybridization() == Chem.HybridizationType.SP2 else 0, 1 if atom.GetHybridization() == Chem.HybridizationType.SP3 else 0, atom.GetFormalCharge(), atom.GetTotalNumHs() ]; node_features.append(features)
        if not node_features: return None, None
        adj_matrix = np.zeros((mol.GetNumAtoms(), mol.GetNumAtoms()), dtype=np.float32)
        for bond in mol.GetBonds(): begin_atom_idx = bond.GetBeginAtomIdx(); end_atom_idx = bond.GetEndAtomIdx(); adj_matrix[begin_atom_idx, end_atom_idx] = 1; adj_matrix[end_atom_idx, begin_atom_idx] = 1
        return np.array(node_features, dtype=np.float32), adj_matrix
    except Exception: return None, None
class ELMModel:
    def __init__(self, input_dim, hidden_dim, activation_function='relu', lambda_reg=0.9, random_state=None):
        self.input_dim = input_dim; self.hidden_dim = hidden_dim; self.activation_function = activation_function; self.lambda_reg = lambda_reg; self.random_state = random_state; self.input_weights = None; self.biases = None; self.output_weights = None
        if self.random_state is not None: np.random.seed(self.random_state)
        self.input_weights = np.random.randn(self.input_dim, self.hidden_dim); self.biases = np.random.randn(self.hidden_dim)
    def _activate(self, x):
        h_temp = x @ self.input_weights + self.biases
        if self.activation_function == 'relu': return np.maximum(0, h_temp)
        elif self.activation_function == 'sigmoid': return 1 / (1 + np.exp(-np.clip(h_temp, -500, 500)))
        elif self.activation_function == 'tanh': return np.tanh(h_temp)
        elif self.activation_function == 'mish': softplus = np.log1p(np.exp(h_temp)); tanh_softplus = np.tanh(softplus); return h_temp * tanh_softplus
        else: raise ValueError(f"Unsupported activation: {self.activation_function}")
    def _softmax(self, x): e_x = np.exp(x - np.max(x, axis=-1, keepdims=True)); return e_x / np.sum(e_x, axis=-1, keepdims=True)
    def fit(self, X, y):
        H = self._activate(X); I = np.identity(self.hidden_dim)
        try: self.output_weights = np.linalg.solve(H.T @ H + self.lambda_reg * I, H.T @ y)
        except np.linalg.LinAlgError: print("LinAlgError: Using pseudo-inverse."); self.output_weights = np.linalg.pinv(H.T @ H + self.lambda_reg * I) @ H.T @ y
    def predict_proba(self, X): H = self._activate(X); scores = H @ self.output_weights; return self._softmax(scores)
    def predict(self, X): probabilities = self.predict_proba(X); return np.argmax(probabilities, axis=-1)
def extract_graph_features_in_batches(node_features, adj_matrices, model, batch_size, return_block): # Renamed for clarity
    num_samples = node_features.shape[0]; all_features = []
    path_type = "Unknown";
    if return_block == 1: path_type = "GCN"
    elif return_block == 2: path_type = "GAT"
    elif return_block == 3: path_type = "ChebConv"
    elif return_block == 'all': path_type = "All Paths"
    print(f"  Processing {num_samples} samples in batches of {batch_size} for path '{path_type}' (block: {return_block})...")
    for i in range(0, num_samples, batch_size):
        batch_nodes = node_features[i:min(i + batch_size, num_samples)]; batch_adj = adj_matrices[i:min(i + batch_size, num_samples)]
        batch_features = model([batch_nodes, batch_adj], training=False, return_block=return_block).numpy()
        all_features.append(batch_features)
    return np.concatenate(all_features, axis=0)

# === GCN/GAT/ChebConv Modeli ===
# (Önceki cevapla aynı - GraphConvAblationModel)
class GraphConvAblationModel(keras.Model):
    def __init__(self, path1_dim, path2_dim, path3_dim, gat_heads, cheb_k, dropout_rate=0.2):
        super(GraphConvAblationModel, self).__init__()
        self.gcn1_1 = GCNConv(path1_dim, activation='relu'); self.gcn1_2 = GCNConv(path1_dim, activation='relu')
        self.dropout1 = Dropout(dropout_rate); self.pool1 = GlobalSumPool()
        self.gat2_1 = GATConv(path2_dim, attn_heads=gat_heads, concat_heads=True, dropout_rate=dropout_rate, activation='relu')
        self.gat2_2 = GATConv(path2_dim, attn_heads=gat_heads, concat_heads=False, dropout_rate=dropout_rate, activation='relu')
        self.dropout2 = Dropout(dropout_rate); self.pool2 = GlobalSumPool()
        self.cheb3_1 = ChebConv(path3_dim, K=cheb_k, activation='relu'); self.cheb3_2 = ChebConv(path3_dim, K=cheb_k, activation='relu')
        self.dropout3 = Dropout(dropout_rate); self.pool3 = GlobalSumPool(); self.concat = Concatenate()
    def call(self, inputs, training=None, return_block='all'):
        x, a = inputs; x1 = self.gcn1_1([x, a]); x1 = self.gcn1_2([x1, a]); x1 = self.dropout1(x1, training=training); out1 = self.pool1(x1)
        x2 = self.gat2_1([x, a]); x2 = self.gat2_2([x2, a]); x2 = self.dropout2(x2, training=training); out2 = self.pool2(x2)
        x3 = self.cheb3_1([x, a]); x3 = self.cheb3_2([x3, a]); x3 = self.dropout3(x3, training=training); out3 = self.pool3(x3)
        if return_block == 1: return out1
        elif return_block == 2: return out2
        elif return_block == 3: return out3
        elif return_block == 'all': return self.concat([out1, out2, out3])
        elif return_block == '1+2': return self.concat([out1, out2])
        elif return_block == '1+3': return self.concat([out1, out3])
        elif return_block == '2+3': return self.concat([out2, out3])
        else: print(f"Warning: Invalid return_block value '{return_block}'. Defaulting to 'all'."); return self.concat([out1, out2, out3])



start_time_data = time.time()
data = pd.read_csv("combined_smiles_data.csv")
print(f"Orijinal veri boyutu: {data.shape}")
data.dropna(subset=['SMILES', 'Label'], inplace=True); data['Label'] = data['Label'].astype(int)
smiles_data = data['SMILES'].tolist(); labels = data['Label'].tolist()
valid_smiles_labels = [(s, l) for s, l in zip(smiles_data, labels) if Chem.MolFromSmiles(s) is not None]
smiles_data = [s for s, l in valid_smiles_labels]; labels = [l for s, l in valid_smiles_labels]
print(f"Geçerli SMILES sonrası veri boyutu: {len(smiles_data)}")
smiles_0 = [s for s, l in zip(smiles_data, labels) if l == 0]; smiles_1 = [s for s, l in zip(smiles_data, labels) if l == 1]
if not smiles_0 or not smiles_1: raise ValueError("Bir veya her iki sınıf için başlangıç verisi yok.")
n_samples_per_class = max(len(smiles_0), len(smiles_1)); target_samples = n_samples_per_class
print(f"Sınıf başına hedef örnek sayısı (oversampling): {target_samples}")
augmented_smiles_0_with_orig = []; augmented_smiles_1_with_orig = []
for smiles in smiles_0: augmented_smiles_0_with_orig.extend(augment_smiles(smiles, n_augments=N_AUGMENTS))
for smiles in smiles_1: augmented_smiles_1_with_orig.extend(augment_smiles(smiles, n_augments=N_AUGMENTS))
if len(augmented_smiles_0_with_orig) < target_samples: smiles_0_resampled = resample(augmented_smiles_0_with_orig, replace=True, n_samples=target_samples, random_state=RANDOM_STATE)
else: smiles_0_resampled = resample(augmented_smiles_0_with_orig, replace=False, n_samples=target_samples, random_state=RANDOM_STATE)
if len(augmented_smiles_1_with_orig) < target_samples: smiles_1_resampled = resample(augmented_smiles_1_with_orig, replace=True, n_samples=target_samples, random_state=RANDOM_STATE)
else: smiles_1_resampled = resample(augmented_smiles_1_with_orig, replace=False, n_samples=target_samples, random_state=RANDOM_STATE)
augmented_smiles_data = smiles_0_resampled + smiles_1_resampled
augmented_labels = [0] * target_samples + [1] * target_samples
print(f"Augmentasyon sonrası toplam veri boyutu: {len(augmented_smiles_data)}")
print(f"Veri yükleme ve ön işleme tamamlandı. Süre: {time.time() - start_time_data:.2f} saniye")

# 2. TÜM VERİ İÇİN Özellik Hesaplama VE TANIMLAYICI SEÇİMİ
# (Önceki cevapla aynı)
print("\n2. TÜM VERİ İÇİN Moleküler Tanımlayıcıların ve Grafiklerin Hesaplanması...")
start_time_features = time.time()
all_descriptors_raw = []; all_graphs_raw = []; all_valid_indices = []
for i, smiles in enumerate(augmented_smiles_data):
    descriptors = calculate_molecular_descriptors(smiles); features, adj_matrix = molecule_to_graph(smiles)
    if descriptors is not None and features is not None and adj_matrix is not None and len(features)>0:
        all_descriptors_raw.append(descriptors); all_graphs_raw.append({'features': features, 'adj': adj_matrix}); all_valid_indices.append(i)
all_valid_labels = np.array(augmented_labels)[all_valid_indices]; all_descriptors = np.array(all_descriptors_raw, dtype=np.float32)
if np.any(np.isnan(all_descriptors)) or np.any(np.isinf(all_descriptors)):
    print("Uyarı: Tanımlayıcılarda NaN/Inf bulundu. Sütun medyanı ile dolduruluyor.")
    col_median = np.nanmedian(all_descriptors, axis=0); col_median = np.nan_to_num(col_median, nan=0.0)
    nan_indices = np.where(np.isnan(all_descriptors)); inf_indices = np.where(np.isinf(all_descriptors))
    all_descriptors[nan_indices] = np.take(col_median, nan_indices[1]); all_descriptors[inf_indices] = np.take(col_median, inf_indices[1])
print(f"Tüm geçerli örnek sayısı: {len(all_valid_labels)}"); print(f"Tüm Tanımlayıcı boyutları (seçim öncesi): {all_descriptors.shape}")
print(f"\nGlobal Chi2 Özellik Seçimi (k={SELECTKBEST_K}) Tanımlayıcılar üzerine uygulanıyor...")
enc_global = OneHotEncoder(handle_unknown='ignore', sparse_output=False); all_labels_onehot_global = enc_global.fit_transform(all_valid_labels.reshape(-1, 1)); all_labels_orig_global = np.argmax(all_labels_onehot_global, axis=1)
scaler_std_global_sel = StandardScaler(); scaler_minmax_global_sel = MinMaxScaler(); all_descriptors_scaled_for_sel = scaler_minmax_global_sel.fit_transform(scaler_std_global_sel.fit_transform(all_descriptors))
k_features_select_global = min(SELECTKBEST_K, all_descriptors_scaled_for_sel.shape[1]); all_descriptors_selected = all_descriptors; applied_global_selector = "None"
if k_features_select_global > 0 and np.all(all_descriptors_scaled_for_sel >= 0):
    try:
        selector_chi2_global = SelectKBest(chi2, k=k_features_select_global); selector_chi2_global.fit(all_descriptors_scaled_for_sel, all_labels_orig_global)
        selected_indices = selector_chi2_global.get_support(indices=True); all_descriptors_selected = all_descriptors[:, selected_indices]
        applied_global_selector = f"Chi2(k={all_descriptors_selected.shape[1]})"; print(f"Tanımlayıcılar için Chi2 seçimi yapıldı. Seçilen özellik sayısı: {all_descriptors_selected.shape[1]}")
    except ValueError as e: print(f"Global Chi2 seçimi sırasında hata: {e}. Tüm tanımlayıcılar kullanılacak.")
else: print("Global Chi2 seçimi atlandı. Tüm tanımlayıcılar kullanılacak.")
if not all_graphs_raw: raise ValueError("Hesaplanacak geçerli grafik bulunamadı!")
max_atoms = max(len(g['features']) for g in all_graphs_raw) if all_graphs_raw else 0; num_node_features = all_graphs_raw[0]['features'].shape[1] if all_graphs_raw else 0
print(f"Grafiklerde maksimum atom sayısı: {max_atoms}"); print(f"Düğüm özellik sayısı: {num_node_features}")
all_node_features_padded = pad_sequences([g['features'] for g in all_graphs_raw], maxlen=max_atoms, padding="post", dtype='float32', value=0.0)
all_adj_matrices_padded = np.array([np.pad(g['adj'], ((0, max_atoms - g['adj'].shape[0]), (0, max_atoms - g['adj'].shape[1])), 'constant', constant_values=0.0) for g in all_graphs_raw], dtype=np.float32)
print(f"Padding sonrası Node Features şekli: {all_node_features_padded.shape}"); print(f"Padding sonrası Adj Matrices şekli: {all_adj_matrices_padded.shape}")
print(f"Tanımlayıcı/Grafik hesaplama ve global tanımlayıcı seçimi tamamlandı. Süre: {time.time() - start_time_features:.2f} saniye")

# 3. TÜM VERİ İÇİN GCN/GAT/ChebConv Özellik Çıkarımı
# (Önceki cevapla aynı)
print("\n3. TÜM VERİ İÇİN GCN/GAT/ChebConv ile Özellik Çıkarımı (Batch Processing)...")
start_time_graph_conv = time.time()
graph_conv_model = GraphConvAblationModel(PATH1_HIDDEN_DIM, PATH2_HIDDEN_DIM, PATH3_HIDDEN_DIM, GAT_ATTN_HEADS, CHEB_K, DROPOUT_RATE)
if all_node_features_padded.shape[0] > 0: _ = graph_conv_model([all_node_features_padded[:1], all_adj_matrices_padded[:1]], return_block='all')
else: raise ValueError("No data available for Graph Conv model building.")
print("  Extracting features for ALL data...")
all_path1_feat = extract_graph_features_in_batches(all_node_features_padded, all_adj_matrices_padded, graph_conv_model, BATCH_SIZE_GCN, return_block=1) # GCN
all_path2_feat = extract_graph_features_in_batches(all_node_features_padded, all_adj_matrices_padded, graph_conv_model, BATCH_SIZE_GCN, return_block=2) # GAT
all_path3_feat = extract_graph_features_in_batches(all_node_features_padded, all_adj_matrices_padded, graph_conv_model, BATCH_SIZE_GCN, return_block=3) # Cheb
print(f"\n  Path 1 (GCN) çıktı boyutu: {all_path1_feat.shape}"); print(f"  Path 2 (GAT) çıktı boyutu: {all_path2_feat.shape}"); print(f"  Path 3 (Cheb) çıktı boyutu: {all_path3_feat.shape}")
all_path1_2_feat = np.concatenate((all_path1_feat, all_path2_feat), axis=1); all_path1_3_feat = np.concatenate((all_path1_feat, all_path3_feat), axis=1)
all_path2_3_feat = np.concatenate((all_path2_feat, all_path3_feat), axis=1); all_paths_feat = np.concatenate((all_path1_feat, all_path2_feat, all_path3_feat), axis=1)
all_paths_desc_feat = np.concatenate((all_paths_feat, all_descriptors_selected), axis=1)
print(f"TÜM VERİ için GCN/GAT/ChebConv özellik çıkarımı tamamlandı. Süre: {time.time() - start_time_graph_conv:.2f} saniye")

# === TÜM VERİ İÇİN Ablasyon Çalışmaları Özellik Setleri (GAT/Cheb dahil) ===
# (Önceki cevapla aynı)
all_feature_sets = { "GCN1": all_path1_feat, "GCN2": all_path2_feat, "GCN3": all_path3_feat, "Descriptors": all_descriptors_selected, "GCN1+GCN2": all_path1_2_feat, "GCN1+GCN3": all_path1_3_feat, "GCN2+GCN3": all_path2_3_feat, "GCN_All": all_paths_feat, "GCN_All+Descriptors": all_paths_desc_feat }
print("\nÖzellik setleri oluşturuldu (GAT/ChebConv dahil):"); [print(f"  {name}: {data.shape}") for name, data in all_feature_sets.items()]

# 4. K-Fold Cross Validation ve ELM ile Sınıflandırma (EĞİTİM METRİKLERİ EKLENDİ)
print(f"\n4. {N_SPLITS}-Fold Cross Validation ile Ablasyon Çalışmaları ve ELM Sınıflandırması...")
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
cv_results = [] # Tüm hiperparametre/config sonuçlarını toplar

for config_name, all_features_raw in all_feature_sets.items():
    if all_features_raw is None or all_features_raw.shape[0] == 0 or all_features_raw.shape[1] == 0: print(f"\n--- Konfigürasyon {config_name} atlanıyor (geçersiz özellik seti) ---"); continue
    config_verbose_name = config_name.replace('GCN1','GCN').replace('GCN2','GAT').replace('GCN3','Cheb')
    print(f"\n--- Çalıştırılan Konfigürasyon: {config_name} (Paths: {config_verbose_name}) ---")
    print(f"  Özellik Boyutu: {all_features_raw.shape}"); start_time_config = time.time()
    for hidden_size in ELM_HIDDEN_SIZES:
        for activation in ELM_ACTIVATIONS:
            # Her hiperparametre seti için katlama metriklerini sıfırla
            fold_val_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'log_loss': [], 'kappa': [], 'mcc': [], 'roc_auc': []}
            fold_train_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'log_loss': [], 'kappa': [], 'mcc': [], 'roc_auc': []} # EĞİTİM METRİKLERİ İÇİN

            for fold, (train_idx, val_idx) in enumerate(kf.split(all_features_raw, all_labels_orig_global)):
                X_train_fold_raw, X_val_fold_raw = all_features_raw[train_idx], all_features_raw[val_idx]
                y_train_fold_orig, y_val_fold_orig = all_labels_orig_global[train_idx], all_labels_orig_global[val_idx]
                y_train_fold_onehot = all_labels_onehot_global[train_idx]
                try:
                    # Ölçekleme (Fold üzerinde)
                    scaler_std = StandardScaler(); X_train_fold_scaled = scaler_std.fit_transform(X_train_fold_raw); X_val_fold_scaled = scaler_std.transform(X_val_fold_raw)
                    scaler_minmax = MinMaxScaler(); X_train_fold_processed = scaler_minmax.fit_transform(X_train_fold_scaled); X_val_fold_processed = scaler_minmax.transform(X_val_fold_scaled)

                    # Özellik seçimi yok (globalde yapıldı)

                    if X_train_fold_processed is None or X_train_fold_processed.shape[1] == 0:
                        # Hata durumunda hem train hem val için NaN ekle
                        for key in fold_val_metrics: fold_val_metrics[key].append(np.nan)
                        for key in fold_train_metrics: fold_train_metrics[key].append(np.nan)
                        continue

                    # ELM Eğitimi
                    input_dim_elm = X_train_fold_processed.shape[1]
                    elm_model = ELMModel(input_dim=input_dim_elm, hidden_dim=hidden_size, activation_function=activation, lambda_reg=ELM_LAMBDA_REG, random_state=RANDOM_STATE + fold)
                    elm_model.fit(X_train_fold_processed, y_train_fold_onehot)

                    # Doğrulama Metrikleri
                    val_predictions = elm_model.predict(X_val_fold_processed); val_proba = elm_model.predict_proba(X_val_fold_processed)
                    fold_val_metrics['accuracy'].append(accuracy_score(y_val_fold_orig, val_predictions))
                    fold_val_metrics['precision'].append(precision_score(y_val_fold_orig, val_predictions, zero_division=0))
                    fold_val_metrics['recall'].append(recall_score(y_val_fold_orig, val_predictions, zero_division=0))
                    fold_val_metrics['f1'].append(f1_score(y_val_fold_orig, val_predictions, zero_division=0))
                    try: fold_val_metrics['log_loss'].append(log_loss(y_val_fold_orig, val_proba, eps=1e-15))
                    except ValueError: fold_val_metrics['log_loss'].append(np.nan)
                    fold_val_metrics['kappa'].append(cohen_kappa_score(y_val_fold_orig, val_predictions))
                    fold_val_metrics['mcc'].append(matthews_corrcoef(y_val_fold_orig, val_predictions))
                    if len(np.unique(y_val_fold_orig)) > 1: fold_val_metrics['roc_auc'].append(roc_auc_score(y_val_fold_orig, val_proba[:, 1]))
                    else: fold_val_metrics['roc_auc'].append(0.5)

                    # <<< EĞİTİM METRİKLERİ HESAPLAMA >>>
                    train_predictions_fold = elm_model.predict(X_train_fold_processed)
                    train_proba_fold = elm_model.predict_proba(X_train_fold_processed)
                    fold_train_metrics['accuracy'].append(accuracy_score(y_train_fold_orig, train_predictions_fold))
                    fold_train_metrics['precision'].append(precision_score(y_train_fold_orig, train_predictions_fold, zero_division=0))
                    fold_train_metrics['recall'].append(recall_score(y_train_fold_orig, train_predictions_fold, zero_division=0))
                    fold_train_metrics['f1'].append(f1_score(y_train_fold_orig, train_predictions_fold, zero_division=0))
                    try: fold_train_metrics['log_loss'].append(log_loss(y_train_fold_orig, train_proba_fold, eps=1e-15))
                    except ValueError: fold_train_metrics['log_loss'].append(np.nan)
                    fold_train_metrics['kappa'].append(cohen_kappa_score(y_train_fold_orig, train_predictions_fold))
                    fold_train_metrics['mcc'].append(matthews_corrcoef(y_train_fold_orig, train_predictions_fold))
                    if len(np.unique(y_train_fold_orig)) > 1: fold_train_metrics['roc_auc'].append(roc_auc_score(y_train_fold_orig, train_proba_fold[:, 1]))
                    else: fold_train_metrics['roc_auc'].append(0.5)
                    # <<< EĞİTİM METRİKLERİ HESAPLAMA SONU >>>

                except np.linalg.LinAlgError as e:
                     print(f"\n Fold {fold+1} LinAlgError (H:{hidden_size}, A:{activation}): {e}. Appending NaNs.")
                     for key in fold_val_metrics: fold_val_metrics[key].append(np.nan)
                     for key in fold_train_metrics: fold_train_metrics[key].append(np.nan) # Train için de NaN ekle
                except Exception as e:
                     print(f"\n Fold {fold+1} Error (H:{hidden_size}, A:{activation}): {e}. Appending NaNs.")
                     for key in fold_val_metrics: fold_val_metrics[key].append(np.nan)
                     for key in fold_train_metrics: fold_train_metrics[key].append(np.nan) # Train için de NaN ekle

            # K-Fold döngüsü sonrası ortalamaları hesapla
            if sum(~np.isnan(fold_val_metrics['accuracy'])) > 0: # Başarılı katlama varsa
                result_row = {
                    'config': config_name, 'hidden_size': hidden_size, 'activation': activation,
                    # Doğrulama Metrikleri (Ortalama ve Std)
                    'mean_val_accuracy': np.nanmean(fold_val_metrics['accuracy']), 'std_val_accuracy': np.nanstd(fold_val_metrics['accuracy']),
                    'mean_val_precision': np.nanmean(fold_val_metrics['precision']), 'std_val_precision': np.nanstd(fold_val_metrics['precision']),
                    'mean_val_recall': np.nanmean(fold_val_metrics['recall']), 'std_val_recall': np.nanstd(fold_val_metrics['recall']),
                    'mean_val_f1': np.nanmean(fold_val_metrics['f1']), 'std_val_f1': np.nanstd(fold_val_metrics['f1']),
                    'mean_val_log_loss': np.nanmean(fold_val_metrics['log_loss']), 'std_val_log_loss': np.nanstd(fold_val_metrics['log_loss']),
                    'mean_val_kappa': np.nanmean(fold_val_metrics['kappa']), 'std_val_kappa': np.nanstd(fold_val_metrics['kappa']),
                    'mean_val_mcc': np.nanmean(fold_val_metrics['mcc']), 'std_val_mcc': np.nanstd(fold_val_metrics['mcc']),
                    'mean_val_roc_auc': np.nanmean(fold_val_metrics['roc_auc']), 'std_val_roc_auc': np.nanstd(fold_val_metrics['roc_auc']),
                    # <<< EĞİTİM METRİKLERİ (Ortalama ve Std) >>>
                    'mean_train_accuracy': np.nanmean(fold_train_metrics['accuracy']), 'std_train_accuracy': np.nanstd(fold_train_metrics['accuracy']),
                    'mean_train_precision': np.nanmean(fold_train_metrics['precision']), 'std_train_precision': np.nanstd(fold_train_metrics['precision']),
                    'mean_train_recall': np.nanmean(fold_train_metrics['recall']), 'std_train_recall': np.nanstd(fold_train_metrics['recall']),
                    'mean_train_f1': np.nanmean(fold_train_metrics['f1']), 'std_train_f1': np.nanstd(fold_train_metrics['f1']),
                    'mean_train_log_loss': np.nanmean(fold_train_metrics['log_loss']), 'std_train_log_loss': np.nanstd(fold_train_metrics['log_loss']),
                    'mean_train_kappa': np.nanmean(fold_train_metrics['kappa']), 'std_train_kappa': np.nanstd(fold_train_metrics['kappa']),
                    'mean_train_mcc': np.nanmean(fold_train_metrics['mcc']), 'std_train_mcc': np.nanstd(fold_train_metrics['mcc']),
                    'mean_train_roc_auc': np.nanmean(fold_train_metrics['roc_auc']), 'std_train_roc_auc': np.nanstd(fold_train_metrics['roc_auc']),
                    # <<< EĞİTİM METRİKLERİ SONU >>>
                    'n_successful_folds': sum(~np.isnan(fold_val_metrics['accuracy']))
                }
                cv_results.append(result_row)
    print(f"--- Konfigürasyon {config_name} tamamlandı. Süre: {time.time() - start_time_config:.2f} saniye ---")
