import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Concatenate, Dropout
from spektral.layers import GCNConv, GlobalSumPool

# Import configuration constants for default values
from config import GCN_HIDDEN_DIM1, GCN_HIDDEN_DIM2, GCN_HIDDEN_DIM3, GCN_DROPOUT_RATE, ELM_LAMBDA_REG

class EnhancedGCNModelAblation(keras.Model):
    """
    Implements the parallel GCN feature extraction component of the XP-GCN model.

    This architecture consists of three parallel GCN blocks, each with three
    convolutional layers. Each block is designed to capture features at a
    different dimensionality (256, 128, and 64), which are then pooled and
    made available for concatenation. This structure supports comprehensive
    ablation studies by providing access to the output of each block individually
    as well as their combined representations.
    """
    def __init__(self, node_feature_dim, hidden_dim1=GCN_HIDDEN_DIM1,
                 hidden_dim2=GCN_HIDDEN_DIM2, hidden_dim3=GCN_HIDDEN_DIM3,
                 dropout_rate=GCN_DROPOUT_RATE):
        super(EnhancedGCNModelAblation, self).__init__()
        # Block 1: 256-dimension feature space
        self.gcn1_block1 = GCNConv(hidden_dim1, activation='relu')
        self.gcn2_block1 = GCNConv(hidden_dim1, activation='relu')
        self.gcn3_block1 = GCNConv(hidden_dim1, activation='relu')
        self.dropout1 = Dropout(dropout_rate)
        self.pool1 = GlobalSumPool()

        # Block 2: 128-dimension feature space
        self.gcn1_block2 = GCNConv(hidden_dim2, activation='relu')
        self.gcn2_block2 = GCNConv(hidden_dim2, activation='relu')
        self.gcn3_block2 = GCNConv(hidden_dim2, activation='relu')
        self.dropout2 = Dropout(dropout_rate)
        self.pool2 = GlobalSumPool()

        # Block 3: 64-dimension feature space
        self.gcn1_block3 = GCNConv(hidden_dim3, activation='relu')
        self.gcn2_block3 = GCNConv(hidden_dim3, activation='relu')
        self.gcn3_block3 = GCNConv(hidden_dim3, activation='relu')
        self.dropout3 = Dropout(dropout_rate)
        self.pool3 = GlobalSumPool()

        # Concatenation layers for combining block outputs
        self.concat_12 = Concatenate(name='concat_12')
        self.concat_123 = Concatenate(name='concat_123')

    def call(self, inputs, training=None):
        x, a = inputs

        # Block 1 Computation -> Pooling
        x1 = self.gcn1_block1([x, a])
        x1 = self.gcn2_block1([x1, a])
        x1 = self.gcn3_block1([x1, a])
        x1 = self.dropout1(x1, training=training)
        x1_pooled = self.pool1(x1)

        # Block 2 Computation -> Pooling
        x2 = self.gcn1_block2([x, a])
        x2 = self.gcn2_block2([x2, a])
        x2 = self.gcn3_block2([x2, a])
        x2 = self.dropout2(x2, training=training)
        x2_pooled = self.pool2(x2)

        # Block 3 Computation -> Pooling
        x3 = self.gcn1_block3([x, a])
        x3 = self.gcn2_block3([x3, a])
        x3 = self.gcn3_block3([x3, a])
        x3 = self.dropout3(x3, training=training)
        x3_pooled = self.pool3(x3)

        # Combined outputs for different ablation scenarios
        concat_12 = self.concat_12([x1_pooled, x2_pooled])
        concat_123 = self.concat_123([x1_pooled, x2_pooled, x3_pooled])

        # Return all relevant outputs in a dictionary for flexible use
        return {
            'block1': x1_pooled,
            'block2': x2_pooled,
            'block3': x3_pooled,
            'concat_12': concat_12,
            'concat_123': concat_123
        }


class ELMModel:
    """
    An implementation of the Extreme Learning Machine (ELM) classifier.

    ELM is a feed-forward neural network with a single hidden layer where the
    hidden node weights are randomly assigned and never updated. The output
    weights are learned analytically, making the training process extremely fast.
    This serves as the final classification component in the XP-GCN model.
    """
    def __init__(self, input_dim, hidden_dim, activation_function='relu',
                 lambda_reg=ELM_LAMBDA_REG, random_state=None):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation_function = activation_function
        self.lambda_reg = lambda_reg # L2 Regularization
        self.random_state = random_state
        self.hidden_weights = None
        self.output_weights = None

        if self.random_state is not None:
            np.random.seed(self.random_state)
        self.hidden_weights = np.random.randn(self.input_dim, self.hidden_dim)

    def _activate(self, x):
        H_temp = x @ self.hidden_weights
        if self.activation_function == 'relu':
            return np.maximum(0, H_temp)
        if self.activation_function == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(H_temp, -50, 50)))
        if self.activation_function == 'tanh':
            return np.tanh(H_temp)
        if self.activation_function == 'mish':
             softplus_x = np.log1p(np.exp(np.clip(H_temp, -50, 50)))
             return H_temp * np.tanh(softplus_x)
        raise ValueError(f"Unsupported activation function: {self.activation_function}")

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def fit(self, X, y_onehot):
        H = self._activate(X)
        H_T_H = H.T @ H
        lambda_I = self.lambda_reg * np.identity(self.hidden_dim)
        try:
            # More stable approach using np.linalg.solve
            self.output_weights = np.linalg.solve(H_T_H + lambda_I, H.T @ y_onehot)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if matrix is singular
            self.output_weights = np.linalg.pinv(H_T_H + lambda_I) @ H.T @ y_onehot

    def predict_proba(self, X):
        H = self._activate(X)
        scores = H @ self.output_weights
        return self._softmax(scores)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=-1)
