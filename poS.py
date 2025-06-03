import numpy as np
import tensorflow as tf

def positional_encoding(seq_len, d_model):
    """
    Computes sinusoidal positional encodings for a sequence.
    
    Args:
    - seq_len: Length of the sequence (number of words/tokens).
    - d_model: Embedding size (must match word embedding size).
    
    Returns:
    - Tensor of shape (seq_len, d_model) containing positional encodings.
    """
    pos = np.arange(seq_len)[:, np.newaxis]  # Shape (seq_len, 1)
    i = np.arange(d_model)[np.newaxis, :]  # Shape (1, d_model)
    
    angles = pos / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    
    # Apply sine to even indices, cosine to odd indices
    pos_encoding = np.zeros((seq_len, d_model))
    pos_encoding[:, 0::2] = np.sin(angles[:, 0::2])  # Apply sin to even indices
    pos_encoding[:, 1::2] = np.cos(angles[:, 1::2])  # Apply cos to odd indices
    
    return tf.convert_to_tensor(pos_encoding, dtype=tf.float32)

# Example Usage
seq_len = 10  # Sequence length (10 words)
d_model = 8   # Embedding size (8 dimensions)

pe = positional_encoding(seq_len, d_model)
print("🔹 Positional Encoding:\n", pe.numpy())

import matplotlib.pyplot as plt

seq_len = 100  # 100 words
d_model = 16   # 16-dimensional embeddings
pe = positional_encoding(seq_len, d_model).numpy()

plt.figure(figsize=(10, 5))
for i in range(4):  # Plot first 4 dimensions
    plt.plot(pe[:, i], label=f"Dim {i}")
plt.legend()
plt.title("Positional Encoding (First 4 Dimensions)")
plt.xlabel("Position")
plt.ylabel("Encoding Value")
plt.show()

# word_embeddings = tf.random.uniform((seq_len, d_model))  # Example word embeddings
# pos_encoding = positional_encoding(seq_len, d_model)  # Compute positional encodings
# encoded_inputs = word_embeddings + pos_encoding  # Add positional information to embeddings
