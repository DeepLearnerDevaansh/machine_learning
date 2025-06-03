import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def multi_head_attention(X, num_heads=2):
    seq_len, d_model = X.shape
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    
    depth = d_model // num_heads  # Size of each head
    
    # Trainable weight matrices for multiple heads
    W_q = [tf.Variable(tf.random.uniform((d_model, depth)), name=f"W_q_{i}") for i in range(num_heads)]
    W_k = [tf.Variable(tf.random.uniform((d_model, depth)), name=f"W_k_{i}") for i in range(num_heads)]
    W_v = [tf.Variable(tf.random.uniform((d_model, depth)), name=f"W_v_{i}") for i in range(num_heads)]
    
    W_o = tf.Variable(tf.random.uniform((d_model, d_model)), name="W_o")  # Final projection matrix 

    # Store all attention outputs
    multi_head_outputs = []
    
    for i in range(num_heads):
        # Compute Q, K, V for this head
        Q = tf.matmul(X, W_q[i])  # (seq_len, depth)
        K = tf.matmul(X, W_k[i])  # (seq_len, depth)
        V = tf.matmul(X, W_v[i])  # (seq_len, depth)

        # Compute scaled dot-product attention
        scores = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(tf.cast(depth, tf.float32))
        attn_weights = tf.nn.softmax(scores, axis=-1)  # (seq_len, seq_len)
        attention_output = tf.matmul(attn_weights, V)  # (seq_len, depth)

        multi_head_outputs.append(attention_output)  # Store output from this head
    
    # Concatenate all heads' outputs
    multi_head_output = tf.concat(multi_head_outputs, axis=-1)  # (seq_len, d_model)

    # Apply final linear transformation
    output = tf.matmul(multi_head_output, W_o)  # (seq_len, d_model)

    return output

# Define a simple example sentence embedding
word_embeddings = {
    "hello": [1.0, 0.5, 0.3, 0.2, 0.7, 0.6, 0.1, 0.9],
    "i": [0.4, 0.8, 0.2, 0.6, 0.9, 0.1, 0.3, 0.5],
    "am": [0.5, 0.9, 0.3, 0.7, 0.2, 0.4, 0.6, 0.8],
    "a": [0.3, 0.2, 0.7, 0.5, 0.9, 0.1, 0.8, 0.6],
    "big": [0.9, 0.1, 0.4, 0.8, 0.2, 0.5, 0.3, 0.7],
    "monkey": [0.6, 0.4, 0.9, 1.0, 0.3, 0.2, 0.8, 0.5],
}

# Convert sentence into a tensor
sentence = ["hello", "i", "am", "a", "big", "monkey"]
X = tf.convert_to_tensor([word_embeddings[word] for word in sentence], dtype=tf.float32)  # (6, 8)

# Apply multi-head attention
output = multi_head_attention(X, num_heads=2)


def find_closest_words(embedding, word_embeddings):
    closest_word = None
    min_distance = float("inf")
    
    for word, vec in word_embeddings.items():
        distance = np.linalg.norm(embedding - np.array(vec))  # Euclidean distance
        if distance < min_distance:
            min_distance = distance
            closest_word = word
    
    return closest_word

# Decode each output embedding
decoded_sentence = [find_closest_words(vec, word_embeddings) for vec in output.numpy()]

print("🔹 Decoded Sentence:\n", decoded_sentence)