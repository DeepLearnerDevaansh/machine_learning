import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hides unnecessary TensorFlow logs
import tensorflow as tf
def single_head_attention(X):
    seq_len,d_k = X.shape
    W_q = tf.Variable(tf.random.uniform((d_k, d_k)), name="W_q")
    W_k = tf.Variable(tf.random.uniform((d_k, d_k)), name="W_k")
    W_v = tf.Variable(tf.random.uniform((d_k, d_k)), name="W_v")
    
    Q = tf.matmul(X, W_q)  # (seq_len, d_k)
    K = tf.matmul(X, W_k)  # (seq_len, d_k)
    V = tf.matmul(X, W_v)  # (seq_len, d_k)
    
    scores = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(tf.cast(d_k, tf.float32))  # (seq_len, seq_len)
     # Apply softmax to get attention weights
    attn_weights = tf.nn.softmax(scores, axis=-1)  # (seq_len, seq_len)

    # Compute final attention output
    output = tf.matmul(attn_weights, V)  # (seq_len, d_k)
    
    return output, attn_weights
  
# import numpy as np
# X = np.random.rand(6,4)#6 words and 4 dimensional


# # Define attention head dimensions
# input_dim = X.shape[1]
# d_k = 4  # Key dimension

# W_q = np.random.rand(input_dim, d_k)
# W_k = np.random.rand(input_dim, d_k)
# W_v = np.random.rand(input_dim, d_k)


# # Compute Q, K, V matrices
# Q = X @ W_q #@is used in the place of np.dot
# K = X @ W_k
# V = X @ W_v
        
# attn_scores = (Q @ K.T) / np.sqrt(d_k)

# #apply softmax
# attn_weights = np.exp(attn_scores) / np.sum(np.exp(attn_scores), axis=1, keepdims=True)


# # Compute output
# output = attn_weights @ V     

# # Print results
# print("Attention Weights:\n", attn_weights)
# print("\nFinal Output:\n", output)
        

#you can make the function of that by just input X(embeddings) and output attn_weights and final output
#now lets try it with some the examples 

word_embeddings = {
    "hello": [1.0, 0.5, 0.3, 0.2],
    "i": [0.4, 0.8, 0.2, 0.6],
    "am": [0.5, 0.9, 0.3, 0.7],
    "a": [0.3, 0.2, 0.7, 0.5],
    "big": [0.9, 0.1, 0.4, 0.8],
    "monkey": [0.6, 0.4, 0.9, 1.0],
}
    
# sentence = ["hello", "i", "am", "a", "big", "monkey"]
# X = tf.convert_to_tensor([word_embeddings[word] for word in sentence], dtype=tf.float32)  # (6, 4)

# # Apply single-head attention
# output, attn_weights = single_head_attention(X)
# print("Single-head attention script executed successfully!")

#its just not calling my function properly so thats why i have to do it
if __name__ == "__main__":
    print("Running single-head attention model...")
    sentence = ["hello", "i", "am", "a", "big", "monkey"]
    X = tf.convert_to_tensor([word_embeddings[word] for word in sentence], dtype=tf.float32)
    output, attn_weights = single_head_attention(X)
    print("Output:\n", output.numpy())