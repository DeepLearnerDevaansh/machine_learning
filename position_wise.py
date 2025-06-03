import tensorflow as tf

class PositionWiseFFN(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFFN, self).__init__()
        self.fc1 = tf.keras.layers.Dense(d_ff, activation='relu')  # Expands dimensionality
        self.fc2 = tf.keras.layers.Dense(d_model)  # Projects back to original size

    def call(self, x):
        return self.fc2(self.fc1(x))  # FFN operation

# Example usage
d_model = 8  # Example embedding size
d_ff = 16    # Expanded hidden size

ffn = PositionWiseFFN(d_model, d_ff)

# Example input (6 tokens, each with d_model=8 dimensions)
X = tf.random.uniform((6, d_model))

output = ffn(X)
print("🔹 Position-wise FFN Output:\n", output.numpy())
