import math

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Inputs (Free, Zoom, Offer)
x1, x2, x3 = 1, 0, 1

# Weights for Hidden Layer
# Neuron H1
w1_h1, w2_h1, w3_h1 = 0.5, -0.2, 0.3

# Neuron H2
w1_h2, w2_h2, w3_h2 = 0.4, 0.1, -0.5

# Hidden layer calculations
h1 = x1*w1_h1 + x2*w2_h1 + x3*w3_h1
h2 = x1*w1_h2 + x2*w2_h2 + x3*w3_h2

print("H1 value:", h1)   # 0.8
print("H2 value:", h2)   # -0.1

# Weights for Output layer
w_h1_out = 0.7
w_h2_out = 0.2

# Output (Spam score before activation)
spam_input = h1*w_h1_out + h2*w_h2_out
print("Spam input:", spam_input)  # 0.54 (≈ 0.56 in notes)

# Final spam probability
spam_probability = sigmoid(spam_input)
print("Spam Probability:", round(spam_probability, 3))