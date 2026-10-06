training_data = [(0.0, 1.0), (1.0, 3.0), (3.0, 7.0)]
validation_data = [(0.5, 2.0), (1.5, 4.0), (2.5, 6.0)]

print("Traingin data:", training_data)
print("Validation data: ", validation_data)

weight = 0.0
bias = 0.0

def predict(x, weight, bias):
    return weight * x + bias

print("Predictions before training")
for x, expected in training_data:
    prediction = predict(x, weight, bias)
    print(f"x={x}, expected={expected}, prediction={prediction}")

def calculate_mse(data, weight, bias):
    total_error = 0.0
    for x, expected in data:
        prediction = predict(x, weight, bias)
        total_error += (prediction - expected) ** 2
    return total_error / len(data)

training_error = calculate_mse(training_data, weight, bias)
validation_error = calculate_mse(validation_data, weight, bias)

print(f"Training MSE:{training_error:.4f}")
print(f"Validation MSE:{validation_error:.4f}")

learning_rate = 0.01
epochs = 100

for epoch in range(epochs):
    weight_gradient = 0.0
    bias_gradient = 0.0

    for x, expected in training_data:
        prediction = predict(x, weight, bias)
        error = prediction - expected
        weight_gradient += 2 * error * x
        bias_gradient += 2 * error

    weight_gradient /= len(training_data)
    bias_gradient /= len(training_data)

    weight -= learning_rate * weight_gradient
    bias -= learning_rate * bias_gradient

    if (epoch + 1) % 10 == 0: 
        training_error = calculate_mse(training_data, weight, bias)
        validation_error = calculate_mse(validation_data, weight, bias)
        print(f"Epoch {epoch + 1}: train={training_error:.4f}, validation={validation_error:.4f}")

print(f"Final weight: {weight:.4f}")
print(f"Final bias: {bias:.4f}")
    