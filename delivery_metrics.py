actual_times = [30, 40, 25]
predicted_times = [28, 43, 25]
predicted_times_large_error = [28, 43, 35]

total_absolute_error = 0.0
total_squared_error = 0.0

for actual, predicted in zip(actual_times, predicted_times_large_error):
    error = predicted - actual
    total_absolute_error += abs(error)
    total_squared_error += error ** 2

mae = total_absolute_error / len(actual_times)
mse = total_squared_error / len(actual_times)

print(f"MAE: {mae:.4f} minutes")
print(f"MSE: {mse:.4f} minutes squared")

