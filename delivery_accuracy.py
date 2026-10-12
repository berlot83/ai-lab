
ON_TIME = "on_time"
LATE = "late"

actual_labels = [ON_TIME, LATE, ON_TIME]
predicted_labels = [ON_TIME, LATE, LATE]

correct_predictions = 0.0

for actual, predicted in zip(actual_labels, predicted_labels):
    if (actual == predicted):
        correct_predictions += 1
    
accuracy = correct_predictions / len(actual_labels)

print(f"Correct predictions: {correct_predictions}")
print(f"Accuracy: {accuracy:.2%}")