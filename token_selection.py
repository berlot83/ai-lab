import math

def calculate_probabilities(logits: list[float]) -> list[float]: 
    exp_values: list[float] =[]
    for logit in logits:
        exp = math.exp(logit)
        exp_values.append(exp)
        print(f"Exp: {exp} | logit: {logit}")

    total_exp_values = sum(exp_values)
    print(f"total_exp_values: {total_exp_values}")

    probabilities: list[float] = []
    for exp in exp_values:
        probability = exp / total_exp_values
        probabilities.append(probability)
        print(f"Probability: {probability}, exp: {exp}")
    total_probabilities = sum(probabilities)
    print(f"total_probabilities: {total_probabilities}")
    return probabilities

token_logits: dict[str, float] = {
    "azul": 2.0,
    "verde": -1.0,
    ".": 0.0,
    "gris": 3.0
}

logits: list[float] = list(token_logits.values())
tokens: list[str] = list(token_logits.keys())
probabilities: list[float] = calculate_probabilities(logits)

# Selected Greedy
best_index = 0
best_probability = probabilities[0]
for index, probability in enumerate(probabilities):
    token = tokens[index]
    if (best_probability < probability):
        best_probability = probability
        best_index = index
    print(f"token: {token}, index: {index}, probability: {probability}")
print(f"Best probability: {best_probability}, best index: {best_index}")
print(tokens[best_index])

