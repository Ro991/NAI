import numpy as np


def predict(inputs, weights):
    threshold = 0.0
    total_activation = 0.0
    for input, weight in zip(inputs, weights):
        total_activation += input * weight
    return 1.0 if total_activation >= threshold else 0.0


def accuracy(matrix, weights):
    correct = 0.0
    preds = []
    for i in range(len(matrix)):
        pred = predict(matrix[i][:-1], weights)
        preds.append(pred)
        if pred == matrix[i][-1]:
            correct += 1.0
    print("Predictions:", preds)
    return correct / float(len(matrix))


def train(matrix, weights, epochs=10, l_rate=1.0, stop_early=True):
    for epoch in range(epochs):
        cur_acc = accuracy(matrix, weights)
        print("\nEpoch %d \nWeights: " % epoch, weights)
        print("Accuracy: ", cur_acc)

        if cur_acc == 1.0 and stop_early:
            break

        for i in range(len(matrix)):
            prediction = predict(matrix[i][:-1], weights)
            error = matrix[i][-1] - prediction
            for k in range(len(weights)):
                weights[k] = weights[k] + (l_rate * error * matrix[i][k])
    return weights


convertfunc = lambda x: 1 if x == 'Iris-virginica' or x == 'Iris-versicolor' else 0

test_arr = np.genfromtxt("iris_test.txt", delimiter='    	 ', converters={-1: convertfunc}, encoding='UTF-8')
training_arr = np.genfromtxt("iris_training.txt", delimiter="    	", converters={-1: convertfunc}, encoding='UTF-8')

weights = [1] * (len(training_arr[0]) - 1)

mins = []
maxs = []

for x in range(len(training_arr[0]) - 1):
    max_tr = [max(i) for i in zip(*training_arr)][x]
    min_tr = [min(i) for i in zip(*training_arr)][x]
    min_te = [min(i) for i in zip(*test_arr)][x]
    max_te = [max(i) for i in zip(*test_arr)][x]
    maxs.append(max_tr if max_tr >= max_te else max_te)
    mins.append(min_tr if min_tr <= min_te else min_te)

for row in training_arr:
    row[:-1] = np.subtract(row[:-1], mins)
    row[:-1] = np.divide(row[:-1], np.subtract(maxs, mins))
for row in test_arr:
    row[:-1] = np.subtract(row[:-1], mins)
    row[:-1] = np.divide(row[:-1], np.subtract(maxs, mins))

weights = train(training_arr, weights)

print("\nTest set")
print(accuracy(test_arr, weights))

while 1:
    val = input("Enter values to check: ")
    if val == "stop":
        break
    val = np.array(val.split(" ")).astype(dtype=np.float)
    val = np.divide(np.subtract(val, mins), np.subtract(maxs, mins))
    if predict(val, weights) == 0:
        print("Setosa")
    else:
        print("Not Setosa")

