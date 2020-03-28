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

weights = train(training_arr, weights)

print("\nTest set")
print(accuracy(test_arr, weights))
