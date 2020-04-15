import collections
import os
import re


def read_texts():
    file = open(filename, encoding='utf-8')
    file_lang = filename.split("\\")[1].split('.')[0]
    langs.append(file_lang)
    for line in file:
        letter_count = collections.Counter(re.findall(r'[qwertyuiopasdfghjklzxcvbnm]', line.lower()))
        letter_freq = sorted([(letter, letter_count[letter]) for letter in latin_letters], key=lambda t: t[0])
        arr = ([x[1] for x in letter_freq])
        arr.append(file_lang)
        data.append(arr)


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


latin_letters = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'z',
                 'x', 'c', 'v', 'b', 'n', 'm']
data = []
files = {}
langs = []
for (dirpath, dirnames, filenames) in os.walk('Languages'):
    for filename in filenames:
        files[filename] = os.sep.join([dirpath, filename])

for key in files:
    filename = files[key]
    read_texts()


