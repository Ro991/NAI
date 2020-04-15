import collections
import os
import re


class Perceptron:
    weights = []
    inputs = 0

    def __init__(self, inputs):
        self.inputs = inputs
        self.weights = [1] * inputs

    def predict(self, inputs):
        threshold = 0.0
        total_activation = 0.0
        for input, weight in zip(inputs, self.weights):
            total_activation += input * weight
        return 1.0 if total_activation >= threshold else 0.0

    def accuracy(self, matrix):
        correct = 0.0
        preds = []
        for i in range(len(matrix)):
            pred = self.predict(matrix[i][:-1], self.weights)
            preds.append(pred)
            if pred == matrix[i][-1]:
                correct += 1.0
        print("Predictions:", preds)
        return correct / float(len(matrix))

    def train(self, matrix, epochs=10, l_rate=1.0, stop_early=True):
        for epoch in range(epochs):
            cur_acc = self.accuracy(matrix, self.weights)
            print("\nEpoch %d \nWeights: " % epoch, self.weights)
            print("Accuracy: ", cur_acc)

            if cur_acc == 1.0 and stop_early:
                break

            for i in range(len(matrix)):
                prediction = self.predict(matrix[i][:-1], self.weights)
                error = matrix[i][-1] - prediction
                for k in range(len(self.weights)):
                    self.weights[k] = self.weights[k] + (l_rate * error * matrix[i][k])

    def __str__(self):
        return self.weights.__str__()


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

perceptrons = []
for x in range(len(langs)):
    perceptrons.append(Perceptron(len(latin_letters)))

    print(perceptrons[x])

