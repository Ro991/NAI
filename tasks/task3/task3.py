import collections
import copy
import os
import re
import tkinter as tk
from tkinter import simpledialog


class Perceptron:
    weights = []
    inputs = 0
    language = ""

    def __init__(self, inputs, language):
        self.inputs = inputs
        self.weights = [0] * inputs
        self.language = language

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
            pred = self.predict(matrix[i][:-1])
            preds.append(pred)
            if pred == matrix[i][-1]:
                correct += 1.0
        print("Predictions:", preds)
        return correct / float(len(matrix))

    def train(self, matrix, epochs=10, l_rate=1.0, stop_early=True):
        for epoch in range(epochs):
            cur_acc = self.accuracy(matrix)
            print("\nEpoch %d \nWeights: " % epoch, self.weights)
            print("Accuracy: ", cur_acc)

            if cur_acc == 1.0 and stop_early:
                break

            for i in range(len(matrix)):
                prediction = self.predict(matrix[i][:-1])
                error = matrix[i][-1] - prediction
                for k in range(len(self.weights)):
                    self.weights[k] = self.weights[k] + (l_rate * error * matrix[i][k])

    def __str__(self):
        return self.language + " " + self.weights.__str__()


def read_texts():
    file = open(filename, encoding='utf-8')
    file_lang = filename.split("\\")[1].split('.')[0]
    langs.append(file_lang)
    langdata = []
    for line in file:
        letter_count = collections.Counter(re.findall(r'[qwertyuiopasdfghjklzxcvbnm]', line.lower()))
        letter_freq = sorted([(letter, letter_count[letter]) for letter in latin_letters], key=lambda t: t[0])
        arr = ([x[1] for x in letter_freq])
        arr.append(file_lang)
        langdata.append(arr)
    data.append(langdata)


def split_data(split):
    """split %"""
    for langdata in data:
        train_size = int((split/100) * len(langdata))
        for x in range(len(langdata)):
            if x < train_size:
                training_data.append(langdata[x])
            else:
                test_data.append(langdata[x])


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
    perceptrons.append(Perceptron(len(latin_letters), langs[x]))

training_data = []
test_data = []

mins = []
maxs = []

for j in range(len(langs)):
    for x in range(len(data[j][0]) - 1):
        max_num = [max(i) for i in zip(*data[j])][x]
        min_num = [min(i) for i in zip(*data[j])][x]
        if j == 0:
            mins.append(min_num)
            maxs.append(max_num)
        else:
            if min_num < mins[x]:
                mins[x] = min_num
            if max_num > maxs[x]:
                maxs[x] = max_num

for lang in data:
    for row in lang:
        for x in range(len(row)-1):
            row[x] -= mins[x]
            row[x] /= (maxs[x]-mins[x])
split_data(60)

index = 0
for p in perceptrons:
    datacopy = copy.deepcopy(training_data)
    for idx, d in enumerate(datacopy):
        if d[-1] == langs[index]:
            datacopy[idx][-1] = 1
        else:
            datacopy[idx][-1] = 0
    print("lang:", langs[index])
    p.train(datacopy, epochs=30)
    index += 1

for x in range(len(langs)):
    print("final weights: ", perceptrons[x])

correct = 0
for p in perceptrons:
    for d in test_data:
        if d[-1] == p.language:
            correct += 1
print("correct in test set:", correct/len(test_data)*100, "%", "\n")

ROOT = tk.Tk()
ROOT.withdraw()
USER_INP = simpledialog.askstring(title="Lang classifier",
                                  prompt="Enter text in a language")
letter_count = collections.Counter(re.findall(r'[qwertyuiopasdfghjklzxcvbnm]', USER_INP.lower()))
letter_freq = sorted([(letter, letter_count[letter]) for letter in latin_letters], key=lambda t: t[0])
arr = ([x[1] for x in letter_freq])
for p in perceptrons:
    if p.predict(arr) == 1:
        print(p.language, "")
