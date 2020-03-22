import math
import numpy as np
import pandas as pd


def count_most_freq_type(types):
    counter = {}
    for t in types['types']:
        if t in counter:
            counter[t] += 1
        else:
            counter[t] = 1
    return max(counter, key=counter.get)


def count_correct(predicted_vs_correct):
    correct = 0
    for index1, pred in predicted_vs_correct.iterrows():
        if pred['types'] == pred['guess']:
            correct += 1
    return correct


def predict(training, test):
    """
    :param training:
    :param test:
    :return a guess of which type:
    """
    for ind, r in training.iterrows():
        dist = 0
        for x in range(len(training.columns) - 2):
            dist += (r[x] - test[x])**2
        dist = math.sqrt(dist)
        r['distance'] = dist
        training.iloc[ind] = r

    return count_most_freq_type(training.sort_values('distance').head(k))


iris_data_test = pd.read_csv("iris_test.txt", header=None, delimiter="    	 ", engine='python')
iris_data_training = pd.read_csv("iris_training.txt", header=None, delimiter="    	", engine='python')
iris_data_training = iris_data_training.rename(columns={len(iris_data_training.columns)-1: "types"}, errors='raise')
iris_data_test = iris_data_test.rename(columns={len(iris_data_test.columns)-1: "types"}, errors='raise')

min_vals = []
max_vals = []

for c in range(len(iris_data_test.columns) - 1):
    iris_data_training[c].astype('float')
    iris_data_test[c].astype('float')
    min_val = iris_data_training[c].min() if iris_data_training[c].min() < iris_data_test[c].min() \
        else iris_data_test[c].min()
    max_val = iris_data_training[c].max() if iris_data_training[c].max() > iris_data_test[c].max() \
        else iris_data_test[c].max()
    min_vals.insert(c, min_val)
    max_vals.insert(c, max_val)
    iris_data_test[c] = (iris_data_test[c] - min_val) / max_val
    iris_data_training[c] = (iris_data_training[c] - min_val) / max_val

k = 5
k = int(input("Enter k: "))
print("Processing")
iris_data_training['distance'] = 0
iris_data_test['guess'] = ""
for index, row in iris_data_test.iterrows():
    row['guess'] = predict(iris_data_training, row)
    iris_data_test.iloc[index] = row
total_correct = count_correct(iris_data_test)
total_sample = len(iris_data_test.index)
print(total_correct, "/", total_sample, round((total_correct/total_sample)*100, 2), "%")

# allow user to input fresh data
print("Input ", len(iris_data_test.columns)-2,
      "parameters for classification in format '3.2,4.1,8.4,0.2' (type 'stop' to exit): ")

inp = ""
while 1:
    inp = input()
    if inp == "stop":
        exit(1)
    test_set = np.array(inp.split(",")).astype(np.float)
    index = 0
    for y in range(len(iris_data_test.columns)-2):
        test_set[y] = (test_set[y]-min_vals[y])/max_vals[y]
    print(predict(iris_data_training, test_set))

