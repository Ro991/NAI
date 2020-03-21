import math

import pandas as pd


def count_most_freq_type(types):
    counter = {}
    for t in types['types']:
        if t in counter:
            counter[t] += 1
        else:
            counter[t] = 1
    return max(counter, key=counter.get)


iris_data_test = pd.read_csv("iris_test.txt", header=None, delimiter="    	 ", engine='python')
iris_data_training = pd.read_csv("iris_training.txt", header=None, delimiter="    	", engine='python')
iris_data_training = iris_data_training.rename(columns={len(iris_data_training.columns)-1: "types"}, errors='raise')
iris_data_test = iris_data_test.rename(columns={len(iris_data_test.columns)-1: "types"}, errors='raise')

for c in range(len(iris_data_test.columns) - 1):
    iris_data_training[c].astype('float')
    iris_data_test[c].astype('float')
    min_val = iris_data_training[c].min() if iris_data_training[c].min() < iris_data_test[c].min() \
        else iris_data_test[c].min()
    max_val = iris_data_training[c].max() if iris_data_training[c].max() > iris_data_test[c].max() \
        else iris_data_test[c].max()
    iris_data_test[c] = (iris_data_test[c] - min_val) / max_val
    iris_data_training[c] = (iris_data_training[c] - min_val) / max_val

k = 5
k = int(input("Enter k: "))
print("Processing")
iris_data_training['distance'] = 0
iris_data_test['guess'] = ""
for index, row in iris_data_test.iterrows():
    for ind, r in iris_data_training.iterrows():
        dist = 0
        for x in range(len(iris_data_test.columns) - 2):
            dist += (r[x] - row[x])**2
        dist = math.sqrt(dist)
        r['distance'] = dist
        iris_data_training.iloc[ind] = r
    row['guess'] = count_most_freq_type(iris_data_training.sort_values('distance').head(k))
    iris_data_test.iloc[index] = row
print(iris_data_test)
