import numpy as np
import pandas as pd

class NaiveBayes:

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # calculate mean, var, and prior for each class
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []

        # calculate posterior probability for each class
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + posterior
            posteriors.append(posterior)

        # return class with highest posterior probability
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator


def read_in_data(f):
    array_lines = []
    for line in f:
        line = line.rstrip('\n').strip()
        columns = line.split('\t')
        for x in range(len(columns)):
            if x != len(columns) - 1:
                columns[x] = float(columns[x])
        array_lines.append(columns)
    return array_lines


def compute_confusion_matrix(true, pred):
    K = len(np.unique(true))  # Number of classes
    result = np.zeros((K, K))

    for i in range(len(true)):
        result[true[i]][pred[i]] += 1

    return result


def map(labels):
    return dict(zip(np.unique(labels), range(len(np.unique(labels)))))


def accuracy(y_true, y_pred):
    test = np.sum(y_true == y_pred)
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


training_arr = read_in_data(open('iris_training.txt'))
training_arr_X = [item[:-1] for item in training_arr]
training_arr_Y = [item[-1] for item in training_arr]
test_arr = read_in_data(open("iris_test.txt"))
test_arr_X = [item[:-1] for item in test_arr]
test_arr_Y = [item[-1].strip() for item in test_arr]
# print(training_arr_Y)


bay = NaiveBayes()
bay.fit(np.array(training_arr_X), np.array(training_arr_Y))
pred = bay.predict(np.array(test_arr_X))

y_actual = pd.Series(test_arr_Y, name='Actual')
y_predicted = pd.Series(pred, name='Predicted')
pd_confusion = pd.crosstab(y_actual, y_predicted)

# recall, precision, accuracy ,F1 score

y_actul_mapped = y_actual.replace(map(y_actual))
y_predicted_mapped = y_predicted.replace(map(y_predicted))

conf_matrix = compute_confusion_matrix(y_actul_mapped, y_predicted_mapped)
ret_mat = np.zeros((3, 3))

print("accuracy = ", accuracy(y_actual, y_predicted) * 100, "%")


for _ in range(len(conf_matrix)):
        ret_mat[_][0] = conf_matrix[_][_]/ (sum(conf_matrix[_]) )
        ret_mat[_][1] = conf_matrix[_][_]/ (np.sum(conf_matrix, axis=1)[_])
        ret_mat[_][2] = 2 * ((ret_mat[_][0] * ret_mat[_][1]) / (ret_mat[_][0] + ret_mat[_][1]))

pandas_output = pd.DataFrame(data=ret_mat, columns=["Precision", "Recall", "F1"], index=np.unique(y_actual))

print(pandas_output*100)

inp = ""
while inp != "stop":
    print("Enter data or 'stop'")
    inp = input()
    if inp != "stop":
        input_set = np.fromstring(inp, dtype=float, sep=' ')
        print(bay.predict([input_set]))
