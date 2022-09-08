import numpy as np
from typing import List
import math
from datetime import datetime

class MLP:
    def __init__(self, npl: List[int]):
        self.d = list(npl)
        self.L = len(self.d) - 1
        self.W = []  # W[l][i][j]

        self.X = []
        self.deltas = []

        self.last_iter = 0
        self.last_learning_rate = 0.01

        for l in range(len(self.d)):
            self.W.append([])

            if l == 0:
                continue

            for i in range(self.d[l - 1] + 1):
                self.W[l].append([])

                for j in range(self.d[l] + 1):
                    self.W[l][i].append(0 if j == 0 else np.random.uniform(-1.0, 1.0))

        for l in range(len(self.d)):
            self.X.append([])
            self.deltas.append([])

            for j in range(self.d[l] + 1):
                self.X[l].append(1.0 if j == 0 else 0.0)
                self.deltas[l].append(0.0)

    def predict(self, sample_inputs: np.ndarray, is_classification: bool = True) -> np.ndarray:
        for j in range(self.d[0]):
            self.X[0][j + 1] = sample_inputs[j]

        for l in range(1, self.L + 1):
            for j in range(1, self.d[l] + 1):
                total = 0.0
                for i in range(self.d[l - 1] + 1):
                    total += self.W[l][i][j] * self.X[l - 1][i]

                if l < self.L or is_classification:
                    total = math.tanh(total)

                self.X[l][j] = total

        return self.X[self.L][1:]

    def train(self,
                  all_samples_inputs: np.ndarray,
                  all_samples_expected_outputs: np.ndarray,
                  learning_rate: float = 0.01,
                  is_classification: bool = True,
                  nb_iter: int = 10000):
        loss_per_iter = []
        self.last_learning_rate = learning_rate
        self.last_iter = nb_iter
        for _ in range(nb_iter):
            k = np.random.choice(range(len(all_samples_inputs)))
            sample_inputs = all_samples_inputs[k]
            sample_expected_output = all_samples_expected_outputs[k]

            _ = self.predict(sample_inputs, is_classification)

            for j in range(1, self.d[self.L] + 1):
                print(sample_expected_output, j)
                semi_gradient = self.X[self.L][j] - sample_expected_output[j - 1]
                if is_classification:
                    semi_gradient = semi_gradient * (1 - self.X[self.L][j] ** 2)
                self.deltas[self.L][j] = semi_gradient

            for l in reversed(range(1, self.L + 1)):
                for i in range(1, self.d[l - 1] + 1):
                    total = 0.0
                    for j in range(1, self.d[l] + 1):
                        total += self.W[l][i][j] * self.deltas[l][j]
                    total = (1 - self.X[l - 1][i] ** 2) * total
                    self.deltas[l - 1][i] = total

            for l in range(1, self.L + 1):
                for i in range(self.d[l - 1] + 1):
                    for j in range(self.d[l] + 1):
                        self.W[l][i][j] -= learning_rate * self.X[l - 1][i] * self.deltas[l][j]

    def save(self, path):
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
        layers_string = str(self.d)
        filename = dt_string + "_" + layers_string
        f = open(path + '/' + filename, "x")
        content = str(self.last_learning_rate) + ";" + str(self.last_iter) + ";"
        for l in range(len(self.d)):
            for i in range(self.d[l - 1] + 1):
                for j in range(self.d[l] + 1):
                    content += self.W[l][i][j]
        content = str(self.last_learning_rate) + ";" + str(self.last_iter) + ";" + str(self.W) + ";" + str(self.X)
        f.write(content)

    def load(self, path, filename):
        with open(path + "/" + filename) as f:
            data = f.read()
        data_array = data.split(";")
        d_string = filename.split("_")[-1][1:-1:]
        self.d = np.fromstring(d_string, dtype=int, sep=',')
        self.last_learning_rate = float(data_array[0])
        self.last_iter = int(data_array[1])
        W_string = data_array[2]
        X_string = data_array[3]
