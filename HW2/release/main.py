import typing as t

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
from sklearn import metrics
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, learning_rate: float = 1e-4, num_iterations: int = 100):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.intercept = None

    def compute_cost(self, x, y):
        m = len(y)
        y_pred = self.sigmoid(np.dot(x, self.weights))
        error = (y * np.log(y_pred)) + ((1 - y) * np.log(1 - y_pred))
        cost = -1 / m * sum(error)
        gradient = np.dot(x.T, (y_pred - y)) / m
        return cost[0], gradient

    def fit(self,
            inputs: npt.NDArray[np.float_],
            targets: t.Sequence[int],
            ) -> None:
        rows = inputs.shape[0]
        targets = targets.reshape(rows, 1)
        costs = []
        inputs = np.append(inputs, np.ones([len(inputs), 1]), 1)
        self.weights = np.zeros(inputs.shape[1])
        self.weights = self.weights.reshape(self.weights.shape[0], 1)

        for i in range(self.num_iterations):
            cost, gradient = self.compute_cost(inputs, targets)
            self.weights -= self.learning_rate * gradient
            costs.append(cost)

        self.intercept = self.weights[len(self.weights) - 1]
        self.intercept = float(self.intercept[0])
        self.weights = self.weights[:-1]

    def predict(
        self,
        inputs: npt.NDArray[np.float_],
    ) -> t.Tuple[t.Sequence[np.float_], t.Sequence[int]]:
        inputs = np.append(inputs, np.ones([len(inputs), 1]), 1)
        self.weights = np.append(self.weights, self.intercept)
        pred = self.sigmoid(np.dot(inputs, self.weights))
        pred_classes = np.zeros(len(pred))
        pred_classes[np.where(pred > 0.5)] = 1

        return pred, pred_classes

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


class FLD:
    def __init__(self):
        self.w = None
        self.m0 = None
        self.m1 = None
        self.sw = None
        self.sb = None
        self.slope = None
        self.thresh = None
        self.intercept = None

    def fit(
        self,
        inputs: npt.NDArray[np.float_],
        targets: t.Sequence[int],
    ) -> None:
        X0 = inputs[targets == 0]
        X1 = inputs[targets == 1]
        self.m0 = np.mean(X0, axis=0)
        self.m1 = np.mean(X1, axis=0)
        s0 = np.cov(X0, rowvar=False)
        s1 = np.cov(X1, rowvar=False)
        self.sw = s0 + s1
        self.sb = np.dot((self.m0 - self.m1).reshape(-1, 1), (self.m0 - self.m1).reshape(-1, 1).T)
        invSW_by_SB = np.dot(np.linalg.inv(self.sw), self.sb)

        eigenvalues, eigenvectors = np.linalg.eig(invSW_by_SB)
        self.w = eigenvectors[:, np.argmax(eigenvalues)].reshape(-1, 1) * -1
        y0 = np.dot(X0, self.w)
        y1 = np.dot(X1, self.w)
        y0m = np.mean(y0)
        y1m = np.mean(y1)
        self.thresh = (y0m + y1m) / 2
        self.slope = self.w[1][0] / self.w[0][0]
        self.slope = float(self.slope)
        # print("slope:", self.slope)

    def predict(
        self,
        inputs: npt.NDArray[np.float_],
    ) -> t.Sequence[t.Union[int, bool]]:
        y_p = np.dot(inputs, self.w)
        y_pred = np.zeros(len(y_p))
        y_pred = y_pred.reshape((len(y_pred), 1))
        y_pred[y_p >= self.thresh] = 1
        y_pred = y_pred.flatten()

        return y_pred

    def plot_projection(self, inputs: npt.NDArray[np.float_]):
        y_pred = self.predict(inputs)
        x0 = inputs[y_pred == 0]
        x1 = inputs[y_pred == 1]
        plt.scatter(x0[:, 0], x0[:, 1], color='red', s=10)
        plt.scatter(x1[:, 0], x1[:, 1], color='blue', s=10)

        for i in range(x0.shape[0]):
            project_point = np.dot(x0[i], self.w) * self.w  # pp = project and multiply dir vector
            plt.scatter(project_point[0][0], project_point[1][0], color='red', s=15)
            plt.plot([project_point[0][0], x0[i][0]], [project_point[1][0], x0[i][1]], color='gray', linewidth=1)

        for i in range(x1.shape[0]):
            project_point = np.dot(x1[i], self.w) * self.w  # pp = project and multiply dir vector
            plt.scatter(project_point[0][0], project_point[1][0], color='blue', s=15)
            plt.plot([project_point[0][0], x1[i][0]], [project_point[1][0], x1[i][1]], color='grey', linewidth=1)

        x_lin = np.linspace(min(min(inputs[:, 0]), min(inputs[:, 1])), max(max(inputs[:, 0]), max(inputs[:, 1])), 100)
        projection_line = self.slope * x_lin
        plt.plot(x_lin, projection_line)
        plt.title("Projection Line: w=%.6f, b=0" % self.slope)

        plt.axis('square')
        plt.show()


def compute_auc(y_trues, y_preds) -> float:
    return metrics.roc_auc_score(y_trues, y_preds)


def accuracy_score(y_trues, y_preds) -> float:
    y_len = len(y_trues)
    sumTP = np.sum(y_trues == y_preds.astype(int))
    return sumTP / y_len


def main():
    # Read data
    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    # Part1: Logistic Regression
    x_train = train_df.drop(['target'], axis=1).to_numpy()  # (n_samples, n_features)
    y_train = train_df['target'].to_numpy()  # (n_samples, )
    print(y_train.shape)

    x_test = test_df.drop(['target'], axis=1).to_numpy()
    y_test = test_df['target'].to_numpy()

    LR = LogisticRegression(
        learning_rate=0.0005,  # You can modify the parameters as you want
        num_iterations=13000,  # You can modify the parameters as you want
    )
    LR.fit(x_train, y_train)
    y_pred_probs, y_pred_classes = LR.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_classes)
    auc_score = compute_auc(y_test, y_pred_probs)
    logger.info(f'LR: Weights: {LR.weights[:5]}, Intercept: {LR.intercept}')
    logger.info(f'LR: Accuracy={accuracy:.4f}, AUC={auc_score:.4f}')

    # Part2: FLD
    cols = ['27', '30']  # Dont modify
    x_train = train_df[cols].to_numpy()
    y_train = train_df['target'].to_numpy()
    x_test = test_df[cols].to_numpy()
    y_test = test_df['target'].to_numpy()

    FLD_ = FLD()
    FLD_.fit(x_train, y_train)
    y_preds = FLD_.predict(x_test)
    accuracy = accuracy_score(y_test, y_preds)
    logger.info(f'FLD: m0={FLD_.m0}, m1={FLD_.m1}')
    logger.info(f'FLD: \nSw=\n{FLD_.sw}')
    logger.info(f'FLD: \nSb=\n{FLD_.sb}')
    logger.info(f'FLD: \nw=\n{FLD_.w}')
    logger.info(f'FLD: Accuracy={accuracy:.4f}')
    FLD_.plot_projection(x_test)


if __name__ == '__main__':
    main()
