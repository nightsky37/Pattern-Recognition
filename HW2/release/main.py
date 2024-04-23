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
        # print("cost: ", cost)
        gradient = np.dot(x.T, (y_pred - y)) / m
        return cost[0], gradient

    def fit(self, 
        inputs: npt.NDArray[np.float_], 
        targets: t.Sequence[int],
    ) -> None:
        rows = inputs.shape[0]
        targets = targets.reshape(rows, 1)
        # print("targets:", targets.shape)
        costs = []
        inputs = np.append(inputs, np.ones([len(inputs), 1]), 1)
        print("inputs:", inputs.shape)
        self.weights = np.zeros(inputs.shape[1])
        self.weights = self.weights.reshape(self.weights.shape[0], 1)
        print("weights:", self.weights.shape)

        for i in range(self.num_iterations):
            cost, gradient = self.compute_cost(inputs, targets)
            self.weights -= self.learning_rate * gradient
            costs.append(cost)

        self.intercept = self.weights[-1]
        self.weights = self.weights[:-1]

    def predict(
        self,
        inputs: npt.NDArray[np.float_],
    ) -> t.Tuple[t.Sequence[np.float_], t.Sequence[int]]:      
        """
        Implement your prediction function here.
        The return should contains
        1. sample probabilty of being class_1
        2. sample predicted class
        """
        inputs = np.append(inputs, np.ones([len(inputs), 1]), 1)
        self.weights = np.append(self.weights, self.intercept)
        pred = self.sigmoid(np.dot(inputs, self.weights))
        pred_classes = np.zeros(len(pred))
        pred_classes[np.where(pred>0.5)] = 1
        # print("pred: ", pred)
        # print(">=0.5: ", np.where(pred>=0.5))
        # print("pred_classes: ", pred_classes)

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
        self.thres = None

    def fit(
        self,
        inputs: npt.NDArray[np.float_],
        targets: t.Sequence[int],
    ) -> None:
        X0 = inputs[targets == 0]
        X1 = inputs[targets == 1]
        print("X0:", X0.shape)
        print("X1:", X1.shape)
        self.m0 = np.mean(X0, axis=0)
        self.m0 = self.m0
        self.m1 = np.mean(X1, axis=0)
        self.m1 = self.m1
        print("m0:", self.m0.shape)
        print("m1-m0:", (self.m1-self.m0).shape)
        # s0 = np.dot((X0-self.m0).T, (X0-self.m0))
        # s1 = np.dot((X1-self.m1).T, (X1-self.m1))
# －0.6275043821234734088800495144518
# －0.6275044049684719859494350420153
        s0 = np.cov(X0, rowvar=False)
        s1 = np.cov(X1, rowvar=False)
        print("s0:", s0.shape)
        print("s1:", s1.shape)
        m = np.mean(inputs, axis=0)
        print("m:", m.shape)
        self.sw = s0+s1
        self.sb = np.dot((self.m1-self.m0).reshape(-1, 1), (self.m1-self.m0).reshape(-1, 1).T) # 直接平方?
        # self.sb = (self.m1-self.m0).reshape(-1, 1)**2
        invSW_by_SB = np.linalg.inv(self.sw) @ self.sb
        print("invSWSB:", invSW_by_SB.shape)
        # self.w = invSW_by_SB.reshape(-1,1)
        
        eigenvalues, eigenvectors = np.linalg.eig(invSW_by_SB)
        print("eigvec:", eigenvectors)
        print("eigval:", eigenvalues)
        self.w = eigenvectors[:, 1].reshape(-1, 1)*-1
        print("w:", self.w.shape)
        # self.slope = eigenvalues

        y0 = np.dot(X0, self.w)
        y1 = np.dot(X1, self.w)
        y0m = np.mean(y0)
        y1m = np.mean(y1)
        self.thres = (y0m+y1m)/2

    def predict(
        self,
        inputs: npt.NDArray[np.float_],
    ) -> t.Sequence[t.Union[int, bool]]:
        y_p = np.dot(inputs, self.w)
        m = float(np.mean(y_p))
        print("y_p:", y_p.shape)
        print("y_pm:", self.thres)
        y_pred = np.zeros(len(y_p))
        print("y_pred:", y_pred.shape)
        y_pred = y_pred.reshape((len(y_pred), 1))
        y_pred[y_p>=self.thres] = 1
        y_pred = y_pred.flatten()

        return y_pred

    def plot_projection(self, inputs: npt.NDArray[np.float_]):
        y_pred = self.predict(inputs)
        y0 = np.zeros(len(y_pred))
        y1 = np.zeros(len(y_pred))
        y0[np.where(y_pred<0)] = 0

def compute_auc(y_trues, y_preds) -> float:
    return metrics.roc_auc_score(y_trues, y_preds)


def accuracy_score(y_trues, y_preds) -> float:
    y_len = len(y_trues)
    print("y_preds: ", y_preds)
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

    LR = LogisticRegression( #13000
        learning_rate = 0.0005,  # You can modify the parameters as you want
        num_iterations = 1000,  # You can modify the parameters as you want
    )
    LR.fit(x_train, y_train)
    y_pred_probs, y_pred_classes = LR.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_classes)
    auc_score = compute_auc(y_test, y_pred_probs)
    logger.info(f'LR: Weights: {LR.weights[:5]}, Intercep: {LR.intercept}')
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
