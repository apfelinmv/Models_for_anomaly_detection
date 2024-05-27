import numpy as np
import matplotlib.pyplot as plt
import time

def c(n):
    if n > 2:
        return 2 * (np.log(n - 1) + 0.5772156649) - (2 * (n - 1) / n)
    elif n == 2:
        return 1
    else:
        return 0


class ExNode:
    def __init__(self, size):
        self.size = size


class InNode:
    def __init__(self, left, right, splitAtt, splitVal):
        self.left = left
        self.right = right
        self.splitAtt = splitAtt
        self.splitVal = splitVal


class ITree:
    def __init__(self, lim_height):
        self.lim_height = lim_height
        self.root = None

    def fit(self, X: np.ndarray, curr_height=0):
        if curr_height >= self.lim_height or len(X) <= 1:
            self.root = ExNode(len(X))
            return self.root

        Q = X.shape[1]
        q = np.random.choice(Q)
        p = np.random.uniform(X[:, q].min(), X[:, q].max())

        x_l = X[X[:, q] < p]
        x_r = X[X[:, q] >= p]

        left = ITree(self.lim_height)
        right = ITree(self.lim_height)
        left.fit(x_l, curr_height + 1)
        right.fit(x_r, curr_height + 1)

        self.root = InNode(left.root, right.root, splitAtt=q, splitVal=p)
        return self.root


class IForest:
    def __init__(self, sample_size=256, tree_num=100):   #psi = 256, t = 100
        self.sample_size = sample_size
        self.tree_num = tree_num
        self.forest = []
        self.height_limit = np.ceil(np.log2(sample_size))

    def fit(self, X: np.ndarray):
        for i in range(self.tree_num):
            x_prime_indices = np.random.choice(X.shape[0], self.sample_size)
            x_prime = X[x_prime_indices]
            tree = ITree(self.height_limit)
            tree.fit(x_prime, 0)
            self.forest.append(tree)
        return self

    def path_length(self, X: np.ndarray):
        allPL = []
        for x in X:
            pathLength = []
            for tree in self.forest:
                pathLength.append(PathLength(x, tree.root, 0))
            allPL.append(pathLength)
        paths = np.array(allPL)
        return np.mean(paths, axis=1)


    def anomaly_score(self, X: np.ndarray):
        avg_path_lengths = self.path_length(X)
        scores = []
        for h in avg_path_lengths:
            score = 2 ** (-h / c(self.sample_size))
            scores.append(score)
        scores = np.array(scores)
        return scores

    def predict(self, X: np.ndarray, threshold: float):
        anomaly_scores = self.anomaly_score(X)
        return np.where(anomaly_scores >= threshold, 1, 0)


def PathLength(x, T, e):

    if isinstance(T, ExNode):
        return e + c(T.size)

    a = T.splitAtt
    if x[a] < T.splitVal:
        return PathLength(x, T.left, e + 1)
    else:
        return PathLength(x, T.right, e + 1)
