import numpy as np


# Z - input data, L - number of trees, N - subsampling size in window, E - current anomaly detector, u - predifined anomaly detector

class IForestASD:
    def __init__(self, window_size=256, tree_num=100,  u=0.5):
        self.window_size = window_size
        self.tree_num = tree_num
        self.u = u

        self.window = None
        self.previous_window = None

        self.all_pred = [] #все аномалии из датасета, чтобы  картинка была
        self.all_anomalies = [] #все аномалии из датасета, чтобы  картинка была

        self.curr_row_num = 0

    def first_fit(self, X: np.ndarray):
        total_row_num = X.shape[0]

        self.window = np.empty((0, X.shape[1]))
        if self.curr_row_num == 0:
            self.iforest_window = IForest(self.window_size, self.tree_num)
            for i in range(self.window_size):
                row = np.reshape(X[i], (1, len(X[i])))
                # не понятно как этот момент будет ирл работать на постоянном потоке, но думаю лего исправить
                self.window = np.append(self.window, row, axis=0)
                self.curr_row_num += 1
            self.iforest_window.fit(self.window)


        for i in range(self.curr_row_num, total_row_num):
            self.second_fit(X[i])


    def second_fit(self, X: np.ndarray):
        X = np.reshape(X, (1, len(X)))
        if self.curr_row_num % self.window_size == 0: #это на заполнение
            self.previous_window = self.window
            self.window = X

            window_score = self.iforest_window.predict(self.previous_window, self.u)
            score_count = np.sum(window_score) / len(window_score)

            if score_count > self.u:
                self.iforest_window.fit(self.previous_window)

            window_anomaly_scores = self.print_anomaly_scores(self.previous_window)
            window_predictions = self.print_predictions(self.previous_window)
            self.all_predicts(window_predictions)
            self.all_anomaly_scores(window_anomaly_scores)
            #print("Window Anomaly Scores:", window_anomaly_scores) #для каждого окна принтит аномали скоры
            print("Window Predictions:", window_predictions) #для каждого окна принтит предикты

        else:
            self.window = np.concatenate((self.window, X))

        self.curr_row_num += 1

    def print_anomaly_scores(self, previous_window):
        window_score = self.iforest_window.anomaly_score(previous_window)
        return window_score

    def print_predictions(self, previous_window):
        window_predictions = self.iforest_window.predict(previous_window, self.u)
        return window_predictions

    def all_predicts(self, window_predicts): #все аномалии из датасета, чтобы  картинка была
        self.all_pred.extend(window_predicts)
        return self.all_pred
    
    def all_anomaly_scores(self, window_anomalies): #все аномалии из датасета, чтобы  картинка была
        self.all_anomalies.extend(window_anomalies)
        return self.all_anomalies



#part 2 iforest model

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
    def __init__(self, sample_size, tree_num):   #psi = 256, t = 100 default
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





