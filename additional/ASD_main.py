import numpy as np
import time

# Z - input data, L - number of trees, N - subsampling size in window, E - current anomaly detector, u - predifined anomaly detector

class IForestASD:
    def __init__(self, window_size=256, tree_num=100, u=0.5, sample_size=256):
        self.window_size = window_size
        self.sample_size = sample_size
        self.tree_num = tree_num
        self.u = u

        self.window = None
        self.previous_window = None

        self.all_pred = []  # все аномалии из датасета, чтобы  картинка была
        self.all_anomalies = []  # все аномалии из датасета, чтобы  картинка была

        self.curr_row_num = 0

    def train(self, X_train: np.ndarray): # этого не написано в бумажке
        self.iforest_window = IForest(self.window_size, self.tree_num)
        self.iforest_window.fit(X_train)

    def first_fit(self, X: np.ndarray):
        total_row_num = X.shape[0]

        self.window = np.empty((0, X.shape[1]))
        if self.curr_row_num == 0:
            for i in range(self.window_size):
                row = np.reshape(X[i], (1, len(X[i])))
                self.window = np.append(self.window, row, axis=0)
                self.curr_row_num += 1


        for i in range(self.curr_row_num, total_row_num):
            self.second_fit(X[i])

    def second_fit(self, X: np.ndarray):
        X = np.reshape(X, (1, len(X)))
        if self.curr_row_num % self.window_size == 0:  # это на заполнение
            self.previous_window = self.window
            self.window = X

            window_score = self.iforest_window.predict(self.previous_window, self.u)
            score_count = np.sum(window_score) / len(window_score)

            if score_count > self.u:
                self.iforest_window.fit(self.previous_window)
                print("model changed")

            window_anomaly_scores = self.print_anomaly_scores(self.previous_window)
            window_predictions = self.print_predictions(self.previous_window)
            self.all_predicts(window_predictions)
            self.all_anomaly_scores(window_anomaly_scores)
            # print("Window Anomaly Scores:", window_anomaly_scores) #для каждого окна принтит аномали скоры
            print("Window Predictions:", window_predictions)  # для каждого окна принтит предикты

        else:
            self.window = np.concatenate((self.window, X))

        self.curr_row_num += 1

    def print_anomaly_scores(self, previous_window):
        window_score = self.iforest_window.anomaly_score(previous_window)
        return window_score

    def print_predictions(self, previous_window):
        window_predictions = self.iforest_window.predict(previous_window, self.u)
        return window_predictions

    def all_predicts(self, window_predicts):  # все аномалии из датасета, чтобы  картинка была
        self.all_pred.extend(window_predicts)
        return self.all_pred

    def all_anomaly_scores(self, window_anomalies):  # все аномалии из датасета, чтобы  картинка была
        self.all_anomalies.extend(window_anomalies)
        return self.all_anomalies


# part 2 iforest model

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
    def __init__(self, sample_size, tree_num):  # psi = 256, t = 100 default
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


import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
import pandas as pd
# Import your classes here
import matplotlib.pyplot as plt




def plot_roc_curve(y_true, y_pred, title, processing_time):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='best')
    if processing_time is not None:
        plt.text(0.6, 0.2, f'Processing Time: {processing_time:.2f} sec', fontsize=12)
    plt.show()



def main():
    window_size = 200
    tree_num = 100
    u = 0.5
    model = IForestASD(window_size, tree_num, u, 64)



   
    X = np.genfromtxt('smtp.csv', delimiter=',')
    true_labels = pd.read_csv('smtp_labels.csv', header=None).to_numpy().flatten()

    X_train = X[:900]
    X_test = X[900:]

    #нужно сделать трейн функ
    y_test = true_labels[900:9700]
    
    start_time = time.time()
    model.train(X_train)
    end_time = time.time()
    print(f"Program train time: {end_time - start_time:.2f} seconds")

    start_time = time.time()
    model.first_fit(X_test)
    end_time = time.time()
    print(f"Program test time: {end_time - start_time:.2f} seconds")

   
    all_predictions = model.all_pred
    all_anomaly_scores = model.all_anomalies


    f1score = f1_score(y_test, all_predictions)


    auc_score = roc_auc_score(y_test, all_predictions)
    print("F1 Score:", f1score)
    print("AUC Score:", auc_score)

    num_predicted_anomalies = np.sum(all_predictions)

    num_true_anomalies = np.sum(y_test)

    total_time = end_time - start_time
    print("Number of anomalies in predicted dataset:", num_predicted_anomalies)
    print("Number of anomalies in true labels:", num_true_anomalies)
    dict = {'name': all_predictions}
    df = pd.DataFrame(dict)
    df.to_csv('all_preds')
    plot_roc_curve(y_test, all_predictions, 'ROC Curve', total_time)

if __name__ == "__main__":
    main()



