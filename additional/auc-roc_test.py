#it works only with iForest model, from iforest.py file from other directory

import pandas as pd

def load_kdd99_http_data():
    X = np.genfromtxt('http_4.csv', delimiter=',')
    print(len(X))


    y = pd.read_csv('http_column.csv', header=None).to_numpy().flatten()
    print(len(y))




    return X, y

from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, f1_score


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

def test_auc():
    X, y = load_kdd99_http_data()

    test_size = int(len(X) * 0.1)
    print(test_size)

    X_train = X[:test_size]
    X_test = X[test_size:]

    y_train = y[:test_size]
    y_test = y[test_size:]
    print(len(y_test))
    print(len(X_test))





    sample_size = 64
    tree_num = 256
    u = 0.5

    iforest_asd = IForest(sample_size, tree_num)
    start_time = time.time()
    iforest_asd.fit(X_train)
    end_time = time.time()
    print(f"Program train time: {end_time - start_time:.2f} seconds")
    start_time = time.time()

    predictions = iforest_asd.predict(X_test, u)

    end_time = time.time()
    print(f"Program test time: {end_time - start_time:.2f} seconds")

    auc = roc_auc_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print(f"F1 Score: {f1}")
    print(f"AUC: {auc}")

    print('sample size =',sample_size,'tree num =', tree_num,'threshold =', u)


    total_time = end_time - start_time
    plot_roc_curve(y_test, predictions, 'ROC Curve', total_time)

    sample_size = 256


    iforest_asd = IForest(sample_size, tree_num)
    start_time = time.time()
    iforest_asd.fit(X_train)
    end_time = time.time()
    print(f"Program train time: {end_time - start_time:.2f} seconds")
    start_time = time.time()

    predictions = iforest_asd.predict(X_test, u)

    end_time = time.time()
    print(f"Program test time: {end_time - start_time:.2f} seconds")
    total_time = end_time - start_time

    auc = roc_auc_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print(f"F1 Score: {f1}")
    print(f"AUC: {auc}")

    print('sample size =',sample_size,'tree num =', tree_num,'threshold =', u)



    plot_roc_curve(y_test, predictions, 'ROC Curve', total_time)

    sample_size = 1024


    iforest_asd = IForest(sample_size, tree_num)
    start_time = time.time()
    iforest_asd.fit(X_train)
    end_time = time.time()
    print(f"Program train time: {end_time - start_time:.2f} seconds")
    start_time = time.time()
    print('start')
    predictions = iforest_asd.predict(X_test, u)

    end_time = time.time()
    print(f"Program test time: {end_time - start_time:.2f} seconds")
    total_time = end_time - start_time

    auc = roc_auc_score(y_test, predictions)

    f1 = f1_score(y_test, predictions)

    print(f"F1 Score: {f1}")
    print(f"AUC: {auc}")

    print('sample size =',sample_size,'tree num =', tree_num,'threshold =', u)

    print('for http')


    plot_roc_curve(y_test, predictions, 'ROC Curve', total_time)

test_auc()
