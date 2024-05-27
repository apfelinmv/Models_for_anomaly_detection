import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, RepeatVector, TimeDistributed, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2
import time


# Load  dataset
X = np.genfromtxt('dataset', delimiter=',')
y = pd.read_csv('dataset_labels').to_numpy().flatten()

print(f"Shape of features (X): {X.shape}")
print(f"Shape of labels (y): {y.shape}")

X = X[:len(y)]

print(f"Shape of features (X) after truncation: {X.shape}")
print(f"Shape of labels (y): {y.shape}")

# Normalize features
scaler = StandardScaler()
features = scaler.fit_transform(X)

labels = y

timesteps = 30
features_num = features.shape[1]

assert len(features) == len(labels), "Features and labels must have the same length"

sequences = []
sequence_labels = []

for i in range(len(features) - timesteps + 1):
    sequence = features[i:i + timesteps]
    sequences.append(sequence)
    sequence_labels.append(labels[i + timesteps - 1])  # Align labels with sequences

data = np.array(sequences)
labels = np.array(sequence_labels)


train_data = data[:6000]
train_labels = labels[:6000]
val_data = data[6000:]
val_labels = labels[6000:]
# Splitting data into training and validation sets
#train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=42, shuffle=True)

# Build LSTM autoencoder model
inputs = Input(shape=(timesteps, features_num))
encoded = LSTM(16, activation='relu', return_sequences=True, kernel_regularizer=l2(0.01))(inputs)
encoded = BatchNormalization()(encoded)
encoded = LSTM(4, activation='relu', return_sequences=False, kernel_regularizer=l2(0.01))(encoded)
encoded = BatchNormalization()(encoded)

# Decoder
decoded = RepeatVector(timesteps)(encoded)
decoded = LSTM(4, activation='relu', return_sequences=True, kernel_regularizer=l2(0.01))(decoded)
decoded = BatchNormalization()(decoded)
decoded = LSTM(16, activation='relu', return_sequences=True, kernel_regularizer=l2(0.01))(decoded)
decoded = BatchNormalization()(decoded)
decoded = TimeDistributed(Dense(features_num))(decoded)

autoencoder = Model(inputs, decoded)
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Measure training time
start_time = time.time()
# Train the model and store the history
history = autoencoder.fit(train_data, train_data, epochs=20, batch_size=64, validation_data=(val_data, val_data), callbacks=[early_stopping])
end_time = time.time()
training_time = end_time - start_time
print(f"Training Time: {training_time:.2f} seconds")


# Plot the training and validation loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

print(autoencoder.summary())

def find_optimal_threshold(reconstruction_error, labels):
    precision, recall, thresholds = precision_recall_curve(labels, reconstruction_error)
    f1_scores = np.zeros_like(thresholds)

    for i in range(len(thresholds)):
        if (precision[i] + recall[i]) > 0:
            f1_scores[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
        else:
            f1_scores[i] = 0

    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold, f1_scores[optimal_idx]


def plot_anomalies_with_optimal_threshold(autoencoder, data, labels):
    # Measure testing time
    start_time = time.time()
    reconstructed_data = autoencoder.predict(data)
    reconstruction_error = np.mean(np.square(data - reconstructed_data), axis=(1, 2))
    end_time = time.time()
    testing_time = end_time - start_time
    print(f"Testing Time: {testing_time:.2f} seconds")

    optimal_threshold, optimal_f1 = find_optimal_threshold(reconstruction_error, labels)
    print(f'Optimal Threshold: {optimal_threshold:.4f}')
    print(f'Optimal F1 Score: {optimal_f1:.4f}')

    anomalies = reconstruction_error > optimal_threshold

    plt.figure(figsize=(12, 6))
    plt.plot(reconstruction_error, label='Reconstruction Error')
    plt.hlines(optimal_threshold, xmin=0, xmax=len(reconstruction_error), colors='r', label='Optimal Threshold')
    plt.scatter(np.where(anomalies)[0], reconstruction_error[anomalies], color='r', label='Anomalies')

    plt.title('Reconstruction Error and Anomalies')
    plt.xlabel('Data Point Index')
    plt.ylabel('Reconstruction Error')
    plt.legend()
    plt.show()

    auc = roc_auc_score(labels, reconstruction_error)
    print(f'AUC: {auc:.4f}')

    f1 = f1_score(labels, anomalies)
    print(f'F1 Score: {f1:.4f}')

    fpr, tpr, roc_thresholds = roc_curve(labels, reconstruction_error)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')
    plt.show()


plot_anomalies_with_optimal_threshold(autoencoder, data, labels)