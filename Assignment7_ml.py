import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Parameters
VOCAB_SIZE = 10000
MAX_LEN = 200

# Load dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)

# Decode reviews
word_index = imdb.get_word_index()
reverse_index = {v + 3: k for k, v in word_index.items()}
reverse_index[0] = "<PAD>"
reverse_index[1] = "<START>"
reverse_index[2] = "<UNK>"
reverse_index[3] = "<UNUSED>"

def decode_review(sequence):
    return " ".join([reverse_index.get(i, "?") for i in sequence])

decoded_train = [decode_review(review) for review in x_train]
decoded_test = [decode_review(review) for review in x_test]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(decoded_train)
X_test_tfidf = vectorizer.transform(decoded_test)

# Naive Bayes Model
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
y_pred_nb = nb_model.predict(X_test_tfidf)

# LSTM Deep Learning Model
x_train_padded = pad_sequences(x_train, maxlen=MAX_LEN, padding='post')
x_test_padded = pad_sequences(x_test, maxlen=MAX_LEN, padding='post')

dl_model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=64, input_length=MAX_LEN),
    LSTM(64, dropout=0.3, recurrent_dropout=0.3),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

dl_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = dl_model.fit(x_train_padded, y_train, validation_data=(x_test_padded, y_test), epochs=3, batch_size=128, verbose=2)

# Evaluation
print("\nClassification Report - Naive Bayes:")
print(classification_report(y_test, y_pred_nb))

dl_score = dl_model.evaluate(x_test_padded, y_test, verbose=0)

# Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_nb), annot=True, fmt='d', cmap='coolwarm')
plt.title("Naive Bayes Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("conf_matrix_nb.png")
plt.show()

# Accuracy Plot
plt.figure(figsize=(8, 4))
plt.plot(history.history['accuracy'], label='Train', linestyle='--')
plt.plot(history.history['val_accuracy'], label='Validation', linestyle='-')
plt.title("LSTM Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("lstm_accuracy_graph.png")
plt.show()

# Summary Table
comparison = pd.DataFrame({
    'Model': ['Naive Bayes', 'LSTM Model'],
    'Accuracy': [nb_model.score(X_test_tfidf, y_test), dl_score[1]]
})
print("\nModel Performance Comparison:")
print(comparison)
