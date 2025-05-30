import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense


vocab_size = 10000
max_len = 200


(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

word_index = imdb.get_word_index()
index_word = {v + 3: k for k, v in word_index.items()}
index_word[0] = '<PAD>'
index_word[1] = '<START>'
index_word[2] = '<UNK>'
index_word[3] = '<UNUSED>'

def decode_review(text):
    return ' '.join([index_word.get(i, '?') for i in text])

x_train_text = [decode_review(x) for x in x_train]
x_test_text = [decode_review(x) for x in x_test]

tfidf = TfidfVectorizer(max_features=5000)
x_train_tfidf = tfidf.fit_transform(x_train_text)
x_test_tfidf = tfidf.transform(x_test_text)

lr_model = LogisticRegression(max_iter=300)
lr_model.fit(x_train_tfidf, y_train)
y_pred_lr = lr_model.predict(x_test_tfidf)

x_train_pad = pad_sequences(x_train, maxlen=max_len, padding='post')
x_test_pad = pad_sequences(x_test, maxlen=max_len, padding='post')

dl_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=64, input_length=max_len),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

dl_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = dl_model.fit(x_train_pad, y_train, validation_data=(x_test_pad, y_test), epochs=3, batch_size=128)

lr_accuracy = accuracy_score(y_test, y_pred_lr)
lr_precision = precision_score(y_test, y_pred_lr)
lr_recall = recall_score(y_test, y_pred_lr)
lr_f1 = f1_score(y_test, y_pred_lr)

dl_eval = dl_model.evaluate(x_test_pad, y_test, verbose=0)
dl_accuracy = dl_eval[1]

results = pd.DataFrame({
    'Model': ['Logistic Regression', 'LSTM (Deep Learning)'],
    'Accuracy': [lr_accuracy, dl_accuracy],
    'Precision': [lr_precision, None],
    'Recall': [lr_recall, None],
    'F1-Score': [lr_f1, None]
})


conf_matrix = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix_lr.png")  
plt.show() 

plt.figure(figsize=(8, 4))
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('LSTM Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig("lstm_accuracy_plot.png")  
plt.show()  
print("\nModel Comparison Table:")
print(results)
