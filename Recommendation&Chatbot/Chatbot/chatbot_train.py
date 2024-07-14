import json
import numpy as np
import random
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt

# Download data NLTK
nltk.download('punkt')
nltk.download('wordnet')

# Memuat intents JSON dari file
with open('intents.json') as file:
    data = json.load(file)

lemmatizer = WordNetLemmatizer()

words = []
classes = []
documents = []
ignore_words = ['?', '!']

# Memuat stopwords Bahasa Indonesia
stopwords_file = 'indonesianstopwords.txt'
stop_words = set()
with open(stopwords_file, 'r', encoding='utf-8') as f:
    for word in f.readlines():
        stop_words.add(word.strip())

# Loop melalui setiap intent dalam data
for intent in data['intents']:
    for pattern in intent['patterns']:
        # Menghapus stopwords dari pola
        pattern_words = [word.lower() for word in pattern.split() if word.lower() not in stop_words]
        words.extend(pattern_words)
        documents.append((pattern_words, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lematisasi kata-kata
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Inisialisasi data pelatihan
training_data = []

# Loop melalui setiap dokumen
for doc in documents:
    # Inisialisasi bag of words
    bag = []
    # Mendapatkan pola dan tag
    pattern_words, tag = doc
    # Lematisasi dan normalisasi pola kata
    # Membuat bag of words
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)

    # Inisialisasi baris output
    output_row = [0] * len(classes)
    output_row[classes.index(tag)] = 1

    # Menggabungkan bag of words dan baris output ke dalam satu daftar
    training_data.append(bag + output_row)

# Mengacak data pelatihan
random.shuffle(training_data)

# Mengonversi data pelatihan ke array NumPy
training_data = np.array(training_data)

train_x = training_data[:, :-len(classes)]
train_y = training_data[:, -len(classes):]

print("Shape of training data:", training_data.shape)

# Melatih Model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax', kernel_regularizer=l2(0.01)))

adam = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# Callback untuk menyimpan model berdasarkan akurasi
class SaveModelOnAccuracy(Callback):
    def __init__(self, filepath, monitor='val_accuracy', save_best_only=True):
        super(SaveModelOnAccuracy, self).__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.best_accuracy = -float('inf')

    def on_epoch_end(self, epoch, logs=None):
        current_accuracy = logs.get(self.monitor)
        if current_accuracy is None:
            return
        if current_accuracy > self.best_accuracy:
            self.best_accuracy = current_accuracy
            if self.save_best_only:
                self.model.save(self.filepath)
            else:
                self.model.save(f"model{epoch + 1}.h5")
        if current_accuracy >= 1.0:
            self.model.save(f"model100.h5")

# Early stopping dan custom model checkpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='auto')
custom_checkpoint = SaveModelOnAccuracy('training_model.h5', save_best_only=True)

history = model.fit(train_x, train_y, epochs=1000, batch_size=16, verbose=1, validation_split=0.1, callbacks=[early_stopping, custom_checkpoint])

print(f"Best Validation Accuracy: {custom_checkpoint.best_accuracy}")

# Plotting grafik pelatihan dan validasi
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Metrics')
plt.xlabel('Epochs')
plt.ylabel('Metrics')
plt.legend()
plt.show()
