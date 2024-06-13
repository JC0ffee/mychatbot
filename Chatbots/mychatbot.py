import json
import numpy as np
import random
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Load intents JSON from file
with open('intents.json') as file:
    data = json.load(file)

lemmatizer = WordNetLemmatizer()

words = []
classes = []
documents = []
ignore_words = ['?', '!']

# Loop through each intent in the data
for intent in data['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Initialize training data
training_data = []

# Loop through each document
for doc in documents:
    # Initialize bag of words
    bag = []
    # Get the patterns and tag
    pattern_words, tag = doc
    # Lemmatize and normalize the pattern words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # Create bag of words
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)

    # Initialize output row
    output_row = [0] * len(classes)
    output_row[classes.index(tag)] = 1

    # Concatenate bag of words and output row into a single list
    training_data.append([bag + output_row])

# Shuffle training data
random.shuffle(training_data)

# Convert training data to NumPy array
training_data = np.array(training_data)

# Reshape training data to remove extra dimension
training_data = training_data.reshape(training_data.shape[0], -1)

# Split input and output
train_x = training_data[:, :-len(classes)]
train_y = training_data[:, -len(classes):]

print("Shape of training data:", training_data.shape)

# BAGIAN 2: Melatih Model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax', kernel_regularizer=l2(0.01)))

# Using Adam optimizer with a different learning rate
adam = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# Custom callback for saving model with different names based on conditions
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
                self.model.save(f"model{epoch + 1}.h5")  # Save model with epoch number
        if current_accuracy >= 0.9:  # Save model if val_accuracy reaches 100%
            self.model.save(f"model100.h5")

# Set up early stopping and custom model checkpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='auto')
custom_checkpoint = SaveModelOnAccuracy('chatbot_model.h5', save_best_only=True)

history = model.fit(train_x, train_y, epochs=1000, batch_size=16, verbose=1, validation_split=0.1, callbacks=[early_stopping, custom_checkpoint])

print(f"Best Validation Accuracy: {custom_checkpoint.best_accuracy}")

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Metrics')
plt.xlabel('Epochs')
plt.ylabel('Metrics')
plt.legend()
plt.show()

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"Found in bag: {w}")
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.2
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    if not return_list:
        return_list.append({"intent": "no_match", "probability": "0.0"})
    return return_list

def get_story_recommendation(intents):
    for intent in intents['intents']:
        if intent['tag'] == 'recommendation':
            if 'stories' in intent:
                stories = intent['stories']
                recommended_story = random.choice(stories)
                return recommended_story
    return "Maaf, tidak ada cerita yang tersedia saat ini."

def get_response(ints, intents_json):
    if not ints:
        return "Maaf, saya tidak mengerti. Jika kamu membutuhkan bantuan, kamu bisa menuliskan 'help' atau 'tolong'."

    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']

    for intent in list_of_intents:
        if intent['tag'] == tag:
            if tag == 'recommendation':
                recommended_story = get_story_recommendation(intents_json)
                result = f"Menurutku, cerita yang menarik untuk kamu adalah {recommended_story}."
            elif tag == 'response_to_greeting':
                result = random.choice(intent['responses'])
            else:
                result = random.choice(intent['responses'])
            break
    else:
        result = "Maaf, saya tidak mengerti. Jika kamu membutuhkan bantuan, kamu bisa menuliskan 'help' atau 'tolong'."

    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = get_response(ints, data)
    return res

# FINAL STEP TESTING
while True:
    message = input("Kamu: ")
    if message.lower() == "finish":
        break
    response = chatbot_response(message)
    print(f"Bot: {response}")