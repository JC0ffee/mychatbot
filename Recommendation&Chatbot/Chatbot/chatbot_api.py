from flask import Flask, request, jsonify
import json
import nltk
import random
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

nltk.download('punkt')
nltk.download('wordnet')

# Initialize Flask app
app = Flask(__name__)

# Load model with a compatible optimizer
model_path = 'I:\Demo\Script\Chatbot\chatbot_model.h5'
model = load_model(model_path, compile=False)
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Load intents JSON from file
with open('I:\Demo\Script\Chatbot\intents.json') as file:
    data = json.load(file)

lemmatizer = WordNetLemmatizer()

# Preparing words and classes from intents data
words = []
classes = []
ignore_words = ['?', '!']

for intent in data['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.2
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
    if not return_list:
        return_list.append({"intent": "no_match", "probability": "0.0"})
    return return_list

def get_story_recommendation(intents):
    for intent in intents['intents']:
        if intent['tag'] == 'recommendation' and 'stories' in intent:
            return random.choice(intent['stories'])
    return "Maaf, tidak ada cerita yang tersedia saat ini."

def welcome_message():
    return "Selamat datang di aplikasi ABA-I, saya adalah Personal Assistant untuk membantumu menjelajahi aplikasi ini. Bagaimana saya bisa membantu kamu hari ini?"

def get_response(ints, intents_json, first_session=True):
    if first_session:
        return welcome_message()
    if not ints:
        return "Maaf, saya tidak mengerti. Jika kamu membutuhkan bantuan, kamu bisa menuliskan 'help' atau 'tolong'."
    
    tag = ints[0]['intent']
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            if tag == 'recommendation':
                recommended_story = get_story_recommendation(intents_json)
                return f"Menurutku, cerita yang menarik untuk kamu adalah {recommended_story}."
            else:
                return random.choice(intent['responses'])
    
    return "Maaf, saya tidak mengerti. Jika kamu membutuhkan bantuan, kamu bisa menuliskan 'help' atau 'tolong'."

def chatbot_response(msg, first_session=True):
    msg = msg.replace('\n', ' ')
    ints = predict_class(msg, model)
    return get_response(ints, data, first_session)

@app.route('/chatbot', methods=['POST'])
def chatbot():
    req_data = request.get_json()
    user_message = req_data.get('message')
    first_session = req_data.get('first_session', False)

    response = chatbot_response(user_message, first_session)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
