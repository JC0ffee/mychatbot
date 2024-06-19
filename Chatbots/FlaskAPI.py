from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import json
import random

# Inisialisasi Flask
app = Flask(__name__)

# Load model dan data
model = load_model('my100baby.h5')

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Load intents JSON data
with open('intents.json') as json_file:
    intents = json.load(json_file)

# Inisialisasi lemmatizer
lemmatizer = WordNetLemmatizer()

# Membuat list dari words dan classes
words = []
classes = []
documents = []
ignore_words = ['?', '!']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=False):
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
    ERROR_THRESHOLD = 0.3
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
    res = get_response(ints, intents)
    return res

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()
    message = data.get('message')
    response = chatbot_response(message)
    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(debug=True)
