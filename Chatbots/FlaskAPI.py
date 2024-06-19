from flask import Flask, request, jsonify
import json
import nltk
import random
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import re

model = load_model('100model.h5')

with open('intents.json') as file:
    data = json.load(file)

lemmatizer = WordNetLemmatizer()

words = []
classes = []
ignore_words = ['?', '!']

# Prepare the data
for intent in data['intents']:
    for pattern in intent['patterns']:
        clean_pattern = re.sub(r'\n', ' ', pattern)
        word_list = nltk.word_tokenize(clean_pattern)
        words.extend(word_list)
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

def clean_up_sentence(sentence):
    clean_sentence = re.sub(r'\n', ' ', sentence)
    sentence_words = nltk.word_tokenize(clean_sentence)
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

def welcome_message():
    return "Selamat datang di aplikasi ABA-I, saya adalah Personal Assistant untuk membantumu menjelajahi aplikasi ini. Bagaimana saya bisa membantu kamu hari ini?"

def get_response(ints, intents_json, user_name="", first_session=True):
    if first_session:
        response = welcome_message()
    else:
        if not ints:
            response = "Maaf, saya tidak mengerti. Jika kamu membutuhkan bantuan, kamu bisa menuliskan 'help' atau 'tolong'."
        else:
            tag = ints[0]['intent']
            list_of_intents = intents_json['intents']
            for intent in list_of_intents:
                if intent['tag'] == tag:
                    if tag == 'recommendation':
                        recommended_story = get_story_recommendation(intents_json)
                        response = f"Menurutku, cerita yang menarik untuk kamu adalah {recommended_story}."
                    else:
                        response = random.choice(intent['responses'])
                    break
            else:
                response = "Maaf, saya tidak mengerti. Jika kamu membutuhkan bantuan, kamu bisa menuliskan 'help' atau 'tolong'."
    return response

def chatbot_response(msg, user_name="", first_session=True):
    msg = msg.replace('\n', ' ')
    ints = predict_class(msg, model)
    res = get_response(ints, data, user_name, first_session)
    return res

app = Flask(__name__)

@app.route('/chatbot', methods=['POST'])
def chat():
    request_data = request.get_json()
    message = request_data['message']
    first_session = request_data.get('first_session', True)
    response = chatbot_response(message, first_session=first_session)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
