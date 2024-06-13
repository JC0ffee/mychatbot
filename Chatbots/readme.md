## Cerita Rakyat Chatbot by Jason
![Plot](https://github.com/JC0ffee/Chatbots/blob/23c71d7edcc7242d1ba83f45b79182168e0018d0/TrainingandValidationMetrics.png)

***Output Training***
###
Epoch 465/1000
5/5 [==============================] - 0s 6ms/step

Batch Detail:
- Batch Size: 5 samples per batch
- Time per Batch: 6ms

Metrics on Current Batch:
- Loss: 1.6187
- Accuracy: 96.20%
  
Best Validation Accuracy:
- Best Validation Accuracy: 100%
  ###
**requirements.txt**
```
Flask==2.0.1
tensorflow==2.6.0
nltk==3.6.2
```

**main.py**
```
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import json
import random

app = Flask(__name__)

model = load_model('chatbot_model.h5')

nltk.download('punkt')
nltk.download('wordnet')

with open('intents.json') as json_file:
    intents = json.load(json_file)

lemmatizer = WordNetLemmatizer()

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
```

## Android Studio
>1.Add Retrofit Dependencies in build.gradle
```
implementation 'com.squareup.retrofit2:retrofit:2.9.0'
implementation 'com.squareup.retrofit2:converter-gson:2.9.0'
```
>2.Create API Interface in Android Studio
```
import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.POST;

public interface ChatbotAPI {
    @POST("/chatbot")
    Call<ChatbotResponse> getResponse(@Body MessageRequest messageRequest);
}
```
>3.Create Model Classes for Request and Response
```
public class MessageRequest {
    private String message;

    public MessageRequest(String message) {
        this.message = message;
    }

    public String getMessage() {
        return message;
    }

    public void setMessage(String message) {
        this.message = message;
    }
}

public class ChatbotResponse {
    private String response;

    public String getResponse() {
        return response;
    }

    public void setResponse(String response) {
        this.response = response;
    }
}
```
>4.Create Retrofit Instance
```
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

Retrofit retrofit = new Retrofit.Builder()
        .baseUrl("https://your-project-id.appspot.com/")
        .addConverterFactory(GsonConverterFactory.create())
        .build();

ChatbotAPI chatbotAPI = retrofit.create(ChatbotAPI.class);
```

>5.Send Request to API
```
MessageRequest request = new MessageRequest("Hi");
Call<ChatbotResponse> call = chatbotAPI.getResponse(request);

call.enqueue(new Callback<ChatbotResponse>() {
    @Override
    public void onResponse(Call<ChatbotResponse> call, Response<ChatbotResponse> response) {
        if (response.isSuccessful()) {
            ChatbotResponse chatbotResponse = response.body();
            if (chatbotResponse != null) {
                String botResponse = chatbotResponse.getResponse();
                // Nampilin response di UI
            }
        }
    }

    @Override
    public void onFailure(Call<ChatbotResponse> call, Throwable t) {
        // Handle failure
    }
});
```
