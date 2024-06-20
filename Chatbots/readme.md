### Introduction

This README provides comprehensive guidelines for implementing and testing the ABA-I Chatbot Machine Learning model through an API built with Flask. It covers setting up the API with Flask, testing it using Postman, and integrating it into an Android application using Retrofit.

### Table of Contents

1. [ABA-I CHATBOT ML](#aba-i-chatbot-ml)
   - [Overview](#overview)
   - [Training Metrics](#training-metrics)
   - [Intents JSON File](#intents-json-file)
   - [Training Process](#training-process)
   - [Chatbot Testing](#chatbot-testing)
2. [Creating API with Flask](#creating-api-with-flask)
   - [Step 1: Installing Flask](#step-1-installing-flask)
   - [Step 2: Implementing Flask API](#step-2-implementing-flask-api)
   - [Step 3: Running Flask API](#step-3-running-flask-api)
   - [Step 4: Testing API with Postman](#step-4-testing-api-with-postman)
3. [Implementing Flask API in Android Application](#implementing-flask-api-in-android-application)
   - [Setup Retrofit Library](#setup-retrofit-library)
   - [Creating API Interface](#creating-api-interface)
   - [Creating POJO Classes](#creating-pojo-classes)
   - [Setting Up Retrofit Client](#setting-up-retrofit-client)
   - [Using API Interface in Android Application](#using-api-interface-in-android-application)
     
### ABA-I CHATBOT ML

#### Overview
The ABA-I Chatbot ML is designed to assist users in navigating an application environment using natural language processing. It utilizes a machine learning model trained to recognize intents from user messages and respond accordingly.

#### Training Metrics
![Training Metrics](https://github.com/JC0ffee/mychatbot/blob/3599f765c835a5438c61fe9eb5c23c99cd4f5274/Chatbots/TrainingandValidationMetrics.png)

- **Output Training**
  - Epoch 465/1000
  - Batch Detail:
    - Batch Size: 5 samples per batch
    - Time per Batch: 6ms
  - Metrics on Current Batch:
    - Loss: 1.6187
    - Accuracy: 96.20%
  
- **Best Validation Accuracy:**
  - Best Validation Accuracy: 100%
 
### Intents JSON File
[Intents JSON File](https://github.com/JC0ffee/mychatbot/blob/main/Chatbots/intents.json)

### Training Process

The training process for the ABA-I Chatbot involves several key steps:

1. **Importing Libraries**: Import necessary libraries for NLP and model training.
   `pip install nltk numpy tensorflow matplotlib`

3. **Loading and Preprocessing Data**:
    - The `intents.json` file is loaded, containing training data with patterns and corresponding tags.
    - Words in the patterns are lemmatized (reduced to base form) and stop words (common words like 'the', 'is') are removed to reduce noise.
    - The text data is tokenized into words, and each word is associated with a tag.

4. **Preparing Training Data**:
    - A bag-of-words model is used to represent each pattern as a vector. Each vector element corresponds to a word in the vocabulary, with a value of 1 if the word is present in the pattern, and 0 otherwise.
    - Tags are one-hot encoded to create output vectors.

5. **Building the Neural Network Model**:
    - A Sequential model is built with dense layers and dropout for regularization.
    - The model is compiled using the Adam optimizer and categorical cross-entropy loss function.

6. **Training the Model**:
    - Early stopping and a custom callback are used to save the best model based on validation accuracy.
    - The model is trained with the training data for up to 1000 epochs, using 10% of the data for validation.

7. **Visualization**:
    - Training and validation loss and accuracy are plotted to visualize the model's performance over epochs.

## Chatbot Testing
### Function predict_class
```
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
```
The predict_class function predicts the intent of a user's input sentence using a trained neural network model. The prediction results are used to determine the appropriate response from the chatbot.

### Function bow
```
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
```
The bow (bag of words) function converts an input sentence into a vector representation using a bag of words approach. It marks the presence of words from a pre-processed and sorted list of lemmatized words.

### Function get_response
```
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
```
The get_response function retrieves the appropriate response based on the predicted intent from the user's input. It handles greetings, provides story recommendations, or defaults to a help message if the intent is not recognized.

**Chatbot Testing Overview**
To test the functionality of ABA-I, follow these steps:

Input Messages: Enter various types of messages that represent different intents, such as greetings, questions, or requests for recommendations.

Expected Outputs: Observe the responses generated by the chatbot based on the predicted intents. The chatbot should respond appropriately with greetings, recommended stories, or help messages if the intent is not recognized.

### Example Testing
```first_session = True

while True:
    if first_session:
        print(welcome_message())
        first_session = False
    message = input("Pesan: ")
    if message.lower() == "finish":
        break
    response = chatbot_response(message, first_session=first_session)
    print(f"Bot: {response}")
```
Session Initialization: The first_session variable ensures the chatbot starts with a welcome message in the first interaction.

Message Input: Users input messages (message) which are then processed by the chatbot_response function to generate a response.

Ending the Session: Typing "finish" ends the interaction loop.

### Creating API with Flask

#### Step 1: Installing Flask

1. Install Python from [python.org](https://www.python.org/downloads/).
2. Install Flask using pip (Python package installer).

#### Step 2: Implementing Flask API

Example of a simple API implementation using Flask:

```
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return "Welcome to my Flask API!"

@app.route('/greet/<name>')
def greet(name):
    return f"Hello, {name}!"

if __name__ == '__main__':
    app.run(debug=True)
 ```
### Step 3: Running Flask API

Open a terminal or command prompt, navigate to the directory where you have saved the 'FlaskAPI.py' file, then execute the command:

 ```
 python FlaskAPI.py
  ```
Output
* Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)

### Langkah 4 : Pengujian API dengan Postman
1. Install Postman from [getpostman.com](https://www.getpostman.com/downloads/)
2. Create a new tab in Postman and select the HTTP method (POST).
3. Enter the Endpoint URL http://127.0.0.1:5000/chatbot
4. Add a 'Content-Type' header with the value 'application/json'.
5. Select the 'Body' tab, choose 'raw' format, and set type to 'JSON'.
6. Enter data in the text area under the Body tab.
Example JSON Data:
 ```
{
    "message": "Halo",
    "first_session": false
}
 ```

## Implementing Flask API in Android Application

```
   implementation 'com.squareup.retrofit2:retrofit:2.9.0'
   implementation 'com.squareup.retrofit2:converter-gson:2.9.0' // Untuk mengonversi JSON response ke objek Java/Gson
   ```

**Create an interface for API**
```public interface ChatbotAPI {
    @POST("chatbot")
    Call<ChatbotResponse> sendMessage(@Body ChatbotRequest request);
}
```

**Create POJO classes for Request and Response**
```public class ChatbotRequest {
    private String message;
    // Constructor, getter, dan setter
}

public class ChatbotResponse {
    private String response;
    // Constructor, getter, dan setter
}
```
**Setting up Retrofit and Connecting to API**
```import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

public class RetrofitClient {
    private static Retrofit retrofit = null;

    public static Retrofit getClient(String baseUrl) {
        if (retrofit == null) {
            retrofit = new Retrofit.Builder()
                    .baseUrl(baseUrl)
                    .addConverterFactory(GsonConverterFactory.create())
                    .build();
        }
        return retrofit;
    }
}
```
**Using the API Interface**
```
Retrofit retrofit = RetrofitClient.getClient("http://alamat-ip-flask-anda:port/");
ChatbotAPI chatbotAPI = retrofit.create(ChatbotAPI.class);

// Buat objek ChatbotRequest
ChatbotRequest request = new ChatbotRequest();
request.setMessage("Halo");

// Kirim permintaan ke API
Call<ChatbotResponse> call = chatbotAPI.sendMessage(request);
call.enqueue(new Callback<ChatbotResponse>() {
    @Override
    public void onResponse(Call<ChatbotResponse> call, Response<ChatbotResponse> response) {
        if (response.isSuccessful()) {
            ChatbotResponse chatbotResponse = response.body();
            String botMessage = chatbotResponse.getResponse();
            // Gunakan botMessage untuk menampilkan respons dari chatbot
        } else {
            // Tangani respons tidak berhasil
        }
    }

    @Override
    public void onFailure(Call<ChatbotResponse> call, Throwable t) {
        // Tangani kegagalan koneksi atau permintaan
    }
});
```
