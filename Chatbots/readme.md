### Introduction

This README provides comprehensive guidelines for implementing and testing the ABA-I Chatbot Machine Learning model through an API built with Flask. It covers setting up the API with Flask, testing it using Postman, and integrating it into an Android application using Retrofit.

### Table of Contents

1. [ABA-I CHATBOT ML](#aba-i-chatbot-ml)
   - [Overview](#overview)
   - [Training Metrics](#training-metrics)
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
4. [Intents JSON File](https://github.com/JC0ffee/mychatbot/blob/main/Chatbots/intents.json)

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

### Creating API with Flask

#### Step 1: Installing Flask

1. Install Python from [python.org](https://www.python.org/downloads/).
2. Install Flask using pip (Python package installer).

#### Step 2: Implementing Flask API

Example of a simple API implementation using Flask:

```python
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
