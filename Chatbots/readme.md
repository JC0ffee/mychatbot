### Introduction

This README provides comprehensive guidelines for implementing and testing the ABA-I Chatbot Machine Learning model through an API built with Flask. It covers setting up the API with Flask, testing it using Postman, and integrating it into an Android application using Retrofit.

### Table of Contents

1. **ABA-I CHATBOT ML**
   - Overview
   - Training Metrics

2. **Pembuatan API dengan Flask**
   - Langkah 1: Instalasi Flask
   - Langkah 2: Implementasi API Flask
   - Langkah 3: Menjalankan API Flask
   - Langkah 4: Pengujian API dengan Postman

3. **Implementasi API Flask pada Aplikasi Android**
   - Setup Retrofit Library
   - Creating API Interface
   - Creating POJO Classes
   - Setting Up Retrofit Client
   - Using API Interface in Android Application

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

### Pembuatan API dengan Flask

#### Langkah 1 : Instalasi Flask

1. Instal Python dari [python.org](https://www.python.org/downloads/)
2. Instal Flask menggunakan pip (package installer Python)

#### Langkah 2 : Implementasi API Flask

Contoh implementasi API sederhana

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
