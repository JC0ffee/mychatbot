## ABA-I CHATBOT ML
![Plot](https://github.com/JC0ffee/mychatbot/blob/3599f765c835a5438c61fe9eb5c23c99cd4f5274/Chatbots/TrainingandValidationMetrics.png)

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

## Pembuatan API dengan Flask
### Langkah 1 : Instalasi Flask
1.Instal Python dari [python.org](https://www.python.org/downloads/)

2.Instal Flask menggunakan pip (package installer Python)

### Langkah 2 : Implementasi API Flask
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
 ```
### Langkah 3 : Menjalankan API Flask
Buka terminal atau command prompt,arahkan ke direktori tempat kalian menyimpan file 'FlaskAPI.py' lalu menjalankan perintah
 ```
 python FlaskAPI.py
  ```
Output
* Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)

### Langkah 4 : Pengujian API dengan Postman
1. Install Postman dari [getpostman.com](https://www.getpostman.com/downloads/)
2. Buat tab baru pada postman dan pilih metode HTTP (POST)
3. Masukkan URL Endpoint http://127.0.0.1:5000/chatbot
4. Tambahkan header 'Content-Type' dengan nilai 'application/json'
5. Pilih tab 'Body' dengan format 'raw' dan jenis 'JSON'
6. Masukkan data di area teks dibawah tab Body
Contoh
 ```
{
    "message": "Halo",
    "first_session": false
}
 ```





## Implementasi API Flask pada Aplikasi Android

```
   implementation 'com.squareup.retrofit2:retrofit:2.9.0'
   implementation 'com.squareup.retrofit2:converter-gson:2.9.0' // Untuk mengonversi JSON response ke objek Java/Gson
   ```

**Buat interface untuk API**
```public interface ChatbotAPI {
    @POST("chatbot")
    Call<ChatbotResponse> sendMessage(@Body ChatbotRequest request);
}
```

**Buat kelas POJO untuk Request dan Response**
```public class ChatbotRequest {
    private String message;
    // Constructor, getter, dan setter
}

public class ChatbotResponse {
    private String response;
    // Constructor, getter, dan setter
}
```
**Pengaturan Retrofit dan Koneksi ke API**
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
**Menggunakan Interface API**
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
