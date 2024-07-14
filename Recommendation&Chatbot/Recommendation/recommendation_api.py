from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import load_model

nltk.download('punkt')
nltk.download('stopwords')

# Path absolut ke model
model_path = 'E:/Visual Studio/Workspace_vscode/Recommendation/rekomendasiByStoryID.h5'
# Load the model
model = load_model(model_path)

# Load the data
data_path = 'E:/Visual Studio/Workspace_vscode/Recommendation/storyy.csv'
df_story = pd.read_csv(data_path)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('indonesian')]
    return ' '.join(tokens)

# Update required_columns dengan nama kolom yang benar
required_columns = ['overview', 'penulis', 'daerah-asal', 'genre']

# Preprocessing kolom yang ada
for col in required_columns:
    if col in df_story.columns:
        df_story[col] = df_story[col].apply(preprocess_text)

df_story['combined_text'] = df_story.apply(lambda row: ' '.join([
    row['overview'],
    row['penulis'],
    row['daerah-asal'],
    row['genre']
]), axis=1)

vectorizer = TfidfVectorizer(max_features=951)
text_vectors = vectorizer.fit_transform(df_story['combined_text']).toarray()

def recommend_stories_for_story(story_id, top_n=5):
    story_text = df_story[df_story['id'] == story_id]['combined_text'].values[0]
    processed_story_text = preprocess_text(story_text)
    story_vector = vectorizer.transform([processed_story_text]).toarray()
    predictions = model.predict(story_vector)[0]
    recommended_story_indices = np.argsort(predictions)[-top_n:][::-1]
    recommended_stories = df_story.iloc[recommended_story_indices]
    recommended_stories = recommended_stories[recommended_stories['id'] != story_id]
    return recommended_stories, predictions

app = Flask(__name__)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    story_id = data.get('story_id')

    if story_id is None or story_id not in df_story['id'].values:
        return jsonify({"error": "Story ID tidak ditemukan. Silakan coba lagi."}), 400

    selected_story = df_story[df_story['id'] == story_id]
    recommended_stories, scores = recommend_stories_for_story(story_id)

    recommendations = []
    for idx, story in recommended_stories.iterrows():
        recommendations.append({
            "judul": story['judul'],
            "score": float(scores[idx])
        })

    return jsonify({
        "story_id": story_id,
        "selected_story": selected_story['judul'].values[0],
        "recommendations": recommendations
    })

if __name__ == '__main__':
    app.run(debug=True)
