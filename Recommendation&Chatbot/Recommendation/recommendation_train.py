import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import train_test_split
import tensorflow as tf
import random

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Download required nltk data
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
df_story = pd.read_csv('storyy.csv')

# Display basic information
print(df_story.head())
print(df_story.info())
print("Jumlah Missing Values dalam data:")
print(df_story.isna().sum())
print("Jumlah Duplikat dalam data:", df_story.duplicated().sum())
print("Apakah terdapat data bernilai ? dalam data:")
print(df_story.isin(['?']).any())
print("Jumlah nilai Nan dalam data:")
print(df_story.isna().sum())

# Drop unnecessary columns and rename column
df_story = df_story.drop(['karakter-utama', 'tema'], axis=1)
df_story.rename(columns={"daerah-asal": "daerah_asal"}, inplace=True)
print(df_story.tail())

# Plot the number of stories based on the region of origin
story_counts = df_story['daerah_asal'].value_counts().sort_index()
fig = go.Figure(data=go.Bar(x=story_counts.index, y=story_counts.values))
fig.update_layout(
    plot_bgcolor='rgb(17, 17, 17)',
    paper_bgcolor='rgb(17, 17, 17)',
    font_color='white',
    title='Number of Story Based on Region of Origin',
    xaxis=dict(title='Region'),
    yaxis=dict(title='Number of Story')
)
fig.update_traces(marker_color='pink')
fig.show()

# Display unique genre labels
unique_genre = df_story['genre'].unique()
print("Variasi penulisan label-genre:")
print(unique_genre)

# Plot genre distribution
story_genre_counts = df_story['genre'].value_counts()
fig = go.Figure(data=go.Pie(labels=story_genre_counts.index, values=story_genre_counts.values))
fig.update_layout(plot_bgcolor='rgb(17, 17, 17)', paper_bgcolor='rgb(17, 17, 17)', font_color='white', title='Genre Distribution of Stories')
fig.show()

# Plot number of stories by top 5 authors
penulis_counts = df_story['penulis'].value_counts()
top_5_penulis = penulis_counts.head(5)
fig = go.Figure(data=go.Bar(x=top_5_penulis.index, y=top_5_penulis.values))
fig.update_layout(
    plot_bgcolor='rgb(17, 17, 17)',
    paper_bgcolor='rgb(17, 17, 17)',
    font_color='white',
    title='Number of Stories by Top 5 Authors',
    xaxis=dict(title='Author'),
    yaxis=dict(title='Number of Stories')
)
fig.update_traces(marker_color='orange')
fig.show()

# Preprocess text function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('indonesian')]
    return ' '.join(tokens)

# Apply preprocessing to text columns
for col in ['overview', 'penulis', 'daerah_asal', 'genre']:
    df_story[col] = df_story[col].apply(preprocess_text)

# Combine text columns into a single column
df_story['combined_text'] = df_story.apply(lambda row: ' '.join([
    row['overview'],
    row['penulis'],
    row['daerah_asal'],
    row['genre']
]), axis=1)

# Vectorize the text data
vectorizer = TfidfVectorizer()
text_vectors = vectorizer.fit_transform(df_story['combined_text']).toarray()

num_stories = len(df_story)
input_shape = text_vectors.shape[1]
input_text = Input(shape=(input_shape,))
x = Dense(128, activation='relu')(input_text)
x = Dense(64, activation='relu')(x)
output = Dense(num_stories, activation='softmax')(x)

model = Model(inputs=input_text, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_labels = np.eye(num_stories)
X_train, X_val, y_train, y_val = train_test_split(text_vectors, train_labels, test_size=0.2, random_state=42)

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_val, y_val))

# Save the trained model to a .h5 file
model.save('story_recommendation_model.h5')

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plot training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Recommendation function
def recommend_stories_for_story(story_id, top_n=5):
    story_text = df_story[df_story['id'] == story_id]['combined_text'].values[0]
    processed_story_text = preprocess_text(story_text)
    story_vector = vectorizer.transform([processed_story_text]).toarray()
    predictions = model.predict(story_vector)
    recommended_story_indices = np.argsort(predictions[0])[-(top_n + 1):][::-1]
    recommended_stories = df_story.iloc[recommended_story_indices]
    recommended_stories = recommended_stories[recommended_stories['id'] != story_id][:top_n]
    return recommended_stories

# Get recommendations for a specific story
story_id = int(input("Story ID:"))  # Specify story ID
recommended_stories = recommend_stories_for_story(story_id)

print("Rekomendasi cerita untuk cerita ID", story_id, ":")
for idx, story in recommended_stories.iterrows():
    print(f"- {story['judul']}")
