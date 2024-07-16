from flask import Flask, request, render_template
import pickle
import re
import os
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Define the base directory for NLTK data
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')

# Set the NLTK data path
nltk.data.path.append(nltk_data_dir)

# Load the model
with open('samodel.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as vect_file:
    vectorizer = pickle.load(vect_file)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocessor(txt):
    txt = re.sub(r'http\S+', '', txt)  # Remove URLs
    txt = re.sub(r'@\w+', '', txt)  # Remove mentions
    txt = re.sub(r'#', '', txt)  # Remove hashtags
    txt = re.sub(r'[^a-zA-Z\s]', '', txt)  # Remove non-alphabetic characters
    txt = txt.lower()  # Convert to lowercase
    tokens = word_tokenize(txt)  # Tokenize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def get_sentiment_message(polarity):
    sentiment_dict = {
        0: "Negative",
        4: "Positive"
    }
    return sentiment_dict.get(polarity, "Neutral")

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    sentiment_message = None
    if request.method == 'POST':
        user_input = request.form['text']
        processed_input = preprocessor(user_input)
        processed_input_vectorized = vectorizer.transform([processed_input])
        prediction = model.predict(processed_input_vectorized)[0]
        sentiment_message = get_sentiment_message(prediction)
    
    return render_template('index.html', prediction=sentiment_message)

if __name__ == "__main__":
    app.run(debug=True)
