from flask import Flask, request, jsonify, render_template
import pandas as pd
import re
from cassandra.cluster import Cluster
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import logging

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# NLTK setup for preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

app = Flask(__name__)

# 1. Connect to Cassandra
def connect_to_cassandra():
    try:
        cluster = Cluster(['127.0.0.1'])
        session = cluster.connect()
        session.set_keyspace('gestionspam')
        logger.info("Connected to Cassandra successfully.")
        return session
    except Exception as e:
        logger.error(f"Error connecting to Cassandra: {e}")
        exit()

# 2. Fetch data from Cassandra
def fetch_data_from_cassandra(session):
    try:
        rows = session.execute("SELECT content, label FROM messages")
        data = [{'email_text': row.content, 'is_spam': 1 if row.label == 'spam' else 0} for row in rows]
        logger.info("Data fetched successfully from Cassandra.")
        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# 3. Preprocess emails
def preprocess_email(email):
    email = re.sub(r'\W', ' ', email)  # Remove non-alphanumeric characters
    email = email.lower()  # Convert to lowercase
    email = re.sub(r'\s+', ' ', email)  # Replace multiple spaces with a single space
    email = ' '.join([lemmatizer.lemmatize(word) for word in email.split() if word not in stop_words])
    return email

# Fonction pour entraîner le modèle de détection de spam
def train_spam_detector(data):
    logger.info("Entraînement du modèle de détection de spam...")

    # Prétraiter le texte des emails
    data['cleaned_text'] = data['email_text'].apply(preprocess_email)

    # Vectoriser le texte des emails en utilisant des bigrammes pour mieux capturer le contexte
    vectorizer = CountVectorizer(ngram_range=(1, 2))
    # Transformation du texte en vecteurs
    X = vectorizer.fit_transform(data['cleaned_text'])  
    y = data['is_spam']  # Variable cible (spam ou non)

    # Entraîner le modèle Naive Bayes
    model = MultinomialNB()
    model.fit(X, y)

    logger.info("Entraînement du modèle terminé.")
    return model, vectorizer


# Flask route to render the HTML page
@app.route("/")
def index():
    return render_template("index.html")

# Flask route to predict if an email is spam
@app.route("/predict", methods=["POST"])
def predict():
    email_text = request.json.get("email", "")
    if not email_text:
        return jsonify({"error": "No email text provided"}), 400

    cleaned_email = preprocess_email(email_text)
    email_vector = vectorizer.transform([cleaned_email])
    prediction = model.predict(email_vector)
    is_spam = bool(prediction[0])  # Convert numpy value to Python boolean

    return jsonify({"is_spam": is_spam})

if __name__ == "__main__":
    # Connect to Cassandra and fetch data
    session = connect_to_cassandra()
    data = fetch_data_from_cassandra(session)

    if not data.empty:
        # Train the spam detection model
        model, vectorizer = train_spam_detector(data)
        app.run(debug=True)
    else:
        logger.error("No data available in the database.")
