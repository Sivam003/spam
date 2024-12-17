from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from langdetect import detect, DetectorFactory
import random

app = Flask(__name__)

DetectorFactory.seed = 0

def train_model():
    emails = [
        "Congratulations, you have won a lottery!", 
        "You have been selected for a prize.",
        "Meeting scheduled at 3 PM tomorrow.",
        "Please find the attached document.",
        "Claim your free gift card now!",
        "Reminder: Submit the report by Friday.",
        "आपको एक मुफ्त गिफ्ट कार्ड मिला है। अभी अपनी विजेता राशि प्राप्त करें!",  # Hindi spam
        "नमस्ते, कृपया ध्यान दें कि आपकी हिंदी कक्षा आज शाम 6 बजे होगी। धन्यवाद।",  # Hindi ham
        "¡Gana dinero rápido! Haz clic aquí para obtener tu oferta de dinero fácil y rápido.",  # Spanish spam
        "Hola, este es un recordatorio para la reunión de equipo programada para el próximo martes a las 11 AM.",  # Spanish ham
    ]
    labels = [1, 1, 0, 0, 1, 0, 1, 0, 1, 0]  

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(emails)

    model = MultinomialNB()
    model.fit(X, labels)

    return model, vectorizer

model, vectorizer = train_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/inspiration')
def inspiration():
    return render_template('inspiration.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        mail_text = request.form.get('mail_text')
        if not mail_text:
            return jsonify({"error": "No email text provided."})
        
        detected_language = detect(mail_text)
        
        transformed_text = vectorizer.transform([mail_text])
        prediction = model.predict(transformed_text)[0]
        result = "Ham" if prediction == 0 else "Spam"
        return jsonify({"result": result, "language": detected_language})

if __name__ == "__main__":
    app.run(debug=True)
