from flask import Flask, render_template, request,jsonify
from preprocess import preprocess
import pickle

app = Flask(__name__)

# Load the trained model and vectorizer
with open('LRmodel.pkl', 'rb') as file:
    model = pickle.load(file)

with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    preprocessed_text = preprocess(text)
    vectorized_text = vectorizer.transform([preprocessed_text])
    sentiment = model.predict(vectorized_text)[0]
    result = 'Positive' if sentiment == 1 else 'Negative'
    return render_template('index.html', text=text, result=result, sentiment=sentiment)

@app.route('/api/predict', methods=['POST'])
def predict_api():
    text = request.get_json(force=True)  # Get data posted as a json
    preprocessed_text = preprocess(text['content'])
    vectorized_text = vectorizer.transform([preprocessed_text]) # X 
    prediction = model.predict(vectorized_text)[0]
    prediction = 1 if prediction == 1 else 0
    return jsonify({'prediction': prediction, 'text': text['content']})  # Return prediction


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=9696,debug=True)