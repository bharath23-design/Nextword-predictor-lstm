from flask import Flask, render_template, request
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

app = Flask(__name__)

# Load the tokenizer and pre-trained model
with open('tokenizer.pkl', 'rb') as tk_file:
    loaded_tokenizer = pickle.load(tk_file)

model = load_model('nextwordDLQ&A.h5')

max_sequence_length = 27
def predict_next_word(seed_text):
    predicted_sentences = []

    for _ in range(10):  # Adjust the number of predictions as needed
        token_text = loaded_tokenizer.texts_to_sequences([seed_text])[0]

        if not token_text:
            return "Empty token_text. Ensure your tokenizer is trained correctly."

        padded_token_text = pad_sequences([token_text], maxlen=max_sequence_length, padding='pre')
        predictions = model.predict(padded_token_text)

        pos = np.argmax(predictions)
        word = loaded_tokenizer.index_word.get(pos, None)

        if word:
            seed_text += " " + word
            predicted_sentences.append(seed_text.capitalize())  # Capitalize the first letter
        else:
            return "Predicted word not found in the vocabulary."

    return '\n'.join(predicted_sentences)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    seed_text = request.form['seed_text']
    prediction = predict_next_word(seed_text)
    return render_template('index.html', seed_text=seed_text, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
