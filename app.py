from flask import Flask, render_template, request
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pickle
import json
import os
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Paths for the saved model and files
MODEL_PATH = './results'
TOKENIZER_PATH = './results'
LABEL_ENCODER_PATH = './results/label_encoder.pkl'
CACHE_FILE = 'prediction_cache.json'
CSV_PATH = "ipc_codes1.csv"  # Update this if your CSV file is located elsewhere

# Load the trained model
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)

# Load the label encoder
with open(LABEL_ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)

# Load or initialize prediction cache
if os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, 'r') as f:
            prediction_cache = json.load(f)
    except json.JSONDecodeError:
        print("Cache file is corrupted or empty. Initializing a new cache.")
        prediction_cache = {}
else:
    prediction_cache = {}

# Load IPC code-to-description mapping
section_desc_df = pd.read_csv(CSV_PATH)
section_to_description = section_desc_df.drop_duplicates(subset='section').set_index('section')['description'].to_dict()

# Function to predict IPC section and description
def predict_section(complaint_description):
    # Check cache first
    if complaint_description in prediction_cache:
        print("Cache hit!")
        cached_section = prediction_cache[complaint_description]
        return cached_section, section_to_description.get(cached_section, "Description not found.")

    # Tokenize and predict
    inputs = tokenizer(
        complaint_description, return_tensors='pt', max_length=128, padding=True, truncation=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1)

    predicted_section = label_encoder.inverse_transform(predicted_class.numpy())[0]
    description = section_to_description.get(predicted_section, "Description not found.")

    # Cache result
    prediction_cache[complaint_description] = predicted_section
    with open(CACHE_FILE, 'w') as f:
        json.dump(prediction_cache, f)

    return predicted_section, description

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        complaint = request.form['complaint']
        if complaint:
            predicted_section, section_description = predict_section(complaint)
            return render_template('index.html', complaint=complaint, result=predicted_section, description=section_description)
        #return render_template('index.html', complaint=complaint, result=predicted_section, description=ipc_description)

        else:
            return render_template('index.html', error="Please enter a valid complaint.")
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
