import pandas as pd
import torch
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import pickle

# Load the dataset
df = pd.read_csv("ipc_codes1.csv", encoding='ISO-8859-1')

# Drop missing values
df.dropna(inplace=True)

# Create mapping BEFORE label encoding
section_to_description = df.drop_duplicates(subset='section').set_index('section')['description'].to_dict()

# Encode the target variable (section)
label_encoder = LabelEncoder()
df['section'] = label_encoder.fit_transform(df['section'])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    df['description'], df['section'], test_size=0.2, random_state=42
)

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'nlpaueb/legal-bert-base-uncased', num_labels=len(label_encoder.classes_)
)

# Tokenization
def tokenize_data(texts):
    return tokenizer(
        texts.tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt"
    )

train_encodings = tokenize_data(X_train)
test_encodings = tokenize_data(X_test)

# Dataset class
class LegalDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = LegalDataset(train_encodings, y_train.values)
test_dataset = LegalDataset(test_encodings, y_test.values)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=60,
    per_device_train_batch_size=4,
    evaluation_strategy="epoch",
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Train the model
trainer.train()

# Evaluate
eval_result = trainer.evaluate()
print("Evaluation Results:", eval_result)

# Save model, tokenizer, and label encoder
output_dir = './results'
os.makedirs(output_dir, exist_ok=True)

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

label_encoder_path = os.path.join(output_dir, 'label_encoder.pkl')
with open(label_encoder_path, 'wb') as f:
    pickle.dump(label_encoder, f)
print(f"Model, tokenizer, and label encoder saved to '{output_dir}'")

# PREDICTION FUNCTION
def predict_section(complaint_description):
    inputs = tokenizer(
        complaint_description, return_tensors='pt', max_length=128, padding=True, truncation=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1)
    
    # Decode section number
    predicted_section = label_encoder.inverse_transform(predicted_class.numpy())[0]
    
    # Get description
    description = section_to_description.get(predicted_section, "Description not found.")
    
    print(f"The suitable section for the complaint '{complaint_description}' is: {predicted_section}")
    print(f"Description of this section: {description}")
    return predicted_section, description
