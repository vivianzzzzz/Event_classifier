import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
from transformers import get_linear_schedule_with_warmup
# import time

# start_time = time.time()
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


events_df = pd.read_csv("events.csv")
#events_df["type_id"] = events_df["type_id"].replace([3, 4, 5], 3) # Combining related category and see if the performance improves
events_df = events_df[events_df["type_id"] <= 3]  # Only keep the first 3 categories
texts = events_df["title_details"].astype(str).tolist()
labels = events_df["type_id"].astype(int) - 1  # Subtract 1 to adjust label values, make it starts with 0
labels = labels.to_list()



train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize input texts and convert to PyTorch tensors
train_encodings = tokenizer(
    train_texts, truncation=True, padding=True, max_length=64, return_tensors="pt"
)
val_encodings = tokenizer(
    val_texts, truncation=True, padding=True, max_length=64, return_tensors="pt"
)
train_labels = torch.tensor(train_labels)
val_labels = torch.tensor(val_labels)

train_dataset = TensorDataset(
    train_encodings["input_ids"], train_encodings["attention_mask"], train_labels
)
val_dataset = TensorDataset(
    val_encodings["input_ids"], val_encodings["attention_mask"], val_labels
)

# Create DataLoader for training and validation sets
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# # Set up optimizer and loss function
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
optimizer = AdamW(model.parameters(), lr=5e-5)
criterion = torch.nn.CrossEntropyLoss()

# # Training loop
num_epochs = 4
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    total_train_loss = 0
    model.train()
    losses = []  # m
    correct_predictions = 0  # m
    total_samples = 0
    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = (
            input_ids.to(device),
            attention_mask.to(device),
            labels.to(device),
        )
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = outputs.loss
        # Get predicted labels
        _, preds = torch.max(outputs.logits, dim=1)

        # Update total correct predictions and total samples
        correct_predictions += torch.sum(preds == labels).item()
        total_samples += labels.size(0)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
    train_accuracy = correct_predictions / total_samples
    print(
        f"Epoch for {epoch + 1}/{num_epochs}, Avg. Training Loss: {total_train_loss}, Training Accuracy: {train_accuracy * 100:.2f}%"
    )
    # Validation loop
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = (
                input_ids.to(device),
                attention_mask.to(device),
                labels.to(device),
            )

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()

            # Calculate accuracy
            _, preds = torch.max(outputs.logits, 1)
            val_total += labels.size(0)
            val_correct += (preds == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = val_correct / val_total

    print(
        f"Epoch {epoch + 1}/{num_epochs}, Avg. val Loss: {avg_val_loss:.4f}, val Accuracy: {val_accuracy:.2%}"
    )