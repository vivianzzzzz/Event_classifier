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
# import time

# start_time = time.time()


events_df = pd.read_csv("events.csv")
#events_df["type_id"] = events_df["type_id"].replace([3, 4, 5], 3) # Combining related category and see if the performance improves
#events_df = events_df[events_df["type_id"] <= 3]  # Only keep the first 3 categories
texts = events_df["title_details"].astype(str).tolist()
labels = events_df["type_id"].astype(int) - 1  # Subtract 1 to adjust label values, make it starts with 0
labels = labels.to_list()


train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

# Tokenize input texts and convert to PyTorch tensors
train_encodings = tokenizer(
    train_texts, truncation=True, padding=True, max_length=64, return_tensors="pt"
)
'''
train_texts: This is a list or iterable containing the training texts that you want to encode.

truncation=True: When set to True, this option truncates sequences to the specified max_length if they exceed it. Truncation is relevant when dealing with sequences that are longer than the specified maximum length.

padding=True: When set to True, this option pads sequences to the specified max_length if they are shorter than it. Padding is used to make all input sequences in a batch have the same length, which is important when training neural networks.

max_length=64: This is the maximum length of the sequences after truncation or padding. Any sequence longer than this length will be truncated, and any sequence shorter than this length will be padded.

return_tensors="pt": This option specifies the format of the output. In this case, it's set to "pt," indicating that the output should be in PyTorch tensors. Other options could include "tf" for TensorFlow tensors or "np" for NumPy arrays.
'''

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
train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=20, shuffle=False)

# # Set up optimizer and loss function
optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
criterion = torch.nn.CrossEntropyLoss()

# # Training loop
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = (
            input_ids.to(device),
            attention_mask.to(device),
            labels.to(device),
        )

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # logits before softmax
        loss = criterion(logits, labels)
        #loss = outputs.loss
        loss.backward()
        optimizer.step()
    # Validation loop
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = (
                input_ids.to(device),
                attention_mask.to(device),
                labels.to(device),
            )
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # logits before softmax
            loss = criterion(logits, labels)
            val_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    accuracy = correct / total

    print(
        f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2%}"
    )

# end_time = time.time()
# execution_time = end_time - start_time
# print("Execution time:", execution_time, "seconds")