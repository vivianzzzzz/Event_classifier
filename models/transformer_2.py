import torch
import os
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertModel,get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from torch.nn.utils import clip_grad_norm_
# import time

# start_time = time.time()
num_labels=5
num_epochs = 10 
#os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("mps")

events_df = pd.read_csv("events.csv")
#print(events_df.head(10)["type_id"])
events_df = events_df.sample(frac = 1)
#print(events_df.head(10)["type_id"])
#events_df["type_id"] = events_df["type_id"].replace([3, 4, 5], 3) # Combining related category and see if the performance improves
events_df = events_df[events_df["type_id"] <= 2]  # Only keep the first 3 categories
texts = events_df["title_details"].astype(str).tolist()
labels = events_df["type_id"].astype(int) - 1  # Subtract 1 to adjust label values, make it starts with 0
labels = labels.to_list()


train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",do_lower_case=True)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", output_attentions=False, output_hidden_states=False, num_labels=2)
#model = BertModel.from_pretrained("bert-base-uncased")

# Tokenize input texts and convert to PyTorch tensors
train_encodings = tokenizer(
    train_texts, truncation=True, padding=True, max_length=200, return_tensors="pt"
)
'''
train_texts: This is a list or iterable containing the training texts that you want to encode.

truncation=True: When set to True, this option truncates sequences to the specified max_length if they exceed it. Truncation is relevant when dealing with sequences that are longer than the specified maximum length.

padding=True: When set to True, this option pads sequences to the specified max_length if they are shorter than it. Padding is used to make all input sequences in a batch have the same length, which is important when training neural networks.

max_length=64: This is the maximum length of the sequences after truncation or padding. Any sequence longer than this length will be truncated, and any sequence shorter than this length will be padded.

return_tensors="pt": This option specifies the format of the output. In this case, it's set to "pt," indicating that the output should be in PyTorch tensors. Other options could include "tf" for TensorFlow tensors or "np" for NumPy arrays.
'''

val_encodings = tokenizer(
    val_texts, truncation=True, padding=True, max_length=200, return_tensors="pt"
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
#optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

optimizer = AdamW(model.parameters(), lr=2e-6)
scheduler = get_linear_schedule_with_warmup(optimizer, 
             num_warmup_steps=0,
            num_training_steps=len(train_loader)*num_epochs )

#criterion = torch.nn.CrossEntropyLoss()
#criterion =  torch.nn.BCEWithLogitsLoss() ##torch.nn.BCELoss()

#classifier = torch.nn.Linear(model.config.hidden_size, 5)

# # Training loop
 


model=model.to(device)

train_loss_per_epoch = []
val_loss_per_epoch = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = (
            input_ids.to(device),
            attention_mask.to(device),
            labels.to(device),
        )

        #optimizer.zero_grad()
        #outputs = model(input_ids, attention_mask=attention_mask)
        
        output = model(input_ids = input_ids, attention_mask=attention_mask, labels= labels)
        loss = output.loss
        train_loss += loss.item()

        model.zero_grad()
        loss.backward()
        del loss

        clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        #outputs = classifier(outputs.pooler_output)
        #outputs = torch.sigmoid(outputs)
        #print(outputs)
        #loss = 0
        #if labels is not None:
        #    loss = criterion(outputs, labels)
        ##=====
        #logits = outputs.logits  # logits before softmax
        #print(logits)
        #loss = criterion(logits, labels)
        #loss = criterion(logits.view(-1, num_labels), labels.view(-1))
        #optimizer.zero_grad()
        #loss.backward()
        #optimizer.step()
    # Validation loop
    train_loss_per_epoch.append(train_loss / (len(train_loader))) 
    #print(f'Epoch: {epoch}, Loss:  {loss.item()}')
    
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    valid_loss = 0
    valid_pred = []    

    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = (
                input_ids.to(device),
                attention_mask.to(device),
                labels.to(device),
            )
            '''
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # logits before softmax
            loss = criterion(logits.view(-1, num_labels), labels.view(-1)) #criterion(logits, labels)
            val_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            '''
            output = model(input_ids = input_ids, attention_mask=attention_mask, labels= labels)

            loss = output.loss
            valid_loss += loss.item()
            val_loss += loss.item()
            _, predicted = torch.max(output.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            #valid_pred.append(np.argmax(output.logits.cpu().detach().numpy(),axis=-1))
            
    #val_loss_per_epoch.append(valid_loss / (step_num_e + 1))
    #valid_pred = np.concatenate(valid_pred)
            
    avg_val_loss = val_loss / len(val_loader)
    accuracy = correct / total

    print(
        f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2%}"
        )


# end_time = time.time()
# execution_time = end_time - start_time
# print("Execution time:", execution_time, "seconds")
