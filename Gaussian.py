import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import torch.nn as nn
from sklearn.metrics import accuracy_score

# Function to read multiple JSONL files and extract texts and labels
def read_multiple_jsonl_files(file_paths):
    texts = []
    labels = []

    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line)
                texts.append(data['text'])
                labels.append(data['label'])

    return texts, labels

# Define label to index mapping
label_to_index = {'llama': 0, 'human': 1, 'gptneo': 2, 'gpt3re': 3, 'gpt2': 4}
num_classes = len(label_to_index)

# Convert labels to indices
def convert_labels_to_indices(labels):
    return [label_to_index[label] for label in labels]

# Define custom dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.noise_dim = 128

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Generate random Gaussian noise
        noise = np.random.normal(0, 1, self.noise_dim)

        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length,
                                  return_tensors='pt')

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long),
            'noise': torch.tensor(noise, dtype=torch.float32)
        }

# Load BERT tokenizer
pretrained_model_name = '/root/NLP/google-bert/bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)

# Define maximum length
max_length = 512

# Read data from JSONL files
file_paths = [
    '/root/NLP/qxp/en_gpt2_lines.jsonl',
    '/root/NLP/qxp/en_gpt3_lines.jsonl',
    '/root/NLP/qxp/en_gptneo_lines.jsonl',
    '/root/NLP/qxp/en_human_lines.jsonl',
    '/root/NLP/qxp/en_llama_lines.jsonl'
]  # Replace with your actual file paths
texts, labels = read_multiple_jsonl_files(file_paths)

# Convert labels to indices
labels = convert_labels_to_indices(labels)

# Create dataset
dataset = TextDataset(texts, labels, tokenizer, max_length)

# Split dataset into train and test sets (e.g., 80% train, 20% test)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Define model class
class AIGTClassifier(nn.Module):
    def __init__(self, pretrained_model_name, num_classes):
        super(AIGTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        # self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(self.bert.config.hidden_size + 128, 128) 
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, input_ids, attention_mask, noise):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        # pooled_output = self.dropout(pooled_output)
        combined_output = torch.cat((pooled_output, noise), dim=1)
        
        intermediate_output = self.fc1(combined_output)
        intermediate_output = torch.relu(intermediate_output)
        logits = self.fc2(intermediate_output)
        return logits

# Define model
pretrained_model_name = '/root/NLP/google-bert/bert-base-cased'
model = AIGTClassifier(pretrained_model_name, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


import numpy as np

# Evaluation function
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            noise = batch['noise'].to(device)
        
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, noise=noise)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)

    print(f"Val Loss: {epoch_loss:.4f}, Val Accuracy: {epoch_acc:.4f}")
    return epoch_loss, epoch_acc

# Training function
def train_model(model, dataloader, val_dataloader, criterion, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            noise = batch['noise'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, noise=noise)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = accuracy_score(all_labels, all_preds)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}")
        val_loss, val_acc = evaluate_model(model, val_dataloader, criterion, device)
        
        yield epoch, val_loss, val_acc


num_epochs = 8
results = []
print(f"Trying learning rate: 2e-5 with 8 epochs...")
for epoch, val_loss, val_acc in train_model(model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs, device):
    result = {'epoch': epoch + 1, 'val_loss': val_loss, 'val_acc': val_acc}
    results.append(result)
print(results)