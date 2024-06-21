import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import torch.nn as nn
from sklearn.metrics import accuracy_score
import nltk
from collections import Counter


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


# Custom function to calculate word frequencies
def get_word_frequencies(text):
    tokens = nltk.word_tokenize(text.lower())
    return Counter(tokens)


# Custom function to calculate type-token ratio (TTR)
def calculate_ttr(text):
    tokens = nltk.word_tokenize(text.lower())
    return len(set(tokens)) / len(tokens)


# Define custom dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize text
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length,
                                  return_tensors='pt')

        # Calculate word frequencies and TTR
        word_freq = get_word_frequencies(text)
        ttr = calculate_ttr(text)

        # Convert word frequencies to tensor
        word_freq_tensor = torch.tensor([word_freq.get(token, 0) for token in self.tokenizer.vocab], dtype=torch.float)

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long),
            'word_freq': word_freq_tensor,
            'ttr': torch.tensor(ttr, dtype=torch.float)
        }


# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define maximum length
max_length = 128

# Read data from JSONL files
file_paths = [
    'qxp/en_gpt2_lines.jsonl',
    'qxp/en_gpt3_lines.jsonl',
    'qxp/en_gptneo_lines.jsonl',
    'qxp/en_human_lines.jsonl',
    'qxp/en_llama_lines.jsonl'
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
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size + tokenizer.vocab_size + 1, num_classes)

    def forward(self, input_ids, attention_mask, word_freq, ttr):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        # Concatenate BERT output with word frequencies and TTR
        combined_features = torch.cat((pooled_output, word_freq, ttr.unsqueeze(1)), dim=1)

        logits = self.fc(combined_features)
        return logits


# Define model
pretrained_model_name = 'bert-base-uncased'
model = AIGTClassifier(pretrained_model_name, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
num_epochs = 3

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# Training function
def train_model(model, dataloader, criterion, optimizer, scheduler, num_epochs, device):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            word_freq = batch['word_freq'].to(device)
            ttr = batch['ttr'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, word_freq=word_freq, ttr=ttr)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        scheduler.step()
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = accuracy_score(all_labels, all_preds)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")


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
            word_freq = batch['word_freq'].to(device)
            ttr = batch['ttr'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, word_freq=word_freq, ttr=ttr)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)

    print(f"Test Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")


# Train the model
train_model(model, train_dataloader, criterion, optimizer, scheduler, num_epochs, device)

# Evaluate the model
evaluate_model(model, test_dataloader, criterion, device)
