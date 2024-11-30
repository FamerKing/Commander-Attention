from model import EmotionClassifier
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset
import pandas as pd
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
import torch.nn as nn


class GoEmotionsDataset(Dataset):
    def __init__(self, filename):
        self.data = pd.read_csv(filename, sep='\t', header=None)
        self.text = self.data[0].tolist()
        self.labels = self.data[1].apply(lambda x: [int(y) for y in x.split(",")]).tolist()
        self.polarity = self.data[2].tolist()
        self.intensity = self.data[3].tolist()

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = self.text[idx]
        labels = torch.tensor(self.labels[idx], dtype=torch.long)
        polarity = torch.tensor(self.polarity[idx], dtype=torch.float)
        intensity = torch.tensor(self.intensity[idx], dtype=torch.float)

        return text, labels, polarity, intensity



def collate_fn(batch):
    texts, labels, polarity, intensity = zip(*batch)
    encoding = tokenizer(list(texts), truncation=True, padding=True, return_tensors='pt')
    return {
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask'],
        'token_type_ids': encoding['token_type_ids'],
        'labels': torch.stack(labels, dim=0),
        'polarity': torch.stack(polarity, dim=0).unsqueeze(-1),
        'intensity': torch.stack(intensity, dim=0).unsqueeze(-1)
    }


train_dataset = GoEmotionsDataset('D:/GOEMOTIONS/GoEmotions-pytorch-master/new_project/train.csv')  # Assuming the data file name
train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn)

model = EmotionClassifier()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
criterion = nn.BCEWithLogitsLoss()

for epoch in range(10):  # 10 epochs for example
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)
        polarity = batch['polarity'].to(device)
        intensity = batch['intensity'].to(device)

        outputs = model(input_ids, attention_mask, token_type_ids, polarity, intensity)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1} finished')
