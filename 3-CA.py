import torch
from torch import nn
from transformers import XLNetModel, XLNetTokenizer

class EmotionClassifier(nn.Module):
    def __init__(self, num_emotions=28, hidden_size=768):
        super(EmotionClassifier, self).__init__()
        # Initialize XLNet as the base model
        self.xlnet = XLNetModel.from_pretrained('xlnet-base-cased')
        # Create a fully connected layer for each emotion category
        self.fcs = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(num_emotions)])
        # Define sigmoid function for converting outputs into probabilities
        self.sigmoid = nn.Sigmoid()
        # Define attention mechanism for each emotion category
        self.attention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size + 2, 256),
                nn.Tanh(),
                nn.Linear(256, 1),
                nn.Softmax(dim=1)
            ) for _ in range(num_emotions)
        ])
        # Define a layer to compute attention weights based on polarity and intensity
        self.emotion_attention_weights = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Linear(256, num_emotions),
            nn.Softmax(dim=1)
        )
        # Linear transformation layer for emotions
        self.emotion_combine = nn.Linear(num_emotions, num_emotions)
        # Multi-head attention mechanism for capturing relationships between emotions
        self.inter_emotion_attention = nn.MultiheadAttention(embed_dim=num_emotions, num_heads=1)

    def forward(self, input_ids, attention_mask, token_type_ids, polarity, intensity, avg_emotion_polarity, avg_emotion_intensity):
        # Forward pass through XLNet
        outputs = self.xlnet(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # Extract sentence-level representation
        sentence_output = outputs[0][:, 0]
        # Expand dimensions for polarity and intensity
        polarity = polarity.unsqueeze(1)
        intensity = intensity.unsqueeze(1)
        # Concatenate sentence representation with polarity and intensity
        combined = torch.cat([sentence_output, polarity, intensity], dim=1)

        # Compute attention weights for each emotion
        emotion_attention_weights = self.emotion_attention_weights(torch.cat([polarity, intensity], dim=1))

        # Apply attention mechanisms for each emotion
        attention_scores = [att(combined) for att in self.attention]
        weighted_outputs = [emotion_attention_weights[:, i:i + 1] * (score * sentence_output) for i, score in enumerate(attention_scores)]
        # Use sigmoid to generate probability outputs for each emotion
        emotions = torch.stack([self.sigmoid(fc(output)).squeeze(-1) for fc, output in zip(self.fcs, weighted_outputs)], dim=1)

        # Apply linear transformation to the emotion outputs
        emotions_combined = self.emotion_combine(emotions)

        # Pass the emotions through multi-head attention mechanism
        emotions, _ = self.inter_emotion_attention(emotions.unsqueeze(0), emotions.unsqueeze(0), emotions.unsqueeze(0))
        emotions = emotions.squeeze(0)

        # Add average emotion polarity and intensity as additional features
        emotions = emotions + avg_emotion_polarity + avg_emotion_intensity

        return emotions, attention_scores  # Ensure two values are returned

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import XLNetTokenizer
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

def load_saved_model(model, model_path):
    # Load pre-trained model if it exists
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Pre-trained model loaded:", model_path)
    else:
        print("No pre-trained model found. Starting training from scratch.")

class GoEmotionsDataset(Dataset):
    def __init__(self, filename, tokenizer, num_emotions=28):
        # Load dataset (tab-separated values)
        self.data = pd.read_csv(filename, sep='\t', header=None)
        self.tokenizer = tokenizer
        self.num_emotions = num_emotions

        # Create lists to store polarity and intensity for each emotion category
        self.emotion_polarity = [[] for _ in range(num_emotions)]
        self.emotion_intensity = [[] for _ in range(num_emotions)]

        # Iterate through the dataset to calculate polarity and intensity for each emotion category
        for idx in range(len(self.data)):
            labels = [int(label) for label in self.data.iloc[idx, 1].split(",")]
            polarity = float(self.data.iloc[idx, 2])  # Convert polarity to float
            intensity = float(self.data.iloc[idx, 3])  # Convert intensity to float

            for label in labels:
                self.emotion_polarity[label].append(polarity)
                self.emotion_intensity[label].append(intensity)

        # Compute average polarity and intensity for each emotion category
        self.avg_emotion_polarity = [torch.mean(torch.tensor(polarity_list)) for polarity_list in self.emotion_polarity]
        self.avg_emotion_intensity = [torch.mean(torch.tensor(intensity_list)) for intensity_list in self.emotion_intensity]

    def __len__(self):
        # Return dataset size
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve a sample from the dataset
        text = self.data.iloc[idx, 0]
        labels = torch.zeros(self.num_emotions)
        for label in self.data.iloc[idx, 1].split(","):
            labels[int(label)] = 1

        labels = labels.float()

        # Retrieve polarity and intensity
        polarity = torch.tensor(self.data.iloc[idx, 2], dtype=torch.float)
        intensity = torch.tensor(self.data.iloc[idx, 3], dtype=torch.float)

        # Encode text using the tokenizer
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
        encoding = {key: tensor.squeeze(0) for key, tensor in encoding.items()}

        # Return a dictionary with inputs, labels, polarity, and intensity
        return {**encoding, 'labels': labels, 'polarity': polarity, 'intensity': intensity}

# Compute accuracy for model predictions
def compute_accuracy(predictions, labels):
    # Convert model predictions to binary (threshold at 0.5)
    preds = torch.sigmoid(predictions) > 0.5
    # Calculate the number of correct predictions
    correct = (preds == labels).sum().item()
    # Calculate the total number of predictions
    total = labels.numel()
    # Compute accuracy
    accuracy = correct / total
    return accuracy

def visualize_attention(attention_scores, input_ids, tokenizer, epoch, batch_idx, output_dir='attention_visualizations'):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Check shape of attention scores
    print(f"Attention Scores Shape: {attention_scores[0].shape}")
    print(f"Attention Scores: {attention_scores[0]}")

    # Ensure attention_scores is a 2D array
    attention = attention_scores[0].squeeze().detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].detach().cpu().numpy())

    # Plot and save attention visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention, xticklabels=tokens, yticklabels=["Attention"], cmap="YlGnBu")
    plt.xlabel("Tokens")
    plt.ylabel("Attention Weights")
    plt.title(f"Attention Weights Visualization - Epoch {epoch+1} Batch {batch_idx+1}")

    # Save the visualization to a file
    file_path = os.path.join(output_dir, f'epoch_{epoch+1}_batch_{batch_idx+1}.png')
    plt.savefig(file_path)
    plt.close()
    print(f"Saved attention visualization to {file_path}")

from torch.utils.data import DataLoader, random_split
from transformers import AdamW
from tqdm import tqdm
from sklearn.metrics import classification_report
import numpy as np

def main():
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    dataset = GoEmotionsDataset('D:/GOEMOTIONS/GoEmotions-pytorch-master/new_project/all.tsv', tokenizer)

    # Split dataset into training and testing sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print("GPU Model:", torch.cuda.get_device_name(0))
        print("Using GPU for acceleration")
    else:
        print("No GPU detected. Using CPU.")

    model = EmotionClassifier().to(device)
    print(model)

    model_path = 'D:/GOEMOTIONS/GoEmotions-pytorch-master/new_project/model/best_model.pth'
    load_saved_model(model, model_path)

    optimizer = AdamW(model.parameters(), lr=1e-5)

    for epoch in range(20):
        total_loss = 0
        total_accuracy = 0
        model.train()
        with tqdm(train_loader, unit='batch', desc=f'Epoch {epoch + 1}/{20}') as t:
            for batch_idx, batch in enumerate(t):
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                polarity = batch['polarity'].to(device)
                intensity = batch['intensity'].to(device)

                avg_emotion_polarity = torch.tensor(train_dataset.dataset.avg_emotion_polarity, dtype=torch.float).to(device)
                avg_emotion_intensity = torch.tensor(train_dataset.dataset.avg_emotion_intensity, dtype=torch.float).to(device)

                predictions, attention_scores = model(input_ids, attention_mask, None, polarity, intensity, avg_emotion_polarity, avg_emotion_intensity)

                labels = labels.float()

                loss = nn.BCELoss(reduction='mean')(torch.sigmoid(predictions), labels)
                accuracy = compute_accuracy(predictions, labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_accuracy += accuracy

                t.set_postfix(train_loss=total_loss / len(train_loader), train_accuracy=total_accuracy / len(train_loader))
                t.update()

            model.eval()
            test_loss = 0
            test_accuracy = 0
            all_labels = []
            all_predictions = []
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(test_loader, unit='batch', desc='Evaluating')):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    polarity = batch['polarity'].to(device)
                    intensity = batch['intensity'].to(device)

                    avg_emotion_polarity = torch.tensor(test_dataset.dataset.avg_emotion_polarity, dtype=torch.float).to(device)
                    avg_emotion_intensity = torch.tensor(test_dataset.dataset.avg_emotion_intensity, dtype=torch.float).to(device)

                    predictions, attention_scores = model(input_ids, attention_mask, None, polarity, intensity, avg_emotion_polarity, avg_emotion_intensity)

                    labels = labels.float()

                    loss = nn.BCELoss(reduction='mean')(torch.sigmoid(predictions), labels)
                    accuracy = compute_accuracy(predictions, labels)

                    test_loss += loss.item()
                    test_accuracy += accuracy

                    all_labels.extend(labels.detach().cpu().numpy())
                    all_predictions.extend(torch.sigmoid(predictions).detach().cpu().numpy())

                    # Visualize and save attention weights
                    visualize_attention(attention_scores, input_ids, tokenizer, epoch, batch_idx)

                print("Test Loss:", test_loss / len(test_loader))
                print("Test Accuracy:", test_accuracy / len(test_loader))
                print(classification_report(all_labels, np.array(all_predictions) > 0.5, zero_division=0))

            best_model_path = 'D:/GOEMOTIONS/GoEmotions-pytorch-master/new_project/model/best_model.pth'
            torch.save(model.state_dict(), best_model_path)
            print("Best model saved to:", best_model_path)

    print(f"Training complete. Average loss: {total_loss / len(train_loader)}, Average accuracy: {total_accuracy / len(train_loader)}")

if __name__ == '__main__':
    main()
