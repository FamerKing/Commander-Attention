import torch
from torch import nn
from transformers import XLNetModel, XLNetTokenizer

import torch
from torch import nn
from transformers import XLNetModel, XLNetTokenizer

class EmotionClassifier(nn.Module):
    def __init__(self, num_emotions=28, hidden_size=768):
        super(EmotionClassifier, self).__init__()
        self.xlnet = XLNetModel.from_pretrained('xlnet-base-cased')
        self.fcs = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(num_emotions)])
        self.sigmoid = nn.Sigmoid()
        self.attention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size + 2, 256),
                nn.Tanh(),
                nn.Linear(256, 1),
                nn.Softmax(dim=1)
            ) for _ in range(num_emotions)
        ])

        self.emotion_attention_weights = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Linear(256, num_emotions),
            nn.Softmax(dim=1)
        )

        self.emotion_combine = nn.Linear(num_emotions, num_emotions)
        self.inter_emotion_attention = nn.MultiheadAttention(embed_dim=num_emotions, num_heads=1)

    def forward(self, input_ids, attention_mask, token_type_ids, polarity, intensity, avg_emotion_polarity, avg_emotion_intensity):
        outputs = self.xlnet(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        sentence_output = outputs[0][:, 0]
        polarity = polarity.unsqueeze(1)
        intensity = intensity.unsqueeze(1)
        combined = torch.cat([sentence_output, polarity, intensity], dim=1)

        emotion_attention_weights = self.emotion_attention_weights(torch.cat([polarity, intensity], dim=1))

        attention_scores = [att(combined) for att in self.attention]
        weighted_outputs = [emotion_attention_weights[:, i:i+1] * (score * sentence_output) for i, score in enumerate(attention_scores)]
        emotions = torch.stack([self.sigmoid(fc(output)).squeeze(-1) for fc, output in zip(self.fcs, weighted_outputs)], dim=1)

        emotions_combined = self.emotion_combine(emotions)

        emotions, _ = self.inter_emotion_attention(emotions.unsqueeze(0), emotions.unsqueeze(0), emotions.unsqueeze(0))
        emotions = emotions.squeeze(0)

        emotions = emotions + avg_emotion_polarity + avg_emotion_intensity

        return emotions, attention_scores
