import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import pandas as pd
import random
from sklearn.metrics import accuracy_score, f1_score
from googletrans import Translator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

translator = Translator()

# =========================
# 1.UDA
# 使用 back_translate() 进行回译增强（中文 → 英文 → 中文）
# 生成语义相近但表达不同的文本，用于提升模型鲁棒性
# =========================
def back_translate(text):
    try:
        en = translator.translate(text, src='zh-cn', dest='en').text
        zh = translator.translate(en, src='en', dest='zh-cn').text
        return zh
    except:
        return text


# =========================
# 2. Dataset
# 读取 CSV 数据（text + label）
# 对原文本和增强文本分别进行 BERT 分词编码
# =========================
class TextDataset(Dataset):
    def __init__(self, file, tokenizer, max_len=128):
        self.data = pd.read_csv(file)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def encode(self, text):
        return self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['text']
        label = self.data.iloc[idx]['label']

        aug_text = back_translate(text)

        enc = self.encode(text)
        enc_aug = self.encode(aug_text)

        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'input_ids_aug': enc_aug['input_ids'].squeeze(0),
            'attention_mask_aug': enc_aug['attention_mask'].squeeze(0),
            'label': torch.tensor(label + 1)
        }


# =========================
# 3. 模型-UDA-BERT-CNN
# BERT：提取上下文语义表示
# CNN：多尺度卷积（3/4/5）捕获局部特征
# =========================
class UDA_BERT_CNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        hidden_size = self.bert.config.hidden_size

        self.convs = nn.ModuleList([
            nn.Conv2d(1, 256, (k, hidden_size))
            for k in [3, 4, 5]
        ])

        self.dropout = nn.Dropout(0.3)  # 强化扰动
        self.fc = nn.Linear(256 * 3, num_classes)

    def conv_pool(self, x, conv):
        x = conv(x)
        x = torch.relu(x.squeeze(3))
        x = torch.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids,
                        attention_mask=attention_mask)

        x = out.last_hidden_state.unsqueeze(1)

        x = torch.cat([self.conv_pool(x, conv) for conv in self.convs], dim=1)

        x = self.dropout(x)  # Dropout扰动
        return self.fc(x)


# =========================
# 4. UDA Loss：让模型对原文本和增强文本输出一致
# 对高置信度样本（>0.8）加约束；使用 KL散度衡量两者分布差异
# 总损失 = 监督损失（交叉熵） + UDA一致性损失
# =========================
def uda_loss(logits, logits_aug, threshold=0.8):
    probs = F.softmax(logits, dim=-1)
    max_probs, _ = torch.max(probs, dim=-1)

    mask = (max_probs > threshold).float()

    p = probs.detach()
    q = F.log_softmax(logits_aug, dim=-1)

    kl = F.kl_div(q, p, reduction='none').sum(dim=1)

    loss = (kl * mask).mean()
    return loss


# =========================
# 5. 评估
# =========================
def evaluate(model, loader):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            label = batch['label'].to(device)

            logits = model(input_ids, mask)
            pred = torch.argmax(logits, dim=1)

            preds.extend(pred.cpu().numpy())
            labels.extend(label.cpu().numpy())

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    return acc, f1


# =========================
# 6. 训练
# =========================
def train():
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    train_dataset = TextDataset("data/train.csv", tokenizer)
    val_dataset = TextDataset("data/val.csv", tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = UDA_BERT_CNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    best_f1 = 0
    patience = 3
    trigger = 0

    for epoch in range(10):
        model.train()
        total_loss = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            input_ids_aug = batch['input_ids_aug'].to(device)
            mask_aug = batch['attention_mask_aug'].to(device)

            logits = model(input_ids, mask)
            logits_aug = model(input_ids_aug, mask_aug)

            loss_sup = F.cross_entropy(logits, labels)
            loss_uda = uda_loss(logits, logits_aug)

            loss = loss_sup + 1.0 * loss_uda

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        acc, f1 = evaluate(model, val_loader)

        print(f"\nEpoch {epoch+1}")
        print(f"Loss: {total_loss:.4f}")
        print(f"Val Acc: {acc:.4f}, F1: {f1:.4f}")

        # Early stopping
        if f1 > best_f1:
            best_f1 = f1
            trigger = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            trigger += 1
            if trigger >= patience:
                print("Early stopping")
                break


if __name__ == "__main__":
    train()
