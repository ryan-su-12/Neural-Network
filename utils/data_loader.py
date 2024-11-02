import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class TextDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length=128):    
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(filepath, 'r') as f:
            self.text = f.readlines()

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.text[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )
        return encoding['input_ids'].squeeze(), encoding['attention_mask'].squeeze()

def get_data_loader(filepath, batch_size=32, max_length=128):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = TextDataset(filepath, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

