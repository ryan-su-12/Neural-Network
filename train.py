import torch
import torch.nn as nn
import torch.optim as optim
from models.transformer import Transformer
from utils.data_loader import get_data_loader
import config

model = Transformer(
    embed_size=config.EMBED_SIZE,
    num_layers=config.NUM_LAYERS,
    heads=config.HEADS,
    forward_expansion=config.FORWARD_EXPANSION,
    vocab_size=config.VOCAB_SIZE,
    max_length=config.MAX_LENGTH
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
train_loader = get_data_loader("data/sample_data.txt", config.BATCH_SIZE, config.MAX_LENGTH)

for epoch in range(config.EPOCHS):
    for batch_idx, (input_ids, attention_mask) in enumerate(train_loader):
        outputs = model(input_ids)
        loss = criterion(outputs.view(-1, config.VOCAB_SIZE), input_ids.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch}/{config.EPOCHS}], Step [{batch_idx}], Loss: {loss.item()}")


