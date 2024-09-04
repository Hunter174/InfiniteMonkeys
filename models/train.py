import torch
import torch.optim as optim
from tqdm import tqdm


def train_model(model, data_loader, criterion, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        for inputs, targets in tqdm(data_loader):
            optimizer.zero_grad()
            output, _ = model(inputs)  # Assuming output is (batch_size, vocab_size)

            # Reshape output for CrossEntropyLoss
            loss = criterion(output, targets)

            loss.backward()
            optimizer.step()


def prepare_optimizer_and_criterion(model):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return optimizer, criterion
