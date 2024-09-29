import torch
import torch.optim as optim
from tqdm import tqdm


def train_model(model, data_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0  # Track loss for each batch
        progress_bar = tqdm(data_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", leave=False)

        for i, (inputs, targets) in enumerate(progress_bar):
            optimizer.zero_grad()
            output, _ = model(inputs)  # Model forward pass
            loss = criterion(output, targets)  # Calculate loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters

            running_loss += loss.item()  # Accumulate loss

            # Update progress bar with the average loss up to this batch
            avg_loss = running_loss / (i + 1)
            progress_bar.set_postfix({'Avg Loss': f'{avg_loss:.4f}'})

def prepare_optimizer_and_criterion(model):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return optimizer, criterion
