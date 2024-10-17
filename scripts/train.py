import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from scripts.utils import get_data_loaders

def train_model(train_loader, val_loader, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load pre-trained model and modify the final layer
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)  # num_classes = number of classes in dataset
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}')

    # Save model checkpoint
    torch.save(model.state_dict(), 'models/resnet18.pth')
    print('Model saved!')

if __name__ == "__main__":
    # Set paths and batch size
    train_dir = 'data/train'
    val_dir = 'data/val'
    batch_size = 32

    # Get dataloaders
    train_loader, val_loader = get_data_loaders(train_dir, val_dir, batch_size)

    # Train the model
    train_model(train_loader, val_loader)
