import torch
from torchvision import models
from scripts.utils import get_data_loaders

def test_model(val_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the trained model
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)  # num_classes = number of classes in dataset
    model.load_state_dict(torch.load('models/resnet18.pth'))
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0

    # Evaluate the model
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy: {100 * correct / total}%')

if __name__ == "__main__":
    val_dir = 'data/val'
    batch_size = 32
    _, val_loader = get_data_loaders(None, val_dir, batch_size)

    test_model(val_loader)
