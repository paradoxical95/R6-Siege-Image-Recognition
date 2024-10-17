import torch
from torchvision import transforms
from PIL import Image
import json
from models.resnet_model import initialize_model

# Define the same transformations used for validation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the trained model
model = initialize_model(num_classes=5)  # Replace with your actual number of classes
model.load_state_dict(torch.load('outputs/model.pth'))
model.eval()

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Function to predict the class of a given image
def predict_image(image_path, model):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    return predicted.item()

# Load class names
class_names = json.load(open('class_names.json'))

# Test with a sample image
image_path = 'path/to/test_image.jpg'
predicted_class = predict_image(image_path, model)
print(f"Predicted class: {class_names[predicted_class]}")
