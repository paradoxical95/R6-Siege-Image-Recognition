import argparse
from train import train_model
from test import predict_image
import torch
import json
from models.resnet_model import initialize_model

def main():
    parser = argparse.ArgumentParser(description="R6 Siege Image Recognition Tool")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'], help="Choose 'train' to train the model, 'test' to test with an image.")
    parser.add_argument('--image', type=str, help="Path to the image for testing")
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # Train the model
        train_model()
    elif args.mode == 'test':
        # Ensure image path is provided
        if args.image is None:
            print("Please provide the image path using --image when testing.")
            return
        
        # Load the class names
        class_names = json.load(open('class_names.json'))

        # Initialize and load the trained model
        model = initialize_model(num_classes=len(class_names))
        model.load_state_dict(torch.load('outputs/model.pth'))
        model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.eval()

        # Predict the class of the input image
        predicted_class = predict_image(args.image, model)
        print(f"Predicted class: {class_names[str(predicted_class)]}")

if __name__ == "__main__":
    main()
