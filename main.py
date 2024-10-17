import argparse
from scripts.train import train_model
from scripts.test import test_model
from scripts.utils import get_data_loaders

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Classifier")
    parser.add_argument('--mode', type=str, default='train', help='train or test the model')
    args = parser.parse_args()

    train_dir = 'data/train'
    val_dir = 'data/val'
    batch_size = 32

    train_loader, val_loader = get_data_loaders(train_dir, val_dir, batch_size)

    if args.mode == 'train':
        train_model(train_loader, val_loader)
    elif args.mode == 'test':
        test_model(val_loader)
    else:
        print("Invalid mode. Choose either 'train' or 'test'.")
