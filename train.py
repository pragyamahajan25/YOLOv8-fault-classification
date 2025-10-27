import torch
import torch.optim as optim
from model import initialize_model, train_model, evaluate_model
from dataset import create_train_val_loaders
from config import device, train_path, val_path
import torch.multiprocessing as mp

def main():
    # Set device (cuda or cpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create data loaders
    train_loader, val_loader = create_train_val_loaders(train_path, val_path)

    # Initialize model, loss, and optimizer
    model = initialize_model(num_classes=4)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Define a learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Train model
    train_model(model, train_loader, criterion, optimizer, num_epochs=20, scheduler=scheduler)

    # Evaluate model on validation set
    evaluate_model(model, val_loader, criterion)

    # Save trained model
    torch.save(model.state_dict(), "blur_classifier.pth")
    print("Model saved as 'blur_classifier.pth'.")


if __name__ == '__main__':
    # Set the start method for multiprocessing to avoid errors on Windows
    mp.set_start_method('spawn', force=True)
    
    # Call the main function
    main()