import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from model import create_model

def generate_sample_data(num_samples=1000, input_size=784, num_classes=10):
    """
    Generate sample data for training and testing.
    
    Args:
        num_samples (int): Number of samples to generate
        input_size (int): Size of each input sample
        num_classes (int): Number of classes
        
    Returns:
        tuple: (features, labels) as torch tensors
    """
    # Generate random features
    features = torch.randn(num_samples, 1, 28, 28)  # Assuming 28x28 images
    
    # Generate random labels
    labels = torch.randint(0, num_classes, (num_samples,))
    
    return features, labels

def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    """
    Train the model.
    
    Args:
        model (nn.Module): The neural network model
        train_loader (DataLoader): DataLoader for training data
        criterion: Loss function
        optimizer: Optimization algorithm
        num_epochs (int): Number of training epochs
        
    Returns:
        list: Training losses for each epoch
    """
    model.train()
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
    
    return losses

def evaluate_model(model, test_loader):
    """
    Evaluate the model on test data.
    
    Args:
        model (nn.Module): The neural network model
        test_loader (DataLoader): DataLoader for test data
        
    Returns:
        float: Accuracy of the model
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    return accuracy

def main():
    # Hyperparameters
    input_size = 784  # 28x28 images
    hidden_size = 128
    num_classes = 10
    batch_size = 32
    num_epochs = 5
    learning_rate = 0.001
    
    # Generate sample data
    print("Generating sample data...")
    train_features, train_labels = generate_sample_data(1000, input_size, num_classes)
    test_features, test_labels = generate_sample_data(200, input_size, num_classes)
    
    # Create data loaders
    train_dataset = TensorDataset(train_features, train_labels)
    test_dataset = TensorDataset(test_features, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    print("Creating model...")
    model = create_model(input_size, hidden_size, num_classes)
    print(model)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    print("Training the model...")
    losses = train_model(model, train_loader, criterion, optimizer, num_epochs)
    
    # Evaluate the model
    print("Evaluating the model...")
    accuracy = evaluate_model(model, test_loader)
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    print("Training loss plot saved as 'training_loss.png'")
    
    # Save the model
    torch.save(model.state_dict(), 'simple_nn_model.pth')
    print("Model saved as 'simple_nn_model.pth'")

if __name__ == "__main__":
    main()