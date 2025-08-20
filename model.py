import torch
import torch.nn as nn
import torch.nn.functional as F


class DeliminatorFunction(torch.autograd.Function):
    """
    A custom autograd function for the deliminator operation.
    This function acts as an identity operation during forward/backward pass,
    but will be exported as a custom ONNX op during ONNX export.
    """
    
    @staticmethod
    def forward(ctx, input):
        """
        Forward pass of the deliminator function.
        This is just an identity operation.
        """
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the deliminator function.
        This is just an identity operation.
        """
        return grad_output
    
    @staticmethod
    def symbolic(g, input):
        """
        Symbolic function for exporting deliminator to ONNX.
        This will create a custom ONNX op called 'Deliminator'.
        """
        return g.op("pytorch_annotation::Deliminator", input)


def deliminator(input):
    """
    Apply the deliminator operation to the input tensor.
    This function acts as an identity operation during PyTorch execution,
    but will be exported as a custom ONNX op during ONNX export.
    
    Args:
        input (torch.Tensor): Input tensor
        
    Returns:
        torch.Tensor: Output tensor (same as input)
    """
    return DeliminatorFunction.apply(input)











# Register the symbolic function with PyTorch
# torch.onnx.register_custom_op_symbolic('pytorch_annotation::deliminator', _deliminator_symbolic, 11)

class SimpleNN(nn.Module):
    """
    A simple neural network with linear and ReLU layers for image classification.
    """
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        """
        Initialize the neural network.
        
        Args:
            input_size (int): Size of the input features (default: 784 for 28x28 images)
            hidden_size (int): Number of neurons in the hidden layer (default: 128)
            num_classes (int): Number of output classes (default: 10 for digits 0-9)
        """
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
    def step(self, x):
        """
        Forward pass of the neural network.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        x = deliminator(x)
        # Flatten the input tensor
        x = x.view(x.size(0), -1)
        
        # If input size is not 784, we need to handle it differently
        if x.size(1) != 784:
            # For subsequent steps, we need to adjust the input to match fc1
            # We can either pad or truncate the input, or use a different approach
            # For simplicity, let's just use the first 784 elements or pad with zeros
            if x.size(1) < 784:
                # Pad with zeros
                padding = torch.zeros(x.size(0), 784 - x.size(1))
                x = torch.cat([x, padding], dim=1)
            else:
                # Truncate to 784 elements
                x = x[:, :784]
        
        # Apply linear and ReLU layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return deliminator(x)

    def forward(self, x):
        for _ in range(3):
            x = self.step(x)
        return x

def create_model(input_size=784, hidden_size=128, num_classes=10):
    """
    Create an instance of the SimpleNN model.
    
    Args:
        input_size (int): Size of the input features
        hidden_size (int): Number of neurons in the hidden layer
        num_classes (int): Number of output classes
        
    Returns:
        SimpleNN: An instance of the SimpleNN model
    """
    return SimpleNN(input_size, hidden_size, num_classes)

if __name__ == "__main__":
    # Example usage
    model = create_model()
    print(model)
    
    # Create a sample input tensor (batch_size=1, channels=1, height=28, width=28)
    sample_input = torch.randn(1, 1, 28, 28)
    
    # Forward pass
    output = model(sample_input)
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")