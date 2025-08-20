import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx

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
        
    def forward(self, x):
        """
        Forward pass of the neural network.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Flatten the input tensor
        x = x.view(x.size(0), -1)
        
        # Apply linear and ReLU layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
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

# Custom ONNX symbolic functions for adding metadata
def add_metadata_to_onnx(onnx_model):
    """
    Add metadata to the ONNX model nodes using built-in fields and custom attributes.
    
    Args:
        onnx_model: The ONNX model
        
    Returns:
        The ONNX model with metadata added to nodes
    """
    # Add metadata to nodes using built-in fields and custom approach
    for i, node in enumerate(onnx_model.graph.node):
        # Add layer name as metadata
        if node.op_type == "Gemm":
            # Determine which layer this is based on the weights
            if "fc1" in node.input[1]:
                layer_name = "fc1"
                layer_description = "First_fully_connected_layer_784_to_128"
            elif "fc2" in node.input[1]:
                layer_name = "fc2"
                layer_description = "Second_fully_connected_layer_128_to_128"
            elif "fc3" in node.input[1]:
                layer_name = "fc3"
                layer_description = "Output_layer_128_to_10"
            else:
                layer_name = f"linear_{i}"
                layer_description = "Linear_layer"
                
            # Add metadata using built-in fields
            node.name = layer_name
            node.doc_string = layer_description.replace("_", " ")
        elif node.op_type == "Relu":
            # Add metadata for ReLU layers
            layer_name = f"relu_{i//2}"  # Every second node is ReLU
            layer_description = f"ReLU_activation_{i//2 + 1}"
            
            # Add metadata using built-in fields
            node.name = layer_name
            node.doc_string = layer_description.replace("_", " ")
    
    return onnx_model

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