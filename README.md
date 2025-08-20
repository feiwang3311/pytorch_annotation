# Simple PyTorch Neural Network for Image Classification

This project implements a simple neural network using PyTorch for image classification. The model uses only linear and ReLU layers, making it lightweight and easy to understand.

## Project Structure

- `model.py`: Contains the neural network model definition
- `train.py`: Training script with sample data generation
- `trace_and_visualize.py`: Script to trace the model to ONNX and visualize the forward path
- `requirements.txt`: Lists the required Python packages

## Requirements

To run this project, you need to have Python 3.6+ and the following packages installed:

- torch (1.9.0 or higher)
- torchvision (0.10.0 or higher)
- matplotlib (3.3.0 or higher)
- numpy (1.21.0 or higher)
- onnx (1.10.0 or higher)
- onnxruntime (1.8.0 or higher)

You can install these packages using pip:

```bash
pip3 install -r requirements.txt
```

## Model Architecture

The neural network consists of three linear layers with ReLU activation functions:

1. Input layer: Flattens the input image (28x28 pixels)
2. Hidden layer 1: Linear transformation + ReLU activation (784 -> 128)
3. Hidden layer 2: Linear transformation + ReLU activation (128 -> 128)
4. Output layer: Linear transformation (128 -> 10 classes)

## Usage

### Training the Model

To train the model with sample data, run:

```bash
python3 train.py
```

This will:
1. Generate sample training and test data
2. Create an instance of the neural network
3. Train the model for 5 epochs
4. Evaluate the model on test data
5. Save the trained model to `simple_nn_model.pth`
6. Generate a plot of training loss and save it as `training_loss.png`

Expected output:
```
Generating sample data...
Creating model...
SimpleNN(
  (fc1): Linear(in_features=784, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=10, bias=True)
)
Training the model...
Epoch [1/5], Step [0/32], Loss: 2.3001
...
Epoch [5/5], Average Loss: 0.0522
Evaluating the model...
Accuracy: 6.00%
Training loss plot saved as 'training_loss.png'
Model saved as 'simple_nn_model.pth'
```

### Tracing the Model and Generating ONNX

To trace the PyTorch model and generate an ONNX file with visualization of the forward path, run:

```bash
python3 trace_and_visualize.py
```

This will:
1. Create an instance of the neural network
2. Trace the model using `torch.jit.trace`
3. Export the traced model to ONNX format (`simple_nn_model.onnx`)
4. Visualize the ONNX model structure
5. Create a visualization of the model's forward path (`model_forward_path.png`)
6. Verify that the ONNX model produces the same output as the PyTorch model

Expected output:
```
Model created successfully
Sample input shape: torch.Size([1, 1, 28, 28])
Model traced and exported to simple_nn_model.onnx
ONNX model is valid
Model IR version: 6
Model opset: [version: 11]
Number of nodes: 12
Model nodes:
  Node 0: Shape - /Shape
  ...
  Node 11: Gemm - /fc3/Gemm
Model graph visualization saved to model_forward_path.png
ONNX model output shape: (1, 10)
PyTorch model output shape: torch.Size([1, 10])
Max difference between PyTorch and ONNX outputs: 5.960464477539063e-08
```

### Using the Model

To use the trained model in your own code:

```python
import torch
from model import create_model

# Create the model
model = create_model()

# Load the trained weights (if available)
# model.load_state_dict(torch.load('simple_nn_model.pth'))

# Create sample input data (batch_size=1, channels=1, height=28, width=28)
sample_input = torch.randn(1, 1, 28, 28)

# Forward pass
output = model(sample_input)
print(f"Output shape: {output.shape}")
```

## Customization

You can customize the model by modifying the parameters in `create_model()`:

```python
# Create a model with custom parameters
model = create_model(
    input_size=784,    # Size of input features
    hidden_size=256,   # Number of neurons in hidden layers
    num_classes=10     # Number of output classes
)
```

## Results

After training, the script will display the accuracy of the model on the test data and save a plot of the training loss over epochs.

## License

This project is released under the MIT License.