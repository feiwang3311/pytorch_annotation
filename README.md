# Simple PyTorch Neural Network for Image Classification

This project implements a simple neural network using PyTorch for image classification. The model uses only linear and ReLU layers, making it lightweight and easy to understand.

## Project Structure

- `model.py`: Contains the neural network model definition
- `train.py`: Training script with sample data generation
- `requirements.txt`: Lists the required Python packages

## Requirements

To run this project, you need to have Python 3.6+ and the following packages installed:

- torch (1.9.0 or higher)
- torchvision (0.10.0 or higher)
- matplotlib (3.3.0 or higher)
- numpy (1.21.0 or higher)

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