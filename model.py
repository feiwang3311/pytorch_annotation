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
    def forward(ctx, input, is_begin):
        """
        Forward pass of the deliminator function.
        This is just an identity operation.

        Args:
            input: Input tensor
            is_begin: Boolean flag - True for partition begin, False for partition end
        """
        return input

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the deliminator function.
        This is just an identity operation.
        """
        return grad_output, None

    @staticmethod
    def symbolic(g, input, is_begin):
        """
        Symbolic function for exporting deliminator to ONNX.
        This will create a custom ONNX op called 'Deliminator' with is_begin attribute.

        Args:
            g: ONNX graph context
            input: Input tensor
            is_begin: Boolean flag - True for partition begin, False for partition end
        """
        return g.op("pytorch_annotation::Deliminator", input, is_begin_i=1 if is_begin else 0)


class SpaceSliceFunction(torch.autograd.Function):
   """
   A custom autograd function for the SpaceSlice operation.
   This function acts as an identity operation during forward/backward pass,
   but will be exported as a custom ONNX op during ONNX export.
   """

   @staticmethod
   def forward(ctx, input, dim=1, size=4):
       """
       Forward pass of the SpaceSlice function.
       This is just an identity operation.

       Args:
           input (torch.Tensor): Input tensor
           dim (int): Dimension attribute (default: 1)
           size (int): Size attribute (default: 4)
       """
       return input

   @staticmethod
   def backward(ctx, grad_output):
       """
       Backward pass of the SpaceSlice function.
       This is just an identity operation.
       """
       return grad_output

   @staticmethod
   def symbolic(g, input, dim=1, size=4):
       """
       Symbolic function for exporting SpaceSlice to ONNX.
       This will create a custom ONNX op called 'SpaceSlice' with attributes.

       Args:
           g: ONNX graph context
           input: Input tensor
           dim (int): Dimension attribute (default: 1)
           size (int): Size attribute (default: 4)
       """
       return g.op("pytorch_annotation::SpaceSlice", input, dim_i=dim, size_i=size)


def space_slice(input, dim=1, size=4):
   """
   Apply the SpaceSlice operation to the input tensor.
   This function acts as an identity operation during PyTorch execution,
   but will be exported as a custom ONNX op during ONNX export.

   Args:
       input (torch.Tensor): Input tensor
       dim (int): Dimension attribute (default: 1)
       size (int): Size attribute (default: 4)

   Returns:
       torch.Tensor: Output tensor (same as input)
   """
   # For PyTorch execution, we just return the input directly
   # For ONNX export, the symbolic function will handle the attributes
   return SpaceSliceFunction.apply(input, dim, size)


def deliminator(input, is_begin=True):
    """
    Apply the deliminator operation to the input tensor.
    This function acts as an identity operation during PyTorch execution,
    but will be exported as a custom ONNX op during ONNX export.

    Args:
        input (torch.Tensor): Input tensor
        is_begin (bool): True for partition begin, False for partition end

    Returns:
        torch.Tensor: Output tensor (same as input)
    """
    # For PyTorch execution, we just return the input directly
    # For ONNX export, the symbolic function will handle the attributes
    return DeliminatorFunction.apply(input, is_begin)


class PartitionBeginFunction(torch.autograd.Function):
    """
    Marks the beginning of a partition in the ONNX graph.
    """

    @staticmethod
    def forward(ctx, input, partition_name):
        ctx.partition_name = partition_name
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

    @staticmethod
    def symbolic(g, input, partition_name):
        return g.op("pytorch_annotation::PartitionBegin", input, partition_s=partition_name)


class PartitionEndFunction(torch.autograd.Function):
    """
    Marks the end of a partition in the ONNX graph.
    """

    @staticmethod
    def forward(ctx, input, partition_name):
        ctx.partition_name = partition_name
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

    @staticmethod
    def symbolic(g, input, partition_name):
        return g.op("pytorch_annotation::PartitionEnd", input, partition_s=partition_name)


def partition(name):
    """
    A decorator that automatically inserts deliminator ops at function boundaries.

    Usage:
        @partition("step_partition")
        def step(self, x):
            # ... operations ...
            return x

        def forward(self, x):
            for i in range(3):
                x = self.step(x)
            return x

    Each call to step() will automatically have deliminator() applied to
    the input tensor(s) (with is_begin=True) and output tensor(s) (with is_begin=False).

    Args:
        name (str): Name for the partition (currently unused, but available
                    for future use in PartitionBegin/End ops if needed)
    """
    import functools

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Apply deliminator to tensor arguments (is_begin=True for inputs)
            new_args = []
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    new_args.append(deliminator(arg, is_begin=True))
                else:
                    new_args.append(arg)

            # Apply deliminator to tensor keyword arguments (is_begin=True for inputs)
            new_kwargs = {}
            for key, val in kwargs.items():
                if isinstance(val, torch.Tensor):
                    new_kwargs[key] = deliminator(val, is_begin=True)
                else:
                    new_kwargs[key] = val

            # Call the original function
            result = func(*new_args, **new_kwargs)

            # Apply deliminator to output (is_begin=False for outputs)
            if isinstance(result, torch.Tensor):
                return deliminator(result, is_begin=False)
            elif isinstance(result, tuple):
                return tuple(
                    deliminator(r, is_begin=False) if isinstance(r, torch.Tensor) else r
                    for r in result
                )
            return result

        return wrapper

    return decorator









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

    @partition("step")
    def step(self, x):
        """
        Forward pass of the neural network.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        # Add Space Slice Attribute
        x = space_slice(x, dim=2, size=5)

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
        return self.fc3(x)

    def forward(self, x):
        for i in range(3):
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