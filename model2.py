import torch
import torch.nn as nn
import torch.nn.functional as F

from model import partition, deliminator


class ClosureTestNN(nn.Module):
    """
    Test case: closure variable escapes the deliminator.
    """
    def __init__(self, hidden_size=128):
        super(ClosureTestNN, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    @partition("step")
    def step(self, x):
        """
        This function uses self.z (a closure variable) that is NOT passed as argument.
        The deliminator will NOT be applied to self.z!
        """
        x = F.relu(self.fc1(x))
        x = x + self.z  # self.z is a closure variable - escapes deliminator!
        x = F.relu(self.fc2(x))
        return x

    def forward(self, x, z):
        self.z = z  # Store z as instance variable (closure)
        for i in range(3):
            x = self.step(x)
        return x


class MultiIONN(nn.Module):
    """
    A neural network with multiple inputs and outputs to test the @partition decorator.
    """
    def __init__(self, hidden_size=128):
        super(MultiIONN, self).__init__()
        # All layers use hidden_size -> hidden_size so iterations work
        self.fc1_a = nn.Linear(hidden_size, hidden_size)
        self.fc1_b = nn.Linear(hidden_size, hidden_size)
        self.fc2_a = nn.Linear(hidden_size, hidden_size)
        self.fc2_b = nn.Linear(hidden_size, hidden_size)

    @partition("step")
    def step(self, x, y):
        """
        A step function with multiple inputs and outputs.

        Args:
            x (torch.Tensor): First input tensor
            y (torch.Tensor): Second input tensor

        Returns:
            tuple: (processed_x, processed_y)
        """
        x = F.relu(self.fc1_a(x))
        y = F.relu(self.fc1_b(y))

        x = F.relu(self.fc2_a(x))
        y = F.relu(self.fc2_b(y))

        return x, y

    def forward(self, x, y):
        for i in range(3):
            x, y = self.step(x, y)
        return x, y


def test_closure():
    """Test that closure variables escape the deliminator."""
    print("=" * 60)
    print("TEST: Closure variable escapes deliminator")
    print("=" * 60)

    model = ClosureTestNN()
    model.eval()
    print(model)

    x = torch.randn(1, 128)
    z = torch.randn(1, 128)  # This will be a closure variable

    # Forward pass
    out = model(x, z)
    print(f"\nInput shapes: x={x.shape}, z={z.shape}")
    print(f"Output shape: {out.shape}")

    # Export to ONNX
    print("\nExporting to ONNX...")
    torch.onnx.export(
        model,
        (x, z),
        'test_closure.onnx',
        input_names=['input_x', 'input_z'],
        output_names=['output'],
        opset_version=11
    )
    print("Exported to test_closure.onnx")

    # Inspect the ONNX graph
    import onnx
    onnx_model = onnx.load('test_closure.onnx')

    print("\n=== ONNX Graph Nodes ===")
    deliminator_count = 0
    for i, node in enumerate(onnx_model.graph.node):
        print(f"  Node {i:2d}: {node.op_type:20s} inputs={list(node.input)}")
        if 'Deliminator' in node.op_type:
            deliminator_count += 1

    print(f"\nTotal Deliminator ops: {deliminator_count}")
    print("Note: z (closure) does NOT have deliminator applied!")
    print("Expected: 6 (1 input x + 1 output) x 3 = 6, NOT 12")


def test_multi_io():
    """Test multiple inputs/outputs."""
    print("\n" + "=" * 60)
    print("TEST: Multiple inputs and outputs")
    print("=" * 60)

    model = MultiIONN()
    model.eval()
    print(model)

    x = torch.randn(1, 128)
    y = torch.randn(1, 128)

    # Forward pass
    out_x, out_y = model(x, y)
    print(f"\nInput shapes: x={x.shape}, y={y.shape}")
    print(f"Output shapes: out_x={out_x.shape}, out_y={out_y.shape}")

    # Export to ONNX
    print("\nExporting to ONNX...")
    torch.onnx.export(
        model,
        (x, y),
        'test_multi_io.onnx',
        input_names=['input_x', 'input_y'],
        output_names=['output_x', 'output_y'],
        opset_version=11
    )
    print("Exported to test_multi_io.onnx")

    # Inspect the ONNX graph
    import onnx
    onnx_model = onnx.load('test_multi_io.onnx')

    print("\n=== ONNX Graph Nodes (Deliminators only) ===")
    deliminator_count = 0
    for i, node in enumerate(onnx_model.graph.node):
        if 'Deliminator' in node.op_type:
            deliminator_count += 1
            print(f"  Node {i:2d}: {node.op_type}")

    print(f"\nTotal Deliminator ops: {deliminator_count}")
    print("Expected: 12 (2 inputs + 2 outputs) x 3 iterations = 12")


if __name__ == "__main__":
    test_closure()
    test_multi_io()
