"""
PyTorch model that produces ONNX identical to deliminator2_sim.onnx

The key insight from deliminator2_sim.onnx:
- Input: [1, 1, 28, 28] -> flattened to [1, 784] via Reshape
- Iteration 1: Deliminator -> Reshape -> FC1 -> ReLU -> FC2 -> ReLU -> FC3 -> Deliminator
- Iteration 2-3: Deliminator -> Reshape -> ConstantOfShape -> Concat (padding) -> FC layers -> Deliminator
- Output: [1, 10]

The simplified ONNX uses pre-computed reshape shapes as constants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import partition, deliminator, reset_partition_counters


class SimpleNN_SimSim(nn.Module):
    """
    A neural network that produces cleaner ONNX matching deliminator2_sim.onnx.
    Uses fixed shapes - no dynamic shape operations.
    """
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleNN_SimSim, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

        # Pre-register fixed padding buffer (784 - 10 = 774 zeros)
        self.register_buffer('padding', torch.zeros(1, input_size - num_classes))

    @partition("partition_name", "scheduling_config")
    def step_first(self, x):
        """First step: input is [1, 1, 28, 28], flatten inside."""
        # Reshape inside the partition so Deliminator comes first
        x = x.view(1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    @partition("step", "scheduling_config")
    def step_rest(self, x):
        """Subsequent steps: input is [1, 10], needs padding to [1, 784]."""
        # Reshape to [1, 10] to match original structure
        x = x.view(1, self.num_classes)
        # Concat with fixed padding: [1, 10] + [1, 774] -> [1, 784]
        x = torch.cat([x, self.padding], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward(self, x):
        """
        Forward pass: 3 iterations.
        """
        x = self.step_first(x)
        x = self.step_rest(x)
        x = self.step_rest(x)
        return x


def create_model_sim_sim():
    """Create an instance of SimpleNN_SimSim."""
    return SimpleNN_SimSim()


def load_weights_from_onnx(model, onnx_path):
    """Load weights from the original ONNX file."""
    import onnx
    from onnx import numpy_helper

    onnx_model = onnx.load(onnx_path)

    # Extract weights
    weights = {}
    for init in onnx_model.graph.initializer:
        weights[init.name] = numpy_helper.to_array(init)

    # Load into PyTorch model
    with torch.no_grad():
        if 'fc1.weight' in weights:
            model.fc1.weight.copy_(torch.from_numpy(weights['fc1.weight']))
        if 'fc1.bias' in weights:
            model.fc1.bias.copy_(torch.from_numpy(weights['fc1.bias']))
        if 'fc2.weight' in weights:
            model.fc2.weight.copy_(torch.from_numpy(weights['fc2.weight']))
        if 'fc2.bias' in weights:
            model.fc2.bias.copy_(torch.from_numpy(weights['fc2.bias']))
        if 'fc3.weight' in weights:
            model.fc3.weight.copy_(torch.from_numpy(weights['fc3.weight']))
        if 'fc3.bias' in weights:
            model.fc3.bias.copy_(torch.from_numpy(weights['fc3.bias']))

    return model


if __name__ == "__main__":
    import onnx

    print("=" * 60)
    print("Testing SimpleNN_SimSim (matches deliminator2_sim.onnx)")
    print("=" * 60)

    # Create model and load weights from original ONNX
    model = create_model_sim_sim()
    model = load_weights_from_onnx(model, 'deliminator2_sim.onnx')
    model.eval()
    print(model)

    # Test forward pass
    x = torch.randn(1, 1, 28, 28)
    out = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {out.shape}")

    # Note: Can't compare with ONNX runtime since Deliminator is a custom op
    # The PyTorch model with same weights should produce identical results

    # Export to ONNX
    print("\n=== Exporting to ONNX ===")
    # Reset partition counters before export so names start from 0
    reset_partition_counters()
    torch.onnx.export(
        model,
        x,
        'model_sim_sim_exported.onnx',
        input_names=['input'],
        output_names=['output'],
        opset_version=11
    )
    print("Exported to model_sim_sim_exported.onnx")

    # Compare node counts
    orig_model = onnx.load('deliminator2_sim.onnx')
    new_model = onnx.load('model_sim_sim_exported.onnx')

    print(f"\nOriginal ONNX nodes: {len(orig_model.graph.node)}")
    print(f"New model nodes: {len(new_model.graph.node)}")

    # Show Deliminator positions
    print("\nOriginal Deliminator nodes:")
    for i, node in enumerate(orig_model.graph.node):
        if 'Deliminator' in node.op_type:
            print(f"  {i}: {node.op_type}")

    print("\nNew model Deliminator nodes:")
    for i, node in enumerate(new_model.graph.node):
        if 'Deliminator' in node.op_type:
            # Extract attributes
            attrs = {}
            for attr in node.attribute:
                if attr.name == 'is_begin':
                    attrs['is_begin'] = attr.i
                elif attr.name == 'partition_name':
                    attrs['partition_name'] = attr.s.decode() if isinstance(attr.s, bytes) else attr.s
                elif attr.name == 'scheduling_config':
                    attrs['scheduling_config'] = attr.s.decode() if isinstance(attr.s, bytes) else attr.s
            print(f"  {i}: {node.op_type} - {attrs}")

    # Show full structure comparison
    print("\n=== Original ONNX structure ===")
    for i, node in enumerate(orig_model.graph.node):
        print(f"  {i:2d}. {node.op_type}")

    print("\n=== New model ONNX structure ===")
    for i, node in enumerate(new_model.graph.node):
        print(f"  {i:2d}. {node.op_type}")
