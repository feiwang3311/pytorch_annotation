import torch
import onnx
import onnxruntime as ort
import matplotlib.pyplot as plt
import numpy as np
from model import create_model

def trace_model_to_onnx(model, input_tensor, onnx_path):
    """
    Trace the PyTorch model and export it to ONNX format.
    
    Args:
        model (nn.Module): The PyTorch model to trace
        input_tensor (torch.Tensor): Sample input tensor for tracing
        onnx_path (str): Path to save the ONNX file
    """
    # Set the model to evaluation mode
    model.eval()
    
    # Export the model to ONNX directly (without tracing first)
    torch.onnx.export(
        model,
        input_tensor,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        # dynamo=True
    )
    
    print(f"Model traced and exported to {onnx_path}")

def visualize_onnx_model(onnx_path):
    """
    Visualize the ONNX model structure.
    
    Args:
        onnx_path (str): Path to the ONNX file
    """
    # Load the ONNX model
    onnx_model = onnx.load(onnx_path)
    
    # Check that the model is well-formed
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid")
    
    # Print model information
    print(f"Model IR version: {onnx_model.ir_version}")
    print(f"Model opset: {onnx_model.opset_import}")
    print(f"Number of nodes: {len(onnx_model.graph.node)}")
    
    # Print node information
    print("\nModel nodes:")
    for i, node in enumerate(onnx_model.graph.node):
        print(f"  Node {i}: {node.op_type} - {node.name}")
        print(f"    Inputs: {node.input}")
        print(f"    Outputs: {node.output}")
    
    return onnx_model

def visualize_model_graph(onnx_model, output_path="model_graph.png"):
    """
    Create a visualization of the model graph.
    
    Args:
        onnx_model: The ONNX model
        output_path (str): Path to save the visualization
    """
    # Create a simple visualization of the model structure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define nodes and their positions
    nodes = [
        ("Input\n(1, 1, 28, 28)", 0, 0),
        ("Flatten", 1, 0),
        ("Linear\n(784 → 128)", 2, 0),
        ("ReLU", 3, 0),
        ("Linear\n(128 → 128)", 4, 0),
        ("ReLU", 5, 0),
        ("Linear\n(128 → 10)", 6, 0),
        ("Output\n(1, 10)", 7, 0)
    ]
    
    # Draw nodes
    for i, (label, x, y) in enumerate(nodes):
        ax.text(x, y, label, ha='center', va='center', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
                fontsize=10)
    
    # Draw arrows between nodes
    for i in range(len(nodes) - 1):
        ax.annotate('', xy=(nodes[i+1][1] - 0.3, nodes[i+1][2]), 
                    xytext=(nodes[i][1] + 0.3, nodes[i][2]),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    # Set plot properties
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.title('SimpleNN Model Forward Path', fontsize=16, pad=20)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Model graph visualization saved to {output_path}")

def load_and_test_onnx_model(onnx_path, input_tensor):
    """
    Load and test the ONNX model.
    
    Args:
        onnx_path (str): Path to the ONNX file
        input_tensor (torch.Tensor): Input tensor for testing
    """
    # Create ONNX Runtime session
    # This line of code errors out because the onnx runtime doesn't know about Deliminator node.
    ort_session = ort.InferenceSession(onnx_path)
    
    # Convert PyTorch tensor to numpy
    input_np = input_tensor.detach().cpu().numpy()
    
    # Run inference
    ort_inputs = {ort_session.get_inputs()[0].name: input_np}
    ort_out = ort_session.run(None, ort_inputs)
    
    print(f"ONNX model output shape: {ort_out[0].shape}")
    print(f"ONNX model output (first 5 values): {ort_out[0][0][:5]}")
    
    return ort_out[0]

def main():
    # Create the model
    model = create_model()
    print("Model created successfully")
    
    # Create a sample input tensor
    sample_input = torch.randn(1, 1, 28, 28)
    print(f"Sample input shape: {sample_input.shape}")
    
    # Trace the model and export to ONNX
    onnx_path = "simple_nn_model.onnx"
    trace_model_to_onnx(model, sample_input, onnx_path)
    
    # Visualize the ONNX model
    onnx_model = visualize_onnx_model(onnx_path)
    
    # Create a visualization of the model graph
    visualize_model_graph(onnx_model, "model_forward_path.png")
    
    # Load and test the ONNX model
    onnx_output = load_and_test_onnx_model(onnx_path, sample_input)
    
    # Compare with PyTorch model output
    model.eval()
    with torch.no_grad():
        pytorch_output = model(sample_input)
    
    print(f"PyTorch model output shape: {pytorch_output.shape}")
    print(f"PyTorch model output (first 5 values): {pytorch_output[0][:5]}")
    
    # Check if outputs are close
    diff = np.abs(onnx_output - pytorch_output.detach().cpu().numpy()).max()
    print(f"Max difference between PyTorch and ONNX outputs: {diff}")

if __name__ == "__main__":
    main()