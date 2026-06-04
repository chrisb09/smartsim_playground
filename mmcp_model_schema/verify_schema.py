import torch
import os
import sys

def verify_model_schema():
    # Configuration based on analysis
    seq_len = 5
    feature_dim = 512
    num_fields = 3
    num_cubes = 10  # Dummy value for testing
    batch_size = num_fields * num_cubes
    forecast_window = 2
    
    # Path to the scripted model identified in the TOML
    model_path = "/rwthfs/rz/cluster/hpcwork/ro092286/MMCP_2026_Artifact_Hybrid_Inference/input/transformer_inference_scripted_fw2.pt"
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return

    print(f"Loading model from: {model_path}")
    try:
        # Load the TorchScript model
        # map_location='cpu' ensures it works even if GPUs aren't immediately available/configured in the current shell
        model = torch.jit.load(model_path, map_location="cpu")
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return

    # Create dummy input: (Batch, Seq, Feature)
    # The wrapper in transformer_inference_to_script.py uses batch_first=True
    src = torch.randn(batch_size, seq_len, feature_dim)
    
    print(f"Input shape (src): {src.shape}")
    print(f"Expected output shape: [{batch_size}, {forecast_window}, {feature_dim}]")

    try:
        with torch.no_grad():
            output = model(src)
        
        print(f"Inference successful!")
        print(f"Output shape: {output.shape}")
        
        # Verify output dimensions
        expected_shape = (batch_size, forecast_window, feature_dim)
        if output.shape == expected_shape:
            print("SUCCESS: Output shape matches expectations.")
        else:
            print(f"WARNING: Output shape {output.shape} differs from expected {expected_shape}")
            
    except Exception as e:
        print(f"ERROR during inference: {e}")
        print("\nAttempting with (Seq, Batch, Feature) just in case...")
        try:
            src_alt = src.transpose(0, 1) # [seq, batch, features]
            print(f"Alternative input shape: {src_alt.shape}")
            output = model(src_alt)
            print(f"Inference successful with Alternative shape!")
            print(f"Output shape: {output.shape}")
        except Exception as e2:
            print(f"ERROR during alternative inference: {e2}")

if __name__ == "__main__":
    verify_model_schema()
