import torch
import os

def verify_type_agnostic():
    model_path = "/rwthfs/rz/cluster/hpcwork/ro092286/MMCP_2026_Artifact_Hybrid_Inference/input/transformer_inference_scripted_fw2.pt"
    if not os.path.exists(model_path):
        print("ERROR: Model not found.")
        return

    print(f"Loading model...")
    model = torch.jit.load(model_path, map_location="cpu")
    model.eval()

    # Test Float (float32)
    print("\n--- Testing float32 ---")
    src_float = torch.randn(1, 5, 512, dtype=torch.float32)
    try:
        out_float = model(src_float)
        print(f"float32 input SUCCESS. Output dtype: {out_float.dtype}")
    except Exception as e:
        print(f"float32 input FAILED: {e}")

    # Test Double (float64)
    print("\n--- Testing float64 (Double) ---")
    src_double = torch.randn(1, 5, 512, dtype=torch.float64)
    try:
        out_double = model(src_double)
        print(f"float64 input SUCCESS. Output dtype: {out_double.dtype}")
    except Exception as e:
        print(f"float64 input FAILED: {e}")
        print("This usually happens if the model has hardcoded float32 layers (Linear, etc.) without explicit casting.")

if __name__ == "__main__":
    verify_type_agnostic()
