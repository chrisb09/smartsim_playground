import torch
import os

def check_5k_limit():
    model_path = "/rwthfs/rz/cluster/hpcwork/ro092286/MMCP_2026_Artifact_Hybrid_Inference/input/transformer_inference_scripted_fw2.pt"
    model = torch.jit.load(model_path, map_location="cpu")
    model.eval()

    # Test 5000 (The limit)
    print("Testing batch size 5000...")
    try:
        src = torch.randn(5000, 5, 512)
        out = model(src)
        print("Batch 5000 SUCCESS")
    except Exception as e:
        print(f"Batch 5000 FAILED: {e}")

    # Test 5001 (Breaking the limit)
    print("\nTesting batch size 5001...")
    try:
        src = torch.randn(5001, 5, 512)
        out = model(src)
        print("Batch 5001 SUCCESS")
    except Exception as e:
        print(f"Batch 5001 FAILED (Expected): {e}")

if __name__ == "__main__":
    check_5k_limit()
