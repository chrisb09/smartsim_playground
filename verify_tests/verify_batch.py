import torch
import os

def verify_large_batch():
    model_path = "/rwthfs/rz/cluster/hpcwork/ro092286/MMCP_2026_Artifact_Hybrid_Inference/input/transformer_inference_scripted_fw2.pt"
    model = torch.jit.load(model_path, map_location="cpu")
    model.eval()

    # Try different batch sizes
    for b in [1, 32, 128, 512, 1024]:
        print(f"Testing batch size: {b}")
        src = torch.randn(b, 5, 512)
        try:
            with torch.no_grad():
                out = model(src)
            print(f"SUCCESS. Output shape: {out.shape}")
        except Exception as e:
            print(f"FAILED: {e}")
            break

if __name__ == "__main__":
    verify_large_batch()
