import torch
import os

def check_5k_div3():
    model_path = "/rwthfs/rz/cluster/hpcwork/ro092286/MMCP_2026_Artifact_Hybrid_Inference/input/transformer_inference_scripted_fw2.pt"
    model = torch.jit.load(model_path, map_location="cpu")
    model.eval()

    # The user said 5k was the limit. Let's test multiples of 3 around there.
    # 5001 is divisible by 3 (1667 * 3)
    # 5004 is divisible by 3 (1668 * 3)
    
    for b in [5000, 5001, 5004, 6000]:
        print(f"Testing batch size {b} (divisible by 3: {b % 3 == 0})...")
        try:
            # We must use (Batch, Seq, Features) as verified before
            src = torch.randn(b, 5, 512)
            out = model(src)
            print(f"Batch {b} SUCCESS. Output shape: {out.shape}")
        except Exception as e:
            print(f"Batch {b} FAILED: {e}")

if __name__ == "__main__":
    check_5k_div3()
