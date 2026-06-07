import torch

class MinimalModel(torch.nn.Module):
    def forward(self, x):
        return x * 2.0

class ShapeMismatchModel(torch.nn.Module):
    def forward(self, x):
        if x.shape[1] != 18:
            raise RuntimeError(f"Expected 18 features, got {x.shape[1]}")
        return x * 2.0

if __name__ == '__main__':
    scripted_model = torch.jit.script(MinimalModel())
    scripted_model.save("minimal_model.pt")

    scripted2 = torch.jit.script(ShapeMismatchModel())
    scripted2.save("shape_mismatch_model.pt")
