#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np

print("Creating a simple PyTorch model and saving it to input/model.pt")

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)

class MoreComplexModel(nn.Module):
    def __init__(self):
        super(MoreComplexModel, self).__init__()
        self.layer1 = nn.Linear(10, 20)
        self.activation1 = nn.ReLU()
        self.layer2 = nn.Linear(20, 32)
        self.activation2 = nn.Sigmoid()
        self.layer3 = nn.Linear(32, 16)
        self.activation3 = nn.Tanh()
        self.layer4 = nn.Linear(16, 8)
        self.activation4 = nn.LeakyReLU()
        self.layer5 = nn.Linear(8, 4)
        self.activation5 = nn.ELU()
        self.layer6 = nn.Linear(4, 2)
        self.activation6 = nn.Identity()
        self.layer7 = nn.Linear(2, 1)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.layer2(x)
        x = self.activation2(x)
        x = self.layer3(x)
        x = self.activation3(x)
        x = self.layer4(x)
        x = self.activation4(x)
        x = self.layer5(x)
        x = self.activation5(x)
        x = self.layer6(x)
        x = self.activation6(x)
        x = self.layer7(x)
        return x

#model = SimpleModel()
model = MoreComplexModel()

def train_label_func(x):
    return (sum(x) / len(x))**0.5

np.random.seed(42)
training_data = np.random.rand(1000, 10)
training_labels = np.array([[train_label_func(row)] for row in training_data], dtype=np.float32)
test_data = np.random.rand(10)
test_data_label = train_label_func(test_data)

# Convert to PyTorch tensors
X_train = torch.from_numpy(training_data.astype(np.float32))
y_train = torch.from_numpy(training_labels)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
epochs = 100000
prev_loss = float('inf')
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        improvement = (prev_loss - loss.item()) / prev_loss * 100 if prev_loss != float('inf') else 0
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}, Improvement: {improvement:.6f}%")
    if loss.item() > 0.99 * prev_loss:  # Stop if loss is not improving significantly
        print("Early stopping due to lack of improvement at epoch", epoch+1)
        break
    prev_loss = loss.item()
model.eval()

with torch.no_grad():
    example_input = torch.from_numpy(test_data.astype(np.float32)).unsqueeze(0)
    example_output = model(example_input)
    print(f"Example input: {example_input}")
    print(f"Example output: {example_output}")
    print("Expected output (label):", test_data_label)

print("Model created")

# Convert to TorchScript and save
example_input = torch.from_numpy(test_data.astype(np.float32)).unsqueeze(0)
scripted_model = torch.jit.trace(model, example_input)
torch.jit.save(scripted_model, "input/model_complex.pt")

print("Model saved to input/model_complex.pt as TorchScript")

# Verify that the model can be loaded correctly
loaded_model = torch.jit.load("input/model_complex.pt")

with torch.no_grad():
    example_input_2 = torch.from_numpy(test_data.astype(np.float32)).unsqueeze(0)
    example_output_2 = loaded_model(example_input_2)
    print(f"Loaded model example input: {example_input_2}")
    print(f"Loaded model example output: {example_output_2}")


print("Model verification complete")
print("Output differences:", example_output - example_output_2)


# Also create dummy data.hdf5 file
import h5py
import numpy as np

with h5py.File("input/data.hdf5", "w") as f:
    data_train = np.random.rand(10, 10).astype(np.float32)
    print("Creating input/data.hdf5 with training data and labels")
    print(data_train)
    real_labels = np.array([[train_label_func(row)] for row in data_train], dtype=np.float32)
    print("Real labels for the training data:")
    print(real_labels)
    print("Predicted labels from the model for the training data:")
    with torch.no_grad():
        predicted_labels = model(torch.from_numpy(data_train.astype(np.float32)))
    print(predicted_labels)
    f.create_dataset("data", data=data_train)
    f.create_dataset("label", data=predicted_labels.numpy())