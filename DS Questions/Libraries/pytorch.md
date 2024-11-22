# PyTorch

## What is PyTorch and how does it differ from other frameworks?

PyTorch is a deep learning framework that provides:
- Dynamic computational graphs (define-by-run)
- Native Python integration
- Imperative programming style
- GPU acceleration
- Rich ecosystem of tools and libraries
- Production deployment capabilities

Key differences from other frameworks:
- More Pythonic than TensorFlow
- Dynamic graphs vs TensorFlow's static graphs (pre 2.0)
- Better debugging capabilities
- More flexible research platform

## What are tensors in PyTorch and how do you create them?

Tensors are multi-dimensional arrays and the fundamental data structure in PyTorch.

Common ways to create tensors:
```python
import torch

# From Python list/array
x = torch.tensor([1, 2, 3])

# Zeros and ones
zeros = torch.zeros(2, 3)  # 2x3 tensor of zeros
ones = torch.ones(2, 3)    # 2x3 tensor of ones

# Random tensors
rand = torch.rand(2, 3)     # uniform random
randn = torch.randn(2, 3)   # normal distribution
randint = torch.randint(0, 10, (2, 3))  # random integers

# Range tensors
arange = torch.arange(0, 10, step=1)
linspace = torch.linspace(0, 10, steps=5)

# From NumPy array
import numpy as np
np_array = np.array([1, 2, 3])
tensor = torch.from_numpy(np_array)
```

## What are the basic tensor operations in PyTorch?

Common tensor operations:
```python
# Arithmetic operations
x + y  # Addition
x - y  # Subtraction
x * y  # Element-wise multiplication
x / y  # Division
x @ y  # Matrix multiplication
x.matmul(y)  # Matrix multiplication
x.mm(y)      # Matrix multiplication

# In-place operations
x.add_(y)    # In-place addition
x.sub_(y)    # In-place subtraction
x.mul_(y)    # In-place multiplication

# Reshaping
x.view(3, 4)     # Reshape tensor
x.reshape(3, 4)  # Reshape tensor (may copy)
x.squeeze()      # Remove dimensions of size 1
x.unsqueeze(0)   # Add dimension
x.transpose(0, 1) # Transpose dimensions

# Indexing and slicing
x[0]           # First element
x[:, 1]        # Second column
x[1:3, 2:4]    # 2D slice

# Concatenation
torch.cat([x, y], dim=0)  # Concatenate along dimension
torch.stack([x, y], dim=0)  # Stack tensors
```

## How do you move tensors between CPU and GPU?

Device management in PyTorch:
```python
# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create tensor on specific device
x = torch.tensor([1, 2, 3], device=device)

# Move tensor to device
x = x.to(device)
x = x.cuda()  # Move to GPU
x = x.cpu()   # Move to CPU

# Check tensor device
x.device

# Move model to device
model = MyModel()
model.to(device)
```

## What are the main components of a neural network in PyTorch?

Basic neural network components:
```python
import torch.nn as nn

class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Layers
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.fc1 = nn.Linear(100, 50)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
        # Pooling layers
        self.maxpool = nn.MaxPool2d(2)
        self.avgpool = nn.AvgPool2d(2)
        
        # Normalization
        self.batchnorm = nn.BatchNorm2d(20)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.fc1(x)
        return x
```

## How do you define loss functions and optimizers?

Common loss functions and optimizers:
```python
import torch.optim as optim

# Loss functions
criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
criterion = nn.BCELoss()
criterion = nn.L1Loss()

# Optimizers
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.RMSprop(model.parameters(), lr=0.01)

# Learning rate schedulers
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
```

## How do you implement a training loop in PyTorch?

Basic training loop structure:
```python
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()  # Set training mode
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            # Move to device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # Print epoch statistics
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}')
```

## How do you implement model evaluation?

Evaluation loop structure:
```python
def evaluate_model(model, test_loader, criterion):
    model.eval()  # Set evaluation mode
    
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():  # Disable gradient computation
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            total_loss += loss.item()
    
    # Calculate metrics
    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy
```

## How do you save and load models in PyTorch?

Model persistence:
```python
# Save model
torch.save(model.state_dict(), 'model.pth')
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss
}, 'checkpoint.pth')

# Load model
model = MyModel()
model.load_state_dict(torch.load('model.pth'))

# Load checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```

## How do you implement data loading in PyTorch?

Data loading and datasets:
```python
from torch.utils.data import Dataset, DataLoader

# Custom dataset
class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Create data loader
dataset = MyDataset(data, labels)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Built-in datasets
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True,
                             download=True, transform=transform)
```

## What are the common data transformations in PyTorch?

Data transformations using torchvision:
```python
from torchvision import transforms

transform = transforms.Compose([
    # Image transformations
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    
    # Convert to tensor
    transforms.ToTensor(),
    
    # Normalize
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

## How do you implement transfer learning in PyTorch?

Transfer learning implementation:
```python
import torchvision.models as models

# Load pre-trained model
model = models.resnet18(pretrained=True)

# Freeze parameters
for param in model.parameters():
    param.requires_grad = False

# Modify final layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

# Fine-tuning
model.train()
optimizer = optim.Adam(model.fc.parameters())  # Only train final layer
```

## How do you handle custom loss functions?

Creating custom loss functions:
```python
class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        loss = torch.mean((pred - target)**2)
        return loss

# Alternative using functional approach
def custom_loss(pred, target):
    return torch.mean((pred - target)**2)
```

## How do you implement early stopping?

Early stopping implementation:
```python
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
```

## How do you handle gradients manually?

Manual gradient operations:
```python
# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Access gradients
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.grad)

# Custom autograd function
class CustomFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input
```