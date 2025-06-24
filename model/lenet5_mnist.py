import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define relevant variables for the ML task
batch_size = 64
num_classes = 10
learning_rate = 0.001
num_epochs = 10

# Device will determine whether to run the training on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

transform = transforms.Compose([transforms.Resize((32,32)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,)) # Mean and Std Dev for MNIST
                                ])

# Download and load the training data
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# Download and load the test data
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

print(f"\n Number of training samples: {len(train_dataset)}")
print(f"\n Number of test samples: {len(test_dataset)}")

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5) # 32x32 -> 28x28

        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2) # 28x28 -> 14x14

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5) # 14x14 -> 10x10

        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2) # 10x10 -> 5x5

        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)

        self.fc2 = nn.Linear(in_features=120, out_features=84)

        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        # Using RELU instead of tanh, which was used in the origicinal LeNet5 Modle
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        # Output layer
        x = self.fc3(x)

        return x

model = LeNet5().to(device)
print(model)

criterion = nn.CrossEntropyLoss();
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(num_epochs):
    # Model Training
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Forward Pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backwards & Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    epoch_train_loss = running_loss / len(train_dataset)
    epoch_train_accuracy = 100 * correct_train / total_train

    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_accuracy)

    # Model Evaluation
    model.eval()
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    epoch_test_accuracy = 100 * correct_test / total_test
    test_accuracies.append(epoch_test_accuracy)

    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {epoch_train_loss:.4f}, '
          f'Train Accuracy: {epoch_train_accuracy:.2f}%, '
          f'Test Accuracy: {epoch_test_accuracy:.2f}%')
    
model.eval() # Ensure model is in evaluation mode
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

final_test_acc = 100 * correct / total
print(f'Final Test Accuracy: {final_test_acc:.2f}%')

model_save_path = "lenet5_mnist_pytorch.pth"
torch.save(model, model_save_path) # save weight AND architecture