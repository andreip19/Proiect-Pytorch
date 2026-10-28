import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

DEVICE = torch.device("cpu")
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 14 * 14, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

def main():
    print("Se pregateste setul de date MNIST cu Data Augmentation...")
    
 
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor()
    ])
    

    test_transform = transforms.ToTensor()
    
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = SimpleCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Incepe antrenarea modelului imbunatatit...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        average_loss = total_loss / len(train_loader)
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images.to(DEVICE))
                predictions = outputs.argmax(dim=1)
                correct += (predictions == labels.to(DEVICE)).sum().item()
                total += labels.size(0)
                
        accuracy = 100.0 * correct / total
        print(f"Epoca {epoch + 1}/{EPOCHS} | Pierdere: {average_loss:.4f} | Acuratete test: {accuracy:.2f}%")
        
    model.eval()
    example_input = torch.zeros(1, 1, 28, 28)
    print("Se exporta modelul TorchScript...")
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save("model_traced.pt")
    print("Model salvat in model_traced.pt")
    print(f"Acuratete finala pe test: {accuracy:.2f}%")

if __name__ == "__main__":
    main()