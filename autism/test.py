import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split


class CustomDataset(Dataset):
    def __init__(self, path, hw, transform=None):
        """
        Dataset personalizado para carregar imagens.
        """
        self.images = []
        self.labels = []
        self.hw = hw
        self.transform = transform

        if not os.path.isdir(path):
            print(f"The folder '{path}' does not exist or is not valid.")
            return

        for file in os.listdir(path):
            if file.lower().endswith('.jpg'):
                try:
                    image_path = os.path.join(path, file)
                    image = Image.open(image_path).convert('RGB')
                    label = 1 if 'autistic' in path.lower() else 0

                    if self.transform:
                        image = self.transform(image)

                    self.images.append(image)
                    self.labels.append(label)
                except Exception as e:
                    print(f"Error processing image '{file}': {e}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(256, 512, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # Calcular tamanho da saída da parte convolucional
        test_input = torch.rand(1, 3, 224, 224)  # Ajustar para o tamanho da entrada
        conv_output_size = self._get_conv_output_size(test_input)

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def _get_conv_output_size(self, input_shape):
        with torch.no_grad():
            output = self.conv_layers(input_shape)
        return output.view(output.size(0), -1).size(1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def evaluate(model, dataloader, device, threshold=0.5):
    """
    Avalia o modelo em um conjunto de dados.
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            # Verifique as formas das entradas
            print(f"Batch {batch_idx + 1}:")
            print(f"Images shape: {images.shape}")
            print(f"Labels shape: {labels.shape}")

            outputs = model(images)
            print(f"Outputs shape: {outputs.shape}")
            print(f"Outputs (raw): {outputs[:5]}")  # Exibe os primeiros 5 resultados brutos

            predictions = (outputs.squeeze(1) > threshold).float()
            print(f"Predictions: {predictions[:5]}")
            print(f"Labels: {labels[:5]}")

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # Opcional: Acompanhe a acurácia parcial
            print(f"Partial accuracy after batch {batch_idx + 1}: {correct / total * 100:.2f}%")

    accuracy = correct / total * 100
    print(f"Final Accuracy: {accuracy:.2f}%")



def main():
    hw = 224
    batch_size = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((hw, hw)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_a = CustomDataset("autism\\images\\train\\autistic", hw, transform=transform)
    train_na = CustomDataset("autism\\images\\train\\non_autistic", hw, transform=transform)
    train_dataset = train_a + train_na

    # Split into training and validation
    train_size = int(0.7 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_data, val_data = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    model = CNNModel().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    while True:
        command = input("'1' Train the model \n'2' Evaluate autistic images \n'3' Evaluate non-autistic images \n'4' Save the model \n'5' Load a model \n'0' Exit: ")

        if command == '1':
            print("Training the model...")
            model.train()
            for epoch in range(10):
                total_loss = 0
                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device).float()
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels.unsqueeze(1))
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

        elif command == '2':
            print("Evaluating autistic images...")
            eval_dataset = CustomDataset("autism/images/valid/autistic", hw, transform=transform)
            eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
            evaluate(model, eval_loader, device)

        elif command == '3':
            print("Evaluating non-autistic images...")
            eval_dataset = CustomDataset("autism/images/valid/non_autistic", hw, transform=transform)
            eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
            evaluate(model, eval_loader, device)

        elif command == '4':
            torch.save(model.state_dict(), 'autism/neural_networks/model.pth')
            print("Model saved.")

        elif command == '5':
            models_path = 'autism/neural_networks'
            try:
                models = [f for f in os.listdir(models_path) if f.endswith('.pth')]
                if not models:
                    print("No models found.")
                    continue
                for i, m in enumerate(models, 1):
                    print(f"{i}. {m}")
                choice = int(input("Choose a model to load: ")) - 1
                model.load_state_dict(torch.load(os.path.join(models_path, models[choice])))
                model.to(device)
                print(f"Model '{models[choice]}' loaded.")
            except Exception as e:
                print(f"Error loading the model: {e}")

        elif command == '0':
            print("Exiting...")
            break

        else:
            print("Invalid command.")


if __name__ == "__main__":
    main()
