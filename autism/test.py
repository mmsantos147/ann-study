import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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
                    label = 0 if 'non_autistic' in path.lower() else 1

                    if self.transform:
                        image = self.transform(image)

                    self.images.append(image)
                    self.labels.append(float(label))  # Garantir que o rótulo seja float
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
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device).float()
            outputs = model(images)
            predictions = (outputs.squeeze(1) > threshold).float()

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    accuracy = correct / total * 100
    print(f"Final Accuracy: {accuracy:.5f}%")

    # Plotando a matriz de confusão
    cm = plot_confusion_matrix(all_labels, all_predictions)
    print("Confusion Matrix:\n", cm)

def plot_confusion_matrix(labels, predictions):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(labels, predictions, labels=[0, 1])  # Especifica as classes [0, 1]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Autistic", "Autistic"])
    disp.plot(cmap="Blues")
    return cm


def main():
    hw = 224
    batch_size = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((hw, hw)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_a = CustomDataset("autism/images/train/autistic", hw, transform=transform)
    train_na = CustomDataset("autism/images/train/non_autistic", hw, transform=transform)
    train_dataset = ConcatDataset([train_a, train_na])

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
                for epoch in range(10):
                    model.train()
                    total_loss, train_correct, train_total = 0, 0, 0
                    for images, labels in train_loader:
                        images, labels = images.to(device), labels.to(device).float()
                        optimizer.zero_grad()
                        outputs = model(images)
                        loss = criterion(outputs.squeeze(1), labels)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()

                        # Calculate training accuracy
                        predictions = (outputs.squeeze(1) > 0.5).float()
                        train_correct += (predictions == labels).sum().item()
                        train_total += labels.size(0)

                    train_accuracy = train_correct / train_total * 100

                    # Validation
                    model.eval()
                    val_loss, val_correct, val_total = 0, 0, 0
                    with torch.no_grad():
                        for images, labels in val_loader:
                            images, labels = images.to(device), labels.to(device).float()
                            outputs = model(images)
                            val_loss += criterion(outputs.squeeze(1), labels).item()
                            predictions = (outputs.squeeze(1) > 0.5).float()
                            val_correct += (predictions == labels).sum().item()
                            val_total += labels.size(0)
                    val_accuracy = val_correct / val_total * 100

                    print(f"Epoch {epoch + 1}, Loss: {total_loss:.5f}, Accuracy: {train_accuracy:.5f}%, Validation Loss: {val_loss:.5f}, Validation Accuracy: {val_accuracy:.5f}%")


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
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, 'autism/neural_networks/model.pth')
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
                checkpoint = torch.load(os.path.join(models_path, models[choice]))
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
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
