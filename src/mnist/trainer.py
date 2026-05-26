import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt

class MNISTTrainer:
    """
    Handles the training, evaluation and saving loop for the MNIST model.
    Uses CrossEntropyLoss for classification and Adam optimizer.
    """
    def __init__(self, model, train_loader, test_loader, lr=0.001):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.best_accuracy = 0.0
        self.train_losses = []
        self.test_accuracies = []

    def train(self, epochs: int):
        """
        Trains the model for a specified number of epochs.
        Iterates through the training DataLoader, computes loss, performs backpropagation, 
        updates weights, and triggers evaluation after each epoch.
        """
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            for images, labels in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.train_loader)
            self.train_losses.append(avg_loss)
            print(f"Époque {epoch+1}/{epochs} - Perte moyenne : {avg_loss:.4f}")
            
            self.evaluate()

    def evaluate(self):
        """
        Evaluates the model on the test dataset without tracking gradients (for memory efficiency).
        Calculates accuracy based on max logit probabilities.
        Automatically saves the model if the current accuracy surpasses the best recorded accuracy.
        """
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        self.test_accuracies.append(accuracy)
        print(f"Précision sur le jeu de test : {accuracy:.2f}%")
        
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self._save_model()

    def save_plot(self, filename="learning_curve.png"):
        epochs = range(1, len(self.train_losses) + 1)
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = 'tab:red'
        ax1.set_xlabel('Époques')
        ax1.set_ylabel('Perte (Loss)', color=color)
        ax1.plot(epochs, self.train_losses, color=color, marker='o', label='Perte d\'entraînement')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Précision (%)', color=color)
        ax2.plot(epochs, self.test_accuracies, color=color, marker='s', label='Précision de test')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title("Évolution de la Perte et de la Précision", pad=20)
        fig.tight_layout()
        
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f"Graphique sauvegardé sous {filename}")

    def _save_model(self):
        """
        Saves the current best model to the filesystem.
        Creates a 'models' directory if it doesn't exist.
        """
        os.makedirs("models", exist_ok=True)
        torch.save(self.model.state_dict(), "models/best_mnist_model.pth")
        print("Nouveau meilleur modèle sauvegardé dans models/best_mnist_model.pth.")