import torch
import torch.nn as nn
import torch.optim as optim
import os

class MNISTTrainer:
    def __init__(self, model, train_loader, test_loader, lr=0.001):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.best_accuracy = 0.0

    def train(self, epochs: int):
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
            print(f"Époque {epoch+1}/{epochs} - Perte moyenne : {avg_loss:.4f}")
            
            self.evaluate()

    def evaluate(self):
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
        print(f"Précision sur le jeu de test : {accuracy:.2f}%")
        
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self._save_model()

    def _save_model(self):
        os.makedirs("models", exist_ok=True)
        torch.save(self.model.state_dict(), "models/best_mnist_model.pth")
        print("Nouveau meilleur modèle sauvegardé dans models/best_mnist_model.pth.")