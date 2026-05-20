import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class WeatherTrainer:
    def __init__(self, model, train_loader, test_loader, criterion, optimizer, scheduler):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, epochs: int):
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            for sequences, target_diffs, _, _ in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(sequences)
                loss = self.criterion(outputs.squeeze(), target_diffs)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                total_loss += loss.item()
            
            self.scheduler.step()
            avg_loss = total_loss / len(self.train_loader)
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Époque {epoch+1}/{epochs} - Perte : {avg_loss:.4f} - LR : {current_lr:.6f}")

    def evaluate(self, diff_mean, diff_std, tolerance_1=1.0, tolerance_2=2.0):
        self.model.eval()
        predictions = []
        actuals = []

        with torch.no_grad():
            for sequences, _, last_actuals, next_actuals in self.test_loader:
                outputs = self.model(sequences).squeeze()
                pred_diffs = (outputs.numpy() * diff_std) + diff_mean
                preds = last_actuals.numpy() + pred_diffs
                
                predictions.extend(preds)
                actuals.extend(next_actuals.numpy())

        actuals_arr = np.array(actuals)
        predictions_arr = np.array(predictions)
        errors = np.abs(actuals_arr - predictions_arr)

        success_rate_1 = np.mean(errors <= tolerance_1) * 100
        success_rate_2 = np.mean(errors <= tolerance_2) * 100

        print(f"\n--- Résultats de l'évaluation ---")
        print(f"Précision à ±{tolerance_1}°C : {success_rate_1:.2f}%")
        print(f"Précision à ±{tolerance_2}°C : {success_rate_2:.2f}%")
        print(f"Erreur moyenne absolue (MAE) : {np.mean(errors):.2f}°C\n")

        plt.figure(figsize=(12, 6))
        plt.plot(actuals[:200], label='Réalité', color='blue')
        plt.plot(predictions[:200], label='Prédictions CNN-LSTM', color='red', linestyle='--')
        plt.title('Prédictions vs Réalité (200 premiers jours de test)')
        plt.xlabel('Jours')
        plt.ylabel('Température (°C)')
        plt.legend()
        plt.savefig("predictions.png")