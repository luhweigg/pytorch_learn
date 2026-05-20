import torch.nn as nn
import torch.optim as optim
from src.weather.model import WeatherCNNLSTM
from src.weather.dataset import get_weather_dataloaders
from src.weather.trainer import WeatherTrainer

def main():
    train_loader, test_loader, diff_mean, diff_std, input_size = get_weather_dataloaders(batch_size=32, sequence_length=30)
    model = WeatherCNNLSTM(input_size=input_size)
    
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    trainer = WeatherTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler
    )

    trainer.train(epochs=30)
    trainer.evaluate(diff_mean, diff_std)

if __name__ == "__main__":
    main()