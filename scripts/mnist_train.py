from src.mnist.model import CNN
from src.mnist.dataset import get_dataloaders
from src.mnist.trainer import MNISTTrainer

def main():
    train_loader, test_loader = get_dataloaders(batch_size=256)
    model = CNN()
    
    trainer = MNISTTrainer(model, train_loader, test_loader, lr=0.0001)
    trainer.train(epochs=10)
    trainer.evaluate()

if __name__ == "__main__":
    main()