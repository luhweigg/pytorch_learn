import sys
import torch
from torchvision import transforms
from PIL import Image, ImageOps
from src.mnist.model import CNN

def preprocess_custom_image(image_path: str) -> torch.Tensor:
    img = Image.open(image_path).convert('L')
    img = ImageOps.invert(img)

    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)

    ratio = 20.0 / max(img.size)
    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
    img = img.resize(new_size, Image.Resampling.LANCZOS)

    final_img = Image.new('L', (28, 28), 0)
    paste_pos = ((28 - img.size[0]) // 2, (28 - img.size[1]) // 2)
    final_img.paste(img, paste_pos)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    return transform(final_img).unsqueeze(0)

def predict_image(image_path: str, model_path: str = "models/best_mnist_model.pth"):
    model = CNN()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    img_tensor = preprocess_custom_image(image_path)

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output.data, 1)

    print(f"Prédiction pour {image_path} : {predicted.item()}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict_image(sys.argv[1])
    else:
        print("Veuillez fournir le chemin vers une image.")