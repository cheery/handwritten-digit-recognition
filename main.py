
import pygame
import torch
from torchvision import transforms
from cnn_model import CNNModel
from PIL import Image, ImageOps

model_save_path = "trained_model.pt"
model = CNNModel()
model.load_state_dict(torch.load(model_save_path))
model.eval()

def crop_image(image):
    inverted_image = ImageOps.invert(image.convert('RGB'))
    box = inverted_image.getbbox()
    cropped_image = image.crop(box)
    return cropped_image

def predict_digit(surface):
    # Apply transformations to the surface
    surface = pygame.surfarray.array3d(
        pygame.transform.rotate(
            pygame.transform.flip(surface, True, False), 90))
    surface = surface[:, :, 0]  # Convert to grayscale (all channels have the same value)

    # Preprocess the drawn image
    transform = transforms.Compose([
        transforms.Lambda(crop_image),  # Crop the image to remove unnecessary borders
        transforms.Lambda(lambda x: ImageOps.expand(x, border=50, fill=255)),  # Add padding to center the digit
        transforms.Resize((28, 28)),
        transforms.Lambda(lambda x: ImageOps.invert(x)),  # Invert colors
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    img_pil = Image.fromarray(surface)
    img_tensor = transform(img_pil)

    # Make a prediction
    with torch.no_grad():
        output = model(img_tensor.unsqueeze(0))
        prediction = output.argmax(dim=1).item()
    return prediction

pygame.init()
screen = pygame.display.set_mode((280, 280))
pygame.display.set_caption("Handwritten Digit Recognition")

screen.fill((255, 255, 255))

running = True
drawing = False
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
            digit_prediction = predict_digit(screen)
            print("Predicted digit:", digit_prediction)
            screen.fill((255, 255, 255))  # Clear the screen after making a prediction
            pygame.display.flip()

    if drawing:
        mouse_position = pygame.mouse.get_pos()
        pygame.draw.circle(screen, (0, 0, 0), mouse_position, 7)
    pygame.display.flip()

pygame.quit()
