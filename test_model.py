import torch
import matplotlib.pyplot as plt
from basic_DCGAN_ARCHITECTURE import Generator  # Make sure this is your generator class file

# Set parameters (use the same values as during training)
Z_DIM = 100  # Dimension of the noise vector
CHANNELS_IMG = 3  # Number of image channels (3 for RGB)
FEATURES_GEN = 64  # Generator feature map size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the Generator model
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
gen.load_state_dict(torch.load("saved_models/generator_epoch_5.pth"))  # Replace X with the correct epoch number
gen.eval()  # Set to evaluation mode

# Generate a batch of random noise vectors for 16 images
num_images = 16  # Number of images for the grid (4x4)
with torch.no_grad():
    noise = torch.randn(num_images, Z_DIM, 1, 1).to(device)  # Generate 16 noise vectors
    fake_images = gen(noise)

# Normalize images from [-1, 1] to [0, 1] for display
fake_images = (fake_images + 1) / 2

# Create a 4x4 grid plot
fig, axs = plt.subplots(4, 4, figsize=(8, 8))  # 4x4 grid, each image 2x2 inches

for i in range(4):
    for j in range(4):
        # Get the image from the batch and format it for display
        img = fake_images[i * 4 + j].cpu().permute(1, 2, 0).numpy()  # HWC format
        axs[i, j].imshow(img)
        axs[i, j].axis("off")  # Hide axes for a cleaner look

plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Optional: Adjust spacing between images
plt.show()
