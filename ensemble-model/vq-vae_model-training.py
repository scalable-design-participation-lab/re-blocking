import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import json
import csv

# Parameters
total_samples = 50000  # start low for testing, should be on +50k
batch_size = 128  # 1024
num_epochs = 20  # 10
learning_rate = 1e-4  # 1e-3
commitment_cost = 0.25
hidden_channels = 256  # 128
embedding_dim = 128  # 64
num_embeddings = 1024
checkpoint_interval = 50  # 50
image_size = (512, 512)
normalize_mean = (0.5,)
normalize_std = (0.5,)
identifier = f"vq-vae_{batch_size}-batch_{total_samples}-samples_{num_epochs}-epochs"  # unique identifier for this run

# Directories
output_dir = os.path.join('/work/re-blocking/ensemble/vae-output', identifier)  # All output will be within this directory
dataset_dirs = [
    '/work/re-blocking/data/ma-boston/parcels',
    '/work/re-blocking/data/nc-charlotte/parcels',
    '/work/re-blocking/data/ny-manhattan/parcels',
    '/work/re-blocking/data/pa-pittsburgh/parcels'
]

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Define transformations
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(normalize_mean, normalize_std)
])

# Collect all image paths
all_image_paths = []
for dataset_dir in dataset_dirs:
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                all_image_paths.append(os.path.join(root, file))

# Randomly sample the images from the collected paths
sampled_image_paths = random.sample(all_image_paths, total_samples)

# Custom dataset to load images from the sampled paths
class SampledImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0  # Returning 0 as a placeholder label

# Create a dataset and dataloader for the sampled images
sampled_dataset = SampledImageDataset(sampled_image_paths, transform=transform)
dataloader = DataLoader(sampled_dataset, batch_size=batch_size, shuffle=True)

# VQ-VAE Model Definition
class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_embeddings, embedding_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, embedding_dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x

class Decoder(nn.Module):
    def __init__(self, embedding_dim, hidden_channels, out_channels):
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(embedding_dim, hidden_channels, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(hidden_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        return x

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)
        self.commitment_cost = commitment_cost

    def forward(self, x):
        flattened = x.view(-1, self.embedding_dim)
        distances = torch.cdist(flattened, self.embedding.weight)
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(encoding_indices).view(x.size())

        e_latent_loss = F.mse_loss(quantized.detach(), x)
        q_latent_loss = F.mse_loss(quantized, x.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = x + (quantized - x).detach()

        return quantized, loss, encoding_indices

class VQVAE(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_embeddings, embedding_dim, commitment_cost):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(in_channels, hidden_channels, num_embeddings, embedding_dim)
        self.decoder = Decoder(embedding_dim, hidden_channels, in_channels)
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

    def forward(self, x):
        encoded = self.encoder(x)
        quantized, vq_loss, _ = self.vq_layer(encoded)
        decoded = self.decoder(quantized)
        return decoded, vq_loss

# Set device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

model = VQVAE(in_channels=3, hidden_channels=hidden_channels, num_embeddings=num_embeddings,
              embedding_dim=embedding_dim, commitment_cost=commitment_cost).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Save training parameters
training_params = {
    "identifier": identifier,
    "total_samples": total_samples,
    "batch_size": batch_size,
    "num_epochs": num_epochs,
    "learning_rate": learning_rate,
    "commitment_cost": commitment_cost,
    "hidden_channels": hidden_channels,
    "embedding_dim": embedding_dim,
    "num_embeddings": num_embeddings,
    "checkpoint_interval": checkpoint_interval
}
params_path = os.path.join(output_dir, 'training_params.json')
with open(params_path, 'w') as f:
    json.dump(training_params, f)
print(f"Training parameters saved to {params_path}")

# Initialize CSV file for logging
log_path = os.path.join(output_dir, 'training_log.csv')
with open(log_path, 'w', newline='') as csvfile:
    fieldnames = ['epoch', 'loss']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

# Training Loop
total_iterations = num_epochs * len(dataloader)
progress_bar = tqdm(total=total_iterations, desc="Training Progress")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, _ in dataloader:
        images = images.to(device)

        reconstructed, vq_loss = model(images)
        recon_loss = criterion(reconstructed, images)
        loss = recon_loss + vq_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        progress_bar.update(1)
        progress_bar.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")

    avg_loss = running_loss / len(dataloader)
    progress_bar.set_postfix(Loss=avg_loss)

    with open(log_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'epoch': epoch + 1, 'loss': avg_loss})

    if (epoch + 1) % checkpoint_interval == 0:
        checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch + 1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

progress_bar.close()

# Save the final trained model
model_output_dir = os.path.join(output_dir, 'models')
os.makedirs(model_output_dir, exist_ok=True)
model_save_path = os.path.join(model_output_dir, f"vq-vae_model_{identifier}.pth")
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Generate and save images
img_output_dir = os.path.join(output_dir, 'images')
os.makedirs(img_output_dir, exist_ok=True)

model.eval()
with torch.no_grad():
    sample_images, _ = next(iter(dataloader))
    sample_images = sample_images.to(device)
    reconstructed, _ = model(sample_images)

progress_bar = tqdm(total=batch_size, desc="Generating Images")

for i in range(batch_size):
    img = reconstructed[i].cpu().numpy().transpose(1, 2, 0)
    img = (img - img.min()) / (img.max() - img.min())  # Normalize the image to [0, 1]

    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.axis('off')

    img_output_path = os.path.join(img_output_dir, f'output_image_{i}.png')
    plt.savefig(img_output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    progress_bar.update(1)

progress_bar.close()
print(f"Generated and saved {batch_size} images to {img_output_dir}")