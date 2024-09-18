import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image, ImageFile
import os
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import random
import time
from datetime import datetime

# Parameters to tweak
batch_size = 64
learning_rate = 1e-3
num_epochs = 50
checkpoint_interval = 25
max_images_per_class = 25000 # limit dataset size (we have 25k per class), go higher with bigger ResNet model
resnet_model = 'ResNet50'  # ResNet18 or 50 (add 34, 101?)

# Directories
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
identifier = f"softmax-{resnet_model}_{num_epochs}-ep_{batch_size}-bs_{max_images_per_class}-images_{current_time}"
class_names = ['Boston', 'Charlotte', 'Manhattan', 'Pittsburgh']
folders = {
    'Boston': '/work/re-blocking/data/ma-boston/buildings',
    'Charlotte': '/work/re-blocking/data/nc-charlotte/buildings',
    'Manhattan': '/work/re-blocking/data/ny-manhattan/buildings',
    'Pittsburgh': '/work/re-blocking/data/pa-pittsburgh/buildings'
}
output_folder = os.path.join('/work/re-blocking/ensemble/softmax-output', identifier)
checkpoint_dir = os.path.join(output_folder, 'checkpoints')
model_save_path = os.path.join(output_folder, f'trained-model_{identifier}.pth')
loss_log_path = os.path.join(output_folder, f'loss-log_{identifier}.json')
training_curves_path = os.path.join(output_folder, f'training-curves_{identifier}.png')
confusion_matrix_path = os.path.join(output_folder, f'confusion-matrix_{identifier}.png')
cross_validation_path = os.path.join(output_folder, f'cross-validation_{identifier}.png')
misclassified_samples_path = os.path.join(output_folder, f'misclassified-samples_{identifier}.png')
report_path = os.path.join(output_folder, f'report_{identifier}.txt')
new_image_path = '/work/re-blocking/data/ny-brooklyn/buildings/buildings_1370.jpg'
predictions_output_file = os.path.join(output_folder, f'predictions_{identifier}.txt')

# More Parameters
normalize_mean = [0.485, 0.456, 0.406] # mean pixel values of the ImageNet dataset
normalize_std = [0.229, 0.224, 0.225] # standard deviation of pixel values in the ImageNet dataset
num_classes = len(class_names)
weight_decay = 1e-4  # Changed from 1e-3 to 1e-4 for regularization

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Define output folder
os.makedirs(output_folder, exist_ok=True)

# Define a custom dataset class
class CityDataset(Dataset):
    def __init__(self, folders, transform=None, max_images_per_class=max_images_per_class):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(folders.keys())}

        for class_name, folder in folders.items():
            class_images = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
            selected_images = random.sample(class_images, min(max_images_per_class, len(class_images)))
            self.image_paths.extend(selected_images)
            self.labels.extend([self.class_to_idx[class_name]] * len(selected_images))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

# Define transformations with data augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomCrop(224, padding=4),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=normalize_mean, std=normalize_std),
])

# Create dataset
dataset = CityDataset(folders, transform=transform)

# Set device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Load a pre-trained ResNet model based on the parameter
if resnet_model == 'ResNet18':
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
elif resnet_model == 'ResNet50':
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
else:
    raise ValueError(f"Unsupported ResNet model: {resnet_model}")

# Modify the final layer to match the number of classes
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, num_classes)
)
model.to(device)

# Save training parameters
training_params = {
    "identifier": identifier,
    "model": resnet_model,
    "device": str(device).split(':')[0],
    "max_images_per_class": max_images_per_class,
    "num_classes": num_classes,
    "class_names": class_names,
    "batch_size": batch_size,
    "num_epochs": num_epochs,
    "learning_rate": learning_rate,
    "checkpoint_interval": checkpoint_interval,
    "normalize_mean": normalize_mean,
    "normalize_std": normalize_std
}
params_path = os.path.join(output_folder, f'training-params_{identifier}.json')
with open(params_path, 'w') as f:
    json.dump(training_params, f, indent=4)  # Add indent parameter for readability
print(f"Training parameters saved to {params_path}")

# Model training function
def train_and_save_model(model, train_loader, val_loader, num_epochs, checkpoint_interval, checkpoint_dir):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    train_loss_log = []
    val_loss_log = []
    val_accuracy_log = []
    epoch_times = []

    # Early stopping parameters
    patience = 10
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    total_iterations = num_epochs * len(train_loader)
    progress_bar = tqdm(total=total_iterations, desc="Training Progress")

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # Training
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            progress_bar.update(1)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_loss_log.append(epoch_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_accuracy = correct / total
        val_loss_log.append(val_epoch_loss)
        val_accuracy_log.append(val_accuracy)

        # Learning rate scheduling
        scheduler.step(val_epoch_loss)

        # Early stopping check
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)

        progress_bar.set_postfix({
            'Epoch': f'{epoch + 1}/{num_epochs}',
            'Train Loss': f'{epoch_loss:.4f}',
            'Val Loss': f'{val_epoch_loss:.4f}',
            'Val Accuracy': f'{val_accuracy:.4f}',
            'Epoch Time (s)': f'{epoch_duration:.2f}'
        })

        # Save checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), checkpoint_path)

    progress_bar.close()

    # Save the final model weights
    torch.save(model.state_dict(), model_save_path)

    # Save the loss and accuracy logs
    with open(loss_log_path, 'w') as f:
        json.dump({
            'train_loss': train_loss_log,
            'val_loss': val_loss_log,
            'val_accuracy': val_accuracy_log,
            'epoch_times': epoch_times
        }, f)

    # Plot the loss and accuracy curves
    plot_training_curves(train_loss_log, val_loss_log, val_accuracy_log)

    return train_loss_log, val_loss_log, val_accuracy_log

def plot_training_curves(train_loss_log, val_loss_log, val_accuracy_log):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_loss_log) + 1), train_loss_log, label='Train Loss')
    plt.plot(range(1, len(val_loss_log) + 1), val_loss_log, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(val_accuracy_log) + 1), val_accuracy_log)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.savefig(training_curves_path)
    plt.close()

def k_fold_cross_validation(dataset, num_folds=3):
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_results = []
    fold_times = []
    all_labels = []
    all_predictions = []
    all_train_loss_logs = []
    all_val_loss_logs = []
    all_val_accuracy_logs = []

    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset), 1):
        print(f"Fold {fold}")
        fold_start_time = time.time()
        
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
        
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler)
        
        if resnet_model == 'ResNet18':
            model = models.resnet18(weights=weights)
        elif resnet_model == 'ResNet50':
            model = models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported ResNet model: {resnet_model}")
        
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.fc.in_features, num_classes)
        )
        model.to(device)
        
        train_loss_log, val_loss_log, val_accuracy_log = train_and_save_model(
            model, train_loader, val_loader, num_epochs, checkpoint_interval, 
            os.path.join(checkpoint_dir, f'fold_{fold}')
        )
        
        all_train_loss_logs.append(train_loss_log)
        all_val_loss_logs.append(val_loss_log)
        all_val_accuracy_logs.append(val_accuracy_log)
        
        model.eval()
        correct = 0
        total = 0
        fold_labels = []
        fold_predictions = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                fold_labels.extend(labels.cpu().numpy())
                fold_predictions.extend(predicted.cpu().numpy())
        
        accuracy = correct / total
        fold_results.append(accuracy)
        all_labels.extend(fold_labels)
        all_predictions.extend(fold_predictions)
        fold_end_time = time.time()
        fold_duration = fold_end_time - fold_start_time
        fold_times.append(fold_duration)
        print(f"Fold {fold} accuracy: {accuracy:.4f}, Time: {fold_duration:.2f} seconds")
    
    average_accuracy = sum(fold_results) / len(fold_results)
    print(f"Average accuracy across folds: {average_accuracy:.4f}")
    print(f"Average time per fold: {sum(fold_times) / len(fold_times):.2f} seconds")

    # Plot cross-validation results
    plot_cross_validation_results(fold_results, average_accuracy, num_folds)

    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_predictions)

    # Plot misclassified samples
    plot_misclassified_samples(dataset, all_labels, all_predictions)

    # Generate classification report
    generate_classification_report(all_labels, all_predictions)

    # Calculate and save additional metrics
    calculate_additional_metrics(all_train_loss_logs, all_val_loss_logs)

def plot_cross_validation_results(fold_results, average_accuracy, num_folds):
    plt.figure()
    plt.bar(range(1, num_folds + 1), fold_results, tick_label=[f'Fold {i}' for i in range(1, num_folds + 1)])
    plt.axhline(y=average_accuracy, color='r', linestyle='--', label=f'Average Accuracy: {average_accuracy:.4f}')
    plt.title('Cross-Validation Accuracy')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(cross_validation_path)
    plt.close()

def plot_confusion_matrix(all_labels, all_predictions):
    cm = confusion_matrix(all_labels, all_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig(confusion_matrix_path)
    plt.close()

def plot_misclassified_samples(dataset, all_labels, all_predictions):
    misclassified_indices = [i for i, (label, pred) in enumerate(zip(all_labels, all_predictions)) if label != pred]
    if misclassified_indices:
        plt.figure(figsize=(12, 12))
        for i, idx in enumerate(misclassified_indices[:16]):  # Show up to 16 misclassified samples
            image, label = dataset[idx]
            plt.subplot(4, 4, i + 1)
            plt.imshow(image.permute(1, 2, 0).numpy())
            plt.title(f'True: {class_names[label]}, Pred: {class_names[all_predictions[idx]]}')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(misclassified_samples_path)
        plt.close()

def generate_classification_report(all_labels, all_predictions):
    report = classification_report(all_labels, all_predictions, target_names=class_names)
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Classification report saved to {report_path}")

def calculate_additional_metrics(train_losses, val_losses, val_accuracies):
    """
    Calculate additional metrics such as convergence rate, overfitting score, and learning plateau.
    """
    # Calculate convergence rate
    convergence_rate = (train_losses[-1] - train_losses[0]) / len(train_losses)
    
    # Calculate overfitting score
    overfitting_score = val_losses[-1] - train_losses[-1]
    
    # Calculate learning plateau
    if len(val_losses) > 1:
        learning_plateau = val_losses[-1] - val_losses[-2]
    else:
        learning_plateau = 0
    
    return convergence_rate, overfitting_score, learning_plateau

# Example usage
train_losses = [0.8, 0.6, 0.4, 0.3, 0.2]
val_losses = [0.9, 0.7, 0.5, 0.35, 0.25]
val_accuracies = [0.7, 0.75, 0.8, 0.85, 0.9]

convergence_rate, overfitting_score, learning_plateau = calculate_additional_metrics(train_losses, val_losses, val_accuracies)

print(f"Convergence Rate: {convergence_rate:.4f}")
print(f"Overfitting Score: {overfitting_score:.4f}")
print(f"Learning Plateau: {learning_plateau:.4f}")

# Predict on a new image
def predict_image(model, image_path, transform):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
    
    return probabilities, predicted_class

# Predict on a new image
new_image_probabilities, new_image_class = predict_image(model, new_image_path, transform)

# Print and save predictions
predictions = [f"{class_names[i]}: {prob:.2f}" for i, prob in enumerate(new_image_probabilities)]
print(f"Predictions for {new_image_path}:")
print(f"Predicted class: {class_names[new_image_class]}")
print("Class probabilities:")
for pred in predictions:
    print(pred)

with open(predictions_output_file, 'w') as f:
    f.write(f"Predictions for {new_image_path}:\n")
    f.write(f"Predicted class: {class_names[new_image_class]}\n")
    f.write("Class probabilities:\n")
    for pred in predictions:
        f.write(f"{pred}\n")

print(f"Predictions saved to {predictions_output_file}")