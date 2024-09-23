import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, Subset
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

class CityDataset(Dataset):
    def __init__(self, folders, transform=None, max_images_per_class=None):
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

class CityClassifier:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.device = self._set_device()
        self.model = self._initialize_model()
        self.transform = self._set_transform()
        self.dataset = self._create_dataset()
        
        # Allow loading of truncated images
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        
        # Generate identifier
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.config['identifier'] = f"softmax-{self.config['resnet_model']}_{self.config['num_epochs']}-ep_{self.config['batch_size']}-bs_{self.config['max_images_per_class']}-images_{current_time}"
        
        # Generate output paths
        self.config['output_folder'] = os.path.join(self.config['output_folder'], self.config['identifier'])
        self.config['checkpoint_dir'] = os.path.join(self.config['output_folder'], 'checkpoints')
        self.config['model_save_path'] = os.path.join(self.config['output_folder'], f'trained-model_{self.config["identifier"]}.pth')
        self.config['loss_log_path'] = os.path.join(self.config['output_folder'], f'loss-log_{self.config["identifier"]}.json')
        self.config['training_curves_path'] = os.path.join(self.config['output_folder'], f'training-curves_{self.config["identifier"]}.png')
        self.config['confusion_matrix_path'] = os.path.join(self.config['output_folder'], f'confusion-matrix_{self.config["identifier"]}.png')
        self.config['cross_validation_path'] = os.path.join(self.config['output_folder'], f'cross-validation_{self.config["identifier"]}.png')
        self.config['misclassified_samples_path'] = os.path.join(self.config['output_folder'], f'misclassified-samples_{self.config["identifier"]}.png')
        self.config['report_path'] = os.path.join(self.config['output_folder'], f'report_{self.config["identifier"]}.txt')
        self.config['predictions_output_file'] = os.path.join(self.config['output_folder'], f'predictions_{self.config["identifier"]}.txt')
        
        # Create output folder
        os.makedirs(self.config['output_folder'], exist_ok=True)

    def _set_device(self):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def _initialize_model(self):
        if self.config['resnet_model'] == 'ResNet18':
            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
        elif self.config['resnet_model'] == 'ResNet50':
            weights = models.ResNet50_Weights.DEFAULT
            model = models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported ResNet model: {self.config['resnet_model']}")
        
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.fc.in_features, self.config['num_classes'])
        )
        return model.to(self.device)

    def _set_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config['normalize_mean'], std=self.config['normalize_std']),
        ])

    def _create_dataset(self):
        return CityDataset(
            self.config['folders'],
            transform=self.transform,
            max_images_per_class=self.config['max_images_per_class']
        )

    def train_and_save_model(self, train_loader, val_loader):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        train_loss_log, val_loss_log, val_accuracy_log = [], [], []
        epoch_times = []

        patience, best_val_loss = 2, float('inf')
        epochs_without_improvement = 0

        total_iterations = self.config['num_epochs'] * len(train_loader)
        progress_bar = tqdm(total=total_iterations, desc="Training Progress")

        for epoch in range(self.config['num_epochs']):
            epoch_start_time = time.time()

            # Training
            self.model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
                progress_bar.update(1)

            epoch_loss = running_loss / len(train_loader.dataset)
            train_loss_log.append(epoch_loss)

            # Validation
            self.model.eval()
            val_loss, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_epoch_loss = val_loss / len(val_loader.dataset)
            val_accuracy = correct / total
            val_loss_log.append(val_epoch_loss)
            val_accuracy_log.append(val_accuracy)

            scheduler.step(val_epoch_loss)

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
                'Epoch': f'{epoch + 1}/{self.config["num_epochs"]}',
                'Train Loss': f'{epoch_loss:.4f}',
                'Val Loss': f'{val_epoch_loss:.4f}',
                'Val Accuracy': f'{val_accuracy:.4f}',
                'Epoch Time (s)': f'{epoch_duration:.2f}'
            })

            if (epoch + 1) % self.config['checkpoint_interval'] == 0:
                self.save_checkpoint(epoch + 1)

        progress_bar.close()

        self.save_model()
        self.save_loss_log(train_loss_log, val_loss_log, val_accuracy_log, epoch_times)
        self.plot_training_curves(train_loss_log, val_loss_log, val_accuracy_log)

        return train_loss_log, val_loss_log, val_accuracy_log

    def save_checkpoint(self, epoch):
        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
        checkpoint_path = os.path.join(self.config['checkpoint_dir'], f'checkpoint_epoch_{epoch}.pth')
        torch.save(self.model.state_dict(), checkpoint_path)

    def save_model(self):
        torch.save(self.model.state_dict(), self.config['model_save_path'])

    def save_loss_log(self, train_loss_log, val_loss_log, val_accuracy_log, epoch_times):
        with open(self.config['loss_log_path'], 'w') as f:
            json.dump({
                'train_loss': train_loss_log,
                'val_loss': val_loss_log,
                'val_accuracy': val_accuracy_log,
                'epoch_times': epoch_times
            }, f)

    def plot_training_curves(self, train_loss_log, val_loss_log, val_accuracy_log):
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
        plt.savefig(self.config['training_curves_path'])
        plt.close()

    def k_fold_cross_validation(self, num_folds=2):
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        fold_results, fold_times = [], []
        all_labels, all_predictions = [], []
        all_train_loss_logs, all_val_loss_logs, all_val_accuracy_logs = [], [], []

        for fold, (train_ids, val_ids) in enumerate(kfold.split(self.dataset), 1):
            print(f"Fold {fold}")
            fold_start_time = time.time()
            
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
            
            train_loader = DataLoader(self.dataset, batch_size=self.config['batch_size'], sampler=train_subsampler)
            val_loader = DataLoader(self.dataset, batch_size=self.config['batch_size'], sampler=val_subsampler)
            
            self.model = self._initialize_model()
            
            train_loss_log, val_loss_log, val_accuracy_log = self.train_and_save_model(train_loader, val_loader)
            
            all_train_loss_logs.append(train_loss_log)
            all_val_loss_logs.append(val_loss_log)
            all_val_accuracy_logs.append(val_accuracy_log)
            
            self.model.eval()
            correct, total = 0, 0
            fold_labels, fold_predictions = [], []
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
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

        self.plot_cross_validation_results(fold_results, average_accuracy, num_folds)
        self.plot_confusion_matrix(all_labels, all_predictions)
        self.plot_misclassified_samples(all_labels, all_predictions)
        self.generate_classification_report(all_labels, all_predictions)
        self.calculate_additional_metrics(all_train_loss_logs, all_val_loss_logs)

    def plot_cross_validation_results(self, fold_results, average_accuracy, num_folds):
        plt.figure()
        plt.bar(range(1, num_folds + 1), fold_results, tick_label=[f'Fold {i}' for i in range(1, num_folds + 1)])
        plt.axhline(y=average_accuracy, color='r', linestyle='--', label=f'Average Accuracy: {average_accuracy:.4f}')
        plt.title('Cross-Validation Accuracy')
        plt.xlabel('Fold')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(self.config['cross_validation_path'])
        plt.close()

    def plot_confusion_matrix(self, all_labels, all_predictions):
        cm = confusion_matrix(all_labels, all_predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.config['class_names'])
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.savefig(self.config['confusion_matrix_path'])
        plt.close()

    def plot_misclassified_samples(self, all_labels, all_predictions):
        misclassified_indices = [i for i, (label, pred) in enumerate(zip(all_labels, all_predictions)) if label != pred]
        if misclassified_indices:
            plt.figure(figsize=(12, 12))
            for i, idx in enumerate(misclassified_indices[:16]):  # Show up to 16 misclassified samples
                image, label = self.dataset[idx]
                plt.subplot(4, 4, i + 1)
                plt.imshow(image.permute(1, 2, 0).numpy())
                plt.title(f'True: {self.config["class_names"][label]}, Pred: {self.config["class_names"][all_predictions[idx]]}')
                plt.axis('off')
            plt.tight_layout()
            plt.savefig(self.config['misclassified_samples_path'])
            plt.close()

    def generate_classification_report(self, all_labels, all_predictions):
        report = classification_report(all_labels, all_predictions, target_names=self.config['class_names'])
        with open(self.config['report_path'], 'w') as f:
            f.write(report)
        print(f"Classification report saved to {self.config['report_path']}")

    def calculate_additional_metrics(self, all_train_loss_logs, all_val_loss_logs):
        avg_train_loss_log = np.mean(all_train_loss_logs, axis=0)
        avg_val_loss_log = np.mean(all_val_loss_logs, axis=0)

        convergence_rate = (avg_train_loss_log[-1] - avg_train_loss_log[0]) / len(avg_train_loss_log)
        overfitting_score = (avg_val_loss_log[-1] - avg_train_loss_log[-1]) / avg_val_loss_log[-1]
        learning_plateau = np.mean(avg_val_loss_log[-5:]) - np.mean(avg_val_loss_log[:5])

        with open(self.config['report_path'], 'a') as f:
            f.write(f"\nConvergence Rate: {convergence_rate:.4f}\n")
            f.write(f"Overfitting Score: {overfitting_score:.4f}\n")
            f.write(f"Learning Plateau: {learning_plateau:.4f}\n")

    def predict_image(self, image_path):
        self.model.eval()
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image)
            probabilities = F.softmax(outputs, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
        
        return probabilities, predicted_class

    def save_predictions(self, image_path, probabilities, predicted_class):
        predictions = [f"{self.config['class_names'][i]}: {prob:.2f}" for i, prob in enumerate(probabilities)]
        print(f"Predictions for {image_path}:")
        print(f"Predicted class: {self.config['class_names'][predicted_class]}")
        print("Class probabilities:")
        for pred in predictions:
            print(pred)

        with open(self.config['predictions_output_file'], 'w') as f:
            f.write(f"Predictions for {image_path}:\n")
            f.write(f"Predicted class: {self.config['class_names'][predicted_class]}\n")
            f.write("Class probabilities:\n")
            for pred in predictions:
                f.write(f"{pred}\n")

        print(f"Predictions saved to {self.config['predictions_output_file']}")

def main(config_path):
    classifier = CityClassifier(config_path)
    print(f"Using device: {classifier.device}")
    
    # Save training parameters
    params_path = os.path.join(classifier.config['output_folder'], f'training-params_{classifier.config["identifier"]}.json')
    with open(params_path, 'w') as f:
        json.dump(classifier.config, f, indent=4)
    print(f"Training parameters saved to {params_path}")

    # Run k-fold cross-validation
    classifier.k_fold_cross_validation(num_folds=2)

    # Predict on a new image
    new_image_probabilities, new_image_class = classifier.predict_image(classifier.config['new_image_path'])
    classifier.save_predictions(classifier.config['new_image_path'], new_image_probabilities, new_image_class)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python city_classifier.py <config_path>")
        sys.exit(1)
    main(sys.argv[1])