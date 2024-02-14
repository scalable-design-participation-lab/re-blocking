import os
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def compare_and_log(model_name, dataset_name, generated_images, ground_truth_images, log_file):
    # Assuming generated_images and ground_truth_images are lists of image paths
    assert len(generated_images) == len(ground_truth_images), "Number of generated and ground truth images must be the same"

    # Initialize lists to store evaluation results
    image_names = []
    mse_scores = []

    # Iterate through images and compare
    for gen_path, gt_path in zip(generated_images, ground_truth_images):
        gen_img = np.array(Image.open(gen_path))
        gt_img = np.array(Image.open(gt_path))

        # Calculate Mean Squared Error (MSE) between generated and ground truth images
        mse = np.sum((gen_img - gt_img) ** 2) / float(gen_img.shape[0] * gen_img.shape[1])

        # Append results to lists
        image_names.append(os.path.basename(gen_path))
        mse_scores.append(mse)

    # Write results to a log file
    with open(log_file, 'a') as file:
        file.write(f"Model: {model_name}, Dataset: {dataset_name}\n")
        for img_name, mse_score in zip(image_names, mse_scores):
            file.write(f"{img_name}: MSE = {mse_score}\n")
        file.write("\n")

def create_confusion_matrix(log_files):
    # Initialize confusion matrix
    confusion_matrix_data = np.zeros((len(log_files), len(log_files)))

    # Populate confusion matrix with relevant metrics (e.g., average MSE)
    for i, log_file in enumerate(log_files):
        # Extract relevant information from log file
        # You may need to modify this depending on the actual structure of your log files
        with open(log_file, 'r') as file:
            lines = file.readlines()
            mse_values = [float(line.split('=')[1]) for line in lines[1:-1]]  # Assuming MSE values are present in the log

        # Populate confusion matrix
        confusion_matrix_data[i, :] = mse_values

    # Visualize confusion matrix using seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix_data, annot=True, fmt=".4f", cmap="Blues",
                xticklabels=log_files, yticklabels=log_files)
    plt.xlabel("Datasets")
    plt.ylabel("Models")
    plt.title("Confusion Matrix")
    plt.show()

# Example usage:
model_name = "CycleGAN_Model_1"
dataset_name = "Test_Dataset_1"
generated_images = ["gen_image_1.png", "gen_image_2.png", "gen_image_3.png"]
ground_truth_images = ["gt_image_1.png", "gt_image_2.png", "gt_image_3.png"]
log_file = "evaluation_log.txt"

compare_and_log(model_name, dataset_name, generated_images, ground_truth_images, log_file)

# For multiple models and datasets, call compare_and_log for each combination
# Then, call create_confusion_matrix with a list of log files
log_files = ["evaluation_log_model1_dataset1.txt", "evaluation_log_model1_dataset2.txt",
             "evaluation_log_model2_dataset1.txt", "evaluation_log_model2_dataset2.txt"]

create_confusion_matrix(log_files)
