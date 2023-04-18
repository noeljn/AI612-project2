import torch
import os
import random
import pickle

def generate_new_data(data_path, num_samples=1000):
    os.makedirs(data_path, exist_ok=True)

    # Generate synthetic features
    features = [torch.randn(12800) for _ in range(num_samples)]

    # Generate synthetic labels
    labels = []
    for _ in range(num_samples):
        label = []
        label.extend([random.randint(0, 1) for _ in range(3)])  # Mortality (short, long) and Readmission
        label.extend([random.randint(0, 1) for _ in range(17)]) # 17 different binary diagnoses
        label.extend([random.randint(0, 2) for _ in range(2)])  # Length of stay (short, long)
        label.extend([random.randint(0, 5) for _ in range(2)])  # Final acuity (6), Imminent discharge (6)
        label.extend([random.randint(0, 4) for _ in range(4)])  # Creatinine, Bilirubin, Platelet, and WBC levels
        labels.append(label)

    # Save features and labels with torch.save
    torch.save(features, os.path.join(data_path, 'features.pkl'))
    torch.save(labels, os.path.join(data_path, 'labels.pkl'))
if __name__ == "__main__":
    data_path = '/Users/noelkj/Documents/GitHub/AI612-project2/train/testdata/'
    generate_new_data(data_path)
