# test.py
import torch
import os

data_path = '/Users/noelkj/Documents/GitHub/AI612-project2/train/testdata/'


def generate_random_data(num_samples, feature_size, num_classes):
    # Generate random feature vectors
    features = torch.randn(num_samples, feature_size)

    # Generate random labels
    labels = torch.randint(0, num_classes, (num_samples,))

    return features, labels

# Example usage
num_samples = 1000
feature_size = 12800
num_classes = 5

features, labels = generate_random_data(num_samples, feature_size, num_classes)

# Save the generated data to data_path
torch.save(features, os.path.join(data_path, 'features.pkl'))
torch.save(labels, os.path.join(data_path, 'labels.pkl'))
