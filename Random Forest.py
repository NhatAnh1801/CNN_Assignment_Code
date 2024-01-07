from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import cifar10
import numpy as np
import time

# Load CIFAR-10 data
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Flatten the images
train_images_flattened = train_images.reshape(train_images.shape[0], -1)
test_images_flattened = test_images.reshape(test_images.shape[0], -1)

# Normalize the images
train_images_flattened = train_images_flattened / 255.0
test_images_flattened = test_images_flattened / 255.0

# Apply PCA for dimensionality reduction
pca = PCA(n_components=0.95)
train_images_pca = pca.fit_transform(train_images_flattened)
test_images_pca = pca.transform(test_images_flattened)

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=0)

# Start timing
start_time = time.time()

# Train the model
rf_classifier.fit(train_images_pca, train_labels.ravel())

# Predict on the test set
rf_predictions = rf_classifier.predict(test_images_pca)

# Calculate accuracy
rf_accuracy = accuracy_score(test_labels.ravel(), rf_predictions)

# End timing
end_time = time.time()

# Print time and accuracy
total_time = end_time - start_time
print(f'Random Forest Training and Prediction Time: {total_time:.2f} seconds')
print(f'Random Forest Test Accuracy: {rf_accuracy:.4f}')

