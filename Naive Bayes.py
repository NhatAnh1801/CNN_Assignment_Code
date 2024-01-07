from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np

# Load CIFAR-10 data
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Flatten the images
train_images_flattened = train_images.reshape(train_images.shape[0], -1)
test_images_flattened = test_images.reshape(test_images.shape[0], -1)

# Create a Gaussian Naive Bayes model
gnb = GaussianNB()

# Train the model
gnb.fit(train_images_flattened, train_labels.ravel())

# Predict on the test set
predicted_labels = gnb.predict(test_images_flattened)

# Calculate accuracy
accuracy = accuracy_score(test_labels.ravel(), predicted_labels)
print(f'Test Accuracy: {accuracy}')

# Plotting
plt.figure(figsize=(6, 4))
plt.bar(['Gaussian Naive Bayes'], [accuracy], color='blue')
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.title('Test Accuracy of Gaussian Naive Bayes on CIFAR-10')
plt.show()
