import time
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import cifar10

# Start the overall timer
overall_start_time = time.time()

# Load and normalize CIFAR-10 data
start_time = time.time()
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
print("Data loading and normalization completed in {:.2f} seconds".format(time.time() - start_time))

# Flatten the images
start_time = time.time()
train_images_flattened = train_images.reshape(train_images.shape[0], -1)
test_images_flattened = test_images.reshape(test_images.shape[0], -1)
print("Data flattening completed in {:.2f} seconds".format(time.time() - start_time))

# PCA for dimensionality reduction
start_time = time.time()
pca = PCA(n_components=0.95)  # Retain 95% of variance
train_images_pca = pca.fit_transform(train_images_flattened)
test_images_pca = pca.transform(test_images_flattened)
print("PCA transformation completed in {:.2f} seconds".format(time.time() - start_time))

# Create an SVM model
svm_model = svm.SVC(kernel='linear')

# Train the model
start_time = time.time()
svm_model.fit(train_images_pca, train_labels.ravel())
print("SVM training completed in {:.2f} seconds".format(time.time() - start_time))

# Predict on the test set
start_time = time.time()
predicted_labels = svm_model.predict(test_images_pca)
print("Prediction completed in {:.2f} seconds".format(time.time() - start_time))

# Calculate accuracy
accuracy = accuracy_score(test_labels.ravel(), predicted_labels)
print(f'Test Accuracy: {accuracy}')

# Print the total time taken
print("Total process completed in {:.2f} seconds".format(time.time() - overall_start_time))
