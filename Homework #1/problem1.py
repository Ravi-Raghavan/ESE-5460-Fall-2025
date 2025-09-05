# Ravi Raghavan, Homework #1, Problem 1

## Part (c)

### Import numpy and set random seed for reproducability
import numpy as np
np.random.seed(42)

### Download the dataset using the given code
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
ds = fetch_openml("mnist_784", as_frame=False)
x, x_test, y, y_test = train_test_split(ds.data, ds.target, test_size=0.2, random_state=42)

m_non_test = x.shape[0]
m_test = x_test.shape[0]

### Get the max/min of x and x_test. Will proceed to convert to uint8
assert np.max(x) == 255
assert np.min(x) == 0
assert np.max(x_test) == 255
assert np.min(x_test) == 0

### Convert x/x_test to uint8
x = x.astype('uint8')
x_test = x_test.astype('uint8')

### Downsample x and x_test from 28 x 28 to 14 x 14
import cv2
def downsample_images(images):
    downsampled_images = []
    for idx in range(images.shape[0]):
        flattened_img = images[idx, :].flatten()
        img_reshaped = flattened_img.reshape(28, 28)
        img_downsampled = cv2.resize(img_reshaped, (14, 14))
        downsampled_images.append(img_downsampled.flatten())
    return np.vstack(downsampled_images)

x = downsample_images(x)
x_test = downsample_images(x_test)

assert x.shape[0] == m_non_test
assert x.shape[1] == 14 * 14
assert x_test.shape[0] == m_test
assert x_test.shape[1] == 14 * 14
assert x.ndim == 2
assert x_test.ndim == 2

### Randomly subsample x to create dataset of size 10000 with 1000 samples per label (subsample y accordingly)
#### First select the indices that we are going to use for the subsampling
chosen_indices = []
for label in np.unique(y):
    indices = np.where(y == label)[0]
    selected_indices = np.sort(np.random.choice(indices, size=1000, replace=False)).tolist()
    chosen_indices.extend(selected_indices)

#### Then, subsample x and y given the above indices 
chosen_indices = np.array(chosen_indices)
x = x[chosen_indices]
y = y[chosen_indices]

### Use train_test_split to split x and y into train and validation sets (80% train, 20% validation)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

## Part (d)
from sklearn import svm
classifier = svm.SVC(C = 1.0, kernel = 'rbf', gamma = 'auto')

### Fit to training data
classifier.fit(x_train, y_train)

### Predict labels of training data using trained classifier
y_train_pred = classifier.predict(x_train)

### Predict labels of validation data using trained classifier
y_val_pred = classifier.predict(x_val)

### Run the classifier on x_test,y_test
y_test_pred = classifier.predict(x_test)

### Training Error
from sklearn.metrics import accuracy_score
training_accuracy_score = accuracy_score(y_train, y_train_pred)
train_error = np.mean(y_train_pred != y_train)
print("Training Accuracy: ", training_accuracy_score)
print("Training Error: ", train_error)

### Validation Error
validation_accuracy_score = accuracy_score(y_val, y_val_pred)
val_error = np.mean(y_val_pred != y_val)
print("Validation Accuracy: ", validation_accuracy_score)
print("Validation Error: ", val_error)

### Compute ratio of the number of support samples to the total number of training samples for classifier
num_support_vectors = len(classifier.support_)
total_training_samples = x_train.shape[0]
support_vector_ratio = num_support_vectors / total_training_samples
print("Support Vector Ratio: ", support_vector_ratio)

### Report the classification error for Test Data
test_accuracy_score = accuracy_score(y_test, y_test_pred)
test_error = np.mean(y_test_pred != y_test)
print("Test Accuracy: ", test_accuracy_score)
print("Test Error: ", test_error)

### Report the 10-class confusion matrix on the test data
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix:\n", conf_matrix)

### "Do you notice any patterns about what kind of mistakes are being made?"
#### Answer: Based on the confusion matrix, it seems that our model is always predicting the class label of '9'
### " Can you explain these mistakes intuitively?"
#### Looking at our SVM hyperparameters and Training/Test outputs, we see that we had a large number of support vectors. This is a key indication of overfitting. 
#### 