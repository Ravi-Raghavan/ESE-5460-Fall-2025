### Import numpy and set random seed for reproducability
import numpy as np
np.random.seed(42)

### Download the dataset using the given code
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
ds = fetch_openml("mnist_784", as_frame=False)
x, x_test, y, y_test = train_test_split(ds.data, ds.target, test_size=0.2, random_state=42)

### Use train_test_split to split x and y into train and validation sets (80% train, 20% validation)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)


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
