# Ravi Raghavan, Homework #1, Problem 1

## Part (c)

### Import numpy and set random seed for reproducability
import numpy as np
random_state = 42
np.random.seed(random_state)

### Download the dataset using the given code
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
ds = fetch_openml("mnist_784", as_frame=False)
x, x_test, y, y_test = train_test_split(ds.data, ds.target, test_size=0.2, random_state=random_state)

### Convert x/x_test to float and normalize(i.e. convert from [0, 255] to [0, 1])
x = x.astype(np.float64) / 255
x_test = x_test.astype(np.float64) / 255

### Downsample x and x_test from 28 x 28 to 14 x 14
import cv2
def downsample_images(images):
    downsampled_images = []
    for idx in range(images.shape[0]):
        reshaped_image = images[idx, :].reshape(28, 28)
        img_downsampled = cv2.resize(reshaped_image, (14, 14))
        downsampled_images.append(img_downsampled.flatten())
    return np.vstack(downsampled_images)

x = downsample_images(x)
x_test = downsample_images(x_test)

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
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=random_state)

## Part (d)
from sklearn import svm
classifier = svm.SVC(C = 1.0, kernel = 'rbf', gamma = 'scale')

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
print("Training Accuracy: ", training_accuracy_score, " Training Error: ", train_error)

### Validation Error
validation_accuracy_score = accuracy_score(y_val, y_val_pred)
val_error = np.mean(y_val_pred != y_val)
print("Validation Accuracy: ", validation_accuracy_score, " Validation Error: ", val_error)

### Compute ratio of the number of support samples to the total number of training samples for classifier
num_support_vectors = len(classifier.support_)
total_training_samples = x_train.shape[0]
support_vector_ratio = num_support_vectors / total_training_samples
print("Support Vector Ratio: ", support_vector_ratio)

### Report the classification error for Test Data
test_accuracy_score = accuracy_score(y_test, y_test_pred)
test_error = np.mean(y_test_pred != y_test)
print("Test Accuracy: ", test_accuracy_score, " Test Error: ", test_error)

### Report the 10-class confusion matrix on the test data
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix:\n", conf_matrix)

## Part (g)
### Applying GridSearchCV on SVC 
from sklearn.model_selection import GridSearchCV

# Define model
model = svm.SVC(kernel="rbf", gamma="auto")

# Define parameter grid: try at least 5 different C values
param_grid = {"C": [0.01, 0.1, 1, 10, 100]}

# GridSearchCV with 5-fold CV on the training set
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="accuracy",
    cv=5,
    return_train_score=True,
    verbose = 2
)

# print("Fitting Grid Search on X_train + X_validation")
# X_combined = np.concatenate([x_train, x_val])
# y_combined = np.concatenate([y_train, y_val])
# grid_search.fit(X_combined, y_combined)

# print("Results from cross-validation on training + validation set:")
# for mean, params in zip(grid_search.cv_results_["mean_test_score"], grid_search.cv_results_["params"]):
#     print(f"C={params['C']}, CV Accuracy={mean:.4f}")


## Part (h)
### Randomly subsample x to create dataset of size 2000 with 200 samples per label [100 for train + 100 for validation]
#### First select the indices that we are going to use for the subsampling
chosen_indices = []
for label in np.unique(y):
    indices = np.where(y == label)[0]
    selected_indices = np.sort(np.random.choice(indices, size=200, replace=False)).tolist()
    chosen_indices.extend(selected_indices)

chosen_indices = np.array(chosen_indices)
x = x[chosen_indices]
y = y[chosen_indices]

print(x.shape, y.shape)

x_train, y_train, x_test, y_test = x[:1000], y[:1000], x[1000:], y[1000:]

from skimage.filters import gabor_kernel , gabor
import matplotlib.pyplot as plt
freq, theta, bandwidth = 0.1, np.pi/4, 1
gk = gabor_kernel(frequency=freq, theta=theta, bandwidth=bandwidth)
plt.figure(1); plt.clf(); plt.imshow(gk.real); plt.title("Real Gabor Coefficients"); plt.savefig("real_coeff_gabor.png")
plt.figure(2); plt.clf(); plt.imshow(gk.imag); plt.title("Imaginary Gabor Coefficients"); plt.savefig("img_coeff_gabor.png")

# convolve the input image with the kernel and get co-efficients
# we will use only the real part and throw away the imaginary
# part of the co-efficients
image = x[0].reshape((14,14))
plt.figure(3); plt.clf(); plt.title("Sample Image"); plt.imshow(image); plt.savefig("img_0.png")

coeff_real, _ = gabor(image, frequency=freq, theta=theta, bandwidth=bandwidth)
plt.figure(4); plt.clf(); plt.title("Real Coefficients of Gabor Convolution"); plt.imshow(coeff_real); plt.savefig("real_coeff_img_0.png")

## Part (j)
thetas = np.arange(0,np.pi,np.pi/4)
frequencies = np.arange(0.05,0.5,0.15)
bandwidths = np.arange(0.3,1,0.3)

# Create Filter Matrix where each row is a filter
T, F, B = np.meshgrid(thetas, frequencies, bandwidths, indexing="ij")
filters = np.column_stack([T.ravel(), F.ravel(), B.ravel()])
print(f"Total filters: {len(filters)}")  # should be 36

# Plotting filter bank
fig, axes = plt.subplots(len(thetas), len(frequencies) * len(bandwidths), figsize=(18, 6))
axes = axes.ravel()

for i, filter in enumerate(filters):
    theta, freq, bw = filter[0], filter[1], filter[2]
    kernel = np.real(gabor_kernel(frequency=freq, theta=theta, bandwidth=bw))
    axes[i].imshow(kernel, cmap='gray')
    axes[i].set_title(f"θ={theta:.2f}, f={freq:.2f}, bw={bw:.2f}", fontsize=8)
    axes[i].axis("off")

fig.suptitle("Gabor Filter Bank")  # global title
plt.tight_layout()
plt.savefig("gabor_filter_bank.png")

# Define function to extract gabor features given images, filter bank
def extract_gabor_features(images, filters, img_shape=(14, 14)):
    n_imgs = len(images)
    n_filters = len(filters)
    n_feats_per_filter = img_shape[0] * img_shape[1]

    # Pre-allocate output array
    feats = np.empty((n_imgs, n_filters * n_feats_per_filter), dtype=np.float32)

    for i, img in enumerate(images):
        print(f"Index: {i}")
        img_reshaped = img.reshape(img_shape)

        # Collect features for all filters
        img_feats = [
            gabor(img_reshaped, frequency=freq, theta=theta, bandwidth=bw)[0].ravel()
            for theta, freq, bw in filters
        ]

        feats[i, :] = np.concatenate(img_feats)

    return feats

# Extract gabor features for train + test
X_train_feats = extract_gabor_features(x_train, filters)
X_test_feats  = extract_gabor_features(x_test, filters)

print("Finished extracting Gabor Features")

# Run PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=200)   # reduce dimensionality if needed
X_train_feats = pca.fit_transform(X_train_feats)
X_test_feats = pca.transform(X_test_feats)

# Train SVC
clf = svm.SVC(kernel="linear")
clf.fit(X_train_feats, y_train)
