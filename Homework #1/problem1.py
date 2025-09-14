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

### Convert x and x_test to np.float64 arrays
x = x.astype(np.float64)
x_test = x_test.astype(np.float64)

### Plot Sample Image
import matplotlib.pyplot as plt
a = x[0].reshape(28, 28)
plt.imshow(a)
plt.savefig("1c_sample_image.png", dpi=300, bbox_inches="tight")
plt.close()

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
chosen_indices = []
for label in np.unique(y):
    indices = np.where(y == label)[0]
    selected_indices = np.sort(np.random.choice(indices, size=1000, replace=False)).tolist()
    chosen_indices.extend(selected_indices)

chosen_indices = np.array(chosen_indices)
x = x[chosen_indices]
y = y[chosen_indices]

### Use train_test_split to split x and y into train and validation sets (80% train, 20% validation)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=random_state)

### Divide all pixel values by 255.0 to get values between 0 and 1
x_train, x_val, x_test = x_train / 255.0, x_val / 255.0, x_test / 255.0

### Standardize Features across training set and use those same mean/std for validation and test set
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_val = scaler.transform(x_val)
# x_test = scaler.transform(x_test)

##------------------------------------------------------------------------------------------------------------------------------------------------------------

## Part (d)
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd

### Define helper function to fit classifier and get necessary metrics
def fit_and_evaluate_svm(x_train, y_train, x_val, y_val, x_test, y_test, gamma = 'scale'):
    classifier = svm.SVC(C = 1.0, kernel = 'rbf', gamma = gamma)

    ### Fit to training data
    classifier.fit(x_train, y_train)

    ### Predict labels of training data using trained classifier
    y_train_pred = classifier.predict(x_train)

    ### Predict labels of validation data using trained classifier
    y_val_pred = classifier.predict(x_val)

    ### Run the classifier on x_test,y_test
    y_test_pred = classifier.predict(x_test)

    ### Training Error
    training_accuracy_score = accuracy_score(y_train, y_train_pred)
    train_error = np.mean(y_train_pred != y_train)

    ### Validation Error
    validation_accuracy_score = accuracy_score(y_val, y_val_pred)
    val_error = np.mean(y_val_pred != y_val)

    ### Compute ratio of the number of support samples to the total number of training samples for classifier
    num_support_vectors = len(classifier.support_)
    total_training_samples = x_train.shape[0]
    support_vector_ratio = num_support_vectors / total_training_samples

    ### Report the classification error for Test Data
    test_accuracy_score = accuracy_score(y_test, y_test_pred)
    test_error = np.mean(y_test_pred != y_test)

    ### Report the 10-class confusion matrix on the test data
    conf_matrix = confusion_matrix(y_test, y_test_pred, labels = classifier.classes_)

    metrics = {
        "Training Accuracy": training_accuracy_score,
        "Training Error": train_error,
        "Validation Accuracy": validation_accuracy_score,
        "Validation Error": val_error,
        "Support Vector Ratio": support_vector_ratio,
        "Test Accuracy": test_accuracy_score,
        "Test Error": test_error,
        "X.var()": x_train.var()
    }

    return metrics, conf_matrix, y_train_pred, y_val_pred, y_test_pred, classifier

### Call Function on scaled data with gamma = 'auto'
gamma_auto_metrics, conf_matrix, _, _, _, classifier = fit_and_evaluate_svm(x_train, y_train, x_val, y_val, x_test, y_test, gamma = 'auto')

#### Save Metrics
pd.DataFrame([gamma_auto_metrics]).to_csv("1d_gamma=auto.csv", index=False)

#### Save Confusion Matrix Plot
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels = classifier.classes_)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix: gamma = auto")
plt.savefig("1d_confusion_matrix_gamma=auto.png", dpi=300, bbox_inches="tight")
plt.close()

### Call Function on scaled data with gamma = 'scale'
gamma_scale_metrics, conf_matrix, _, _, y_test_pred, classifier = fit_and_evaluate_svm(x_train, y_train, x_val, y_val, x_test, y_test, gamma = 'scale')

#### Save Metrics
pd.DataFrame([gamma_scale_metrics]).to_csv("1d_gamma=scale.csv", index=False)

#### Save Confusion Matrix Plot
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels = classifier.classes_)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix: gamma = scale")
plt.savefig("1d_confusion_matrix_gamma=scale.png", dpi=300, bbox_inches="tight")
plt.close()

#### Error Analysis on Confusion Matrix
def plot_misclassified_sample(true_label, pred_label):
    mask = (y_test == true_label) & (y_test_pred == pred_label)
    indices = np.where(mask)[0]

    ##### Save test image plot
    random_index = indices[-1]
    sample_misclassified_test_image = x_test[random_index].reshape(14, 14)
    plt.imshow(sample_misclassified_test_image, cmap = "gray")
    plt.title(f"True Label = {true_label}, Predicted Label = {pred_label}")
    plt.savefig(f"1d_misclassified_test_image_true={true_label}_pred={pred_label}.png", dpi = 300, bbox_inches = "tight")
    plt.close()

#### Part 1: find all samples where true = 4, predicted = 9 and plot one of them for visual inspection
true_label = '4'
pred_label = '9'
plot_misclassified_sample(true_label, pred_label)

#### Part 2: find all samples where true = 3, predicted = 5 and plot one of them for visual inspection
true_label = '3'
pred_label = '5'
plot_misclassified_sample(true_label, pred_label)

##------------------------------------------------------------------------------------------------------------------------------------------------------------

# Part (g)
from sklearn.model_selection import GridSearchCV
model = svm.SVC(kernel="rbf", gamma="scale") # Use parameters: kernel = 'rbf', gamma = 'scale'

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

# Fit Grid Search on train + val combined
x_train_val_combined = np.concatenate([x_train, x_val])
y_train_val_combined = np.concatenate([y_train, y_val])
grid_search.fit(x_train_val_combined, y_train_val_combined)

# Extract results and save to CSV
results = []
for mean, params in zip(grid_search.cv_results_["mean_test_score"], grid_search.cv_results_["params"]):
    results.append({
        "C": params["C"],
        "CV_Accuracy": mean
    })

df = pd.DataFrame(results)
df.to_csv("1g_cv_results.csv", index=False)

##------------------------------------------------------------------------------------------------------------------------------------------------------------

## Part (h)
### Randomly subsample x to create dataset of size 2000 with 200 samples per label [100 for train + 100 for validation]
chosen_indices_training = []
chosen_indices_validation = []
for label in np.unique(y):
    indices = np.where(y == label)[0]
    assert len(indices) >= 200, f"Not enough samples for label {label}"
    selected_indices = np.sort(np.random.choice(indices, size=200, replace=False)).tolist()
    chosen_indices_training.extend(selected_indices[:100])
    chosen_indices_validation.extend(selected_indices[100:])

chosen_indices_training = np.array(chosen_indices_training)
chosen_indices_validation = np.array(chosen_indices_validation)

x_train, y_train, x_validation, y_validation = x[chosen_indices_training], y[chosen_indices_training], \
x[chosen_indices_validation], y[chosen_indices_validation]

#### Plot Gabor Coefficients[Real + Imaginary]
from skimage.filters import gabor_kernel , gabor
freq, theta, bandwidth = 0.1, np.pi/4, 1
gk = gabor_kernel(frequency=freq, theta=theta, bandwidth=bandwidth)
plt.figure(1); plt.clf(); plt.imshow(gk.real); plt.title("Real Gabor Coefficients"); plt.savefig("1h_real_coeff_gabor.png")
plt.figure(2); plt.clf(); plt.imshow(gk.imag); plt.title("Imaginary Gabor Coefficients"); plt.savefig("1h_img_coeff_gabor.png")

# Convolve the input image with the kernel and get co-efficients. Only use the real part
image = x[0].reshape((14,14))
plt.figure(3); plt.clf(); plt.title("Sample Image"); plt.imshow(image); plt.savefig("1h_img_0.png")
coeff_real, _ = gabor(image, frequency=freq, theta=theta, bandwidth=bandwidth)
plt.figure(4); plt.clf(); plt.title("Real Coefficients after Gabor Convolution"); plt.imshow(coeff_real); plt.savefig("1h_real_coeff_img_0.png")

##------------------------------------------------------------------------------------------------------------------------------------------------------------

## Part (j)
# Define thetas, frequencies, and bandwidths: Parameters of the Gabor Filters
thetas = np.arange(0,np.pi,np.pi/4)
frequencies = np.arange(0.05,0.5,0.15)
bandwidths = np.arange(0.3,1,0.3)

# Create Filter Matrix where each row is a filter
T, F, B = np.meshgrid(thetas, frequencies, bandwidths, indexing="ij")
filters = np.column_stack([T.ravel(), F.ravel(), B.ravel()])
print(f"Total filters: {len(filters)}")

# Plotting filter bank
fig, axes = plt.subplots(len(thetas), len(frequencies) * len(bandwidths), figsize=(18, 6))
axes = axes.ravel()
for i, filter in enumerate(filters):
    theta, freq, bw = filter[0], filter[1], filter[2]
    kernel = np.real(gabor_kernel(frequency=freq, theta=theta, bandwidth=bw))
    axes[i].imshow(kernel, cmap='gray')
    axes[i].set_title(f"θ={theta:.2f}, f={freq:.2f}, bw={bw:.2f}", fontsize=8)
    axes[i].axis("off")

fig.suptitle("Gabor Filter Bank: Filters = 36")
plt.tight_layout()
plt.savefig("1j_gabor_filter_bank_filters=36.png")

# Define function to extract gabor features given images, filter bank
def extract_gabor_features(images, filters, img_shape=(14, 14)):
    n_imgs = len(images)
    n_filters = len(filters)
    n_feats_per_filter = img_shape[0] * img_shape[1]

    # Pre-allocate output array
    feats = np.empty((n_imgs, n_filters * n_feats_per_filter), dtype=np.float64)

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

## Check if gabor features are saved, if not re-create them and save them
import os
if os.path.exists("x_train_gabor_feats_36.npy") and os.path.exists("x_validation_gabor_feats_36.npy"):
    x_train_gabor_feats = np.load("x_train_gabor_feats_36.npy")
    x_validation_gabor_feats = np.load("x_validation_gabor_feats_36.npy")
else:
    # Extract gabor features for train + test
    x_train_gabor_feats = extract_gabor_features(x_train, filters)
    x_validation_gabor_feats  = extract_gabor_features(x_validation, filters)

    ## Standardize Features across training set and use those same mean/std for validation set
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    x_train_gabor_feats = scaler.fit_transform(x_train_gabor_feats)
    x_validation_gabor_feats = scaler.transform(x_validation_gabor_feats)
    np.save("x_train_gabor_feats_36.npy", x_train_gabor_feats)
    np.save("x_validation_gabor_feats_36.npy", x_validation_gabor_feats)

print("Finished extracting Gabor Features from Train + Validation")

# Train SVC on Gabor Features
classifier = svm.SVC(kernel = 'rbf', C = 1.0, gamma = 'scale')
classifier.fit(x_train_gabor_feats, y_train)
y_train_pred = classifier.predict(x_train_gabor_feats)
y_val_pred = classifier.predict(x_validation_gabor_feats)
train_accuracy = accuracy_score(y_train, y_train_pred)
train_error = np.mean(y_train_pred != y_train)
val_accuracy = accuracy_score(y_validation, y_val_pred)
val_error = np.mean(y_val_pred != y_validation)
conf_matrix = confusion_matrix(y_validation, y_val_pred, labels = classifier.classes_)

# Save Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels = classifier.classes_)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix: Gabor Features with 36 filters")
plt.savefig("1j_confusion_matrix_gabor_features_filters=36.png", dpi=300, bbox_inches="tight")
plt.close()

## Store Results in CSV
results = {
    "Training Accuracy": train_accuracy,
    "Training Error": train_error,
    "Validation Accuracy": val_accuracy,
    "Validation Error": val_error,
    "X.var()": x_train_gabor_feats.var()
}

pd.DataFrame([results]).to_csv("1j_gabor_validation_results_filters=36.csv", index=False)

# Increase number of filters to 75 by adding more theta, frequency, and bandwidth values
thetas = np.arange(0, np.pi, np.pi / 5)  # 5 values
frequencies = np.arange(0.05, 0.5, 0.1)  # 5 values
bandwidths = np.arange(0.3, 1, 0.3)    # 3 values

# Create Filter Matrix where each row is a filter
T, F, B = np.meshgrid(thetas, frequencies, bandwidths, indexing="ij")
filters = np.column_stack([T.ravel(), F.ravel(), B.ravel()])
print(f"Total filters: {len(filters)}")  # should be 75

# Plotting filter bank
fig, axes = plt.subplots(len(thetas), len(frequencies) * len(bandwidths), figsize=(18, 6))
axes = axes.ravel()
for i, filter in enumerate(filters):
    theta, freq, bw = filter[0], filter[1], filter[2]
    kernel = np.real(gabor_kernel(frequency=freq, theta=theta, bandwidth=bw))
    axes[i].imshow(kernel, cmap='gray')
    axes[i].set_title(f"θ={theta:.2f}, f={freq:.2f}, bw={bw:.2f}", fontsize=8)
    axes[i].axis("off")

fig.suptitle("Gabor Filter Bank: Filters = 75")
plt.tight_layout()
plt.savefig("1j_gabor_filter_bank_filters=75.png")

# Extract gabor features for train + test
if os.path.exists("x_train_gabor_feats_75.npy") and os.path.exists("x_validation_gabor_feats_75.npy"):
    x_train_gabor_feats = np.load("x_train_gabor_feats_75.npy")
    x_validation_gabor_feats = np.load("x_validation_gabor_feats_75.npy")
else:
    # Extract gabor features for train + test
    x_train_gabor_feats = extract_gabor_features(x_train, filters)
    x_validation_gabor_feats  = extract_gabor_features(x_validation, filters)

    ## Standardize Features across training set and use those same mean/std for validation set
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    x_train_gabor_feats = scaler.fit_transform(x_train_gabor_feats)
    x_validation_gabor_feats = scaler.transform(x_validation_gabor_feats)
    np.save("x_train_gabor_feats_75.npy", x_train_gabor_feats)
    np.save("x_validation_gabor_feats_75.npy", x_validation_gabor_feats)

print("Finished extracting Gabor Features from Train + Validation")

# Run PCA on Training + Test Gabor Features
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95, random_state=random_state)
x_train_gabor_feats_pca = pca.fit_transform(x_train_gabor_feats)
x_validation_gabor_feats_pca = pca.transform(x_validation_gabor_feats)

# Train SVC on PCA transformed features
classifier = svm.SVC(kernel = 'rbf', C = 1.0, gamma = 'scale')
classifier.fit(x_train_gabor_feats_pca, y_train)
y_train_pred = classifier.predict(x_train_gabor_feats_pca)
y_val_pred = classifier.predict(x_validation_gabor_feats_pca)
train_accuracy = accuracy_score(y_train, y_train_pred)
train_error = np.mean(y_train_pred != y_train)
val_accuracy = accuracy_score(y_validation, y_val_pred)
val_error = np.mean(y_val_pred != y_validation)
conf_matrix = confusion_matrix(y_validation, y_val_pred, labels = classifier.classes_)

## Store Results in CSV
results = {
    "Training Accuracy": train_accuracy,
    "Training Error": train_error,
    "Validation Accuracy": val_accuracy,
    "Validation Error": val_error,
    "X.var()": x_train_gabor_feats.var()
}

pd.DataFrame([results]).to_csv("1j_gabor_pca_validation_results_filters=75.csv", index=False)
