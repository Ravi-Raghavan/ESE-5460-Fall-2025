# Ravi Raghavan, Homework #1, Problem 3

## Part (a)
import torchvision as thv
train = thv.datasets.MNIST("./", download=True, train=True)
val = thv.datasets.MNIST("./", download=True, train=False)
print(train.data.shape, len(train.targets))
print(val.data.shape, len(val.targets))

### Convert everything to numpy. From here on out, NO Torch/Other Deep Learning Library 
import numpy as np
X_train = train.data.numpy()
y_train = train.targets.numpy()
X_val = val.data.numpy()
y_val = val.targets.numpy()

print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

### Take 50% of each class from training and validation
def downsample(X, y):
    # Store downsampled X and y
    X_downsampled, y_downsampled = [], []

    # Iterate through the unique labels of y
    for label in np.unique(y):
        indices = np.where(y == label)[0] # Fetch indices where y == label
        indices = np.sort(indices)
        half = len(indices) // 2 # Get half
        X_downsampled.append(X[indices[:half]])
        y_downsampled.append(y[indices[:half]])
    
    return np.concatenate(X_downsampled, axis = 0), np.concatenate(y_downsampled, axis = 0)

X_train, y_train = downsample(X_train, y_train)
X_val, y_val = downsample(X_val, y_val)

print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

## Plot the images of a few images in the dataset just to see if label is right
# Function to plot a grid of images with labels
import matplotlib.pyplot as plt
def plot_images(X, y, start, plot):
    if not plot:
        return
    plt.figure(figsize=(8, 8))
    for i in range(start, start + 16):
        offset = i - start
        plt.subplot(4, 4, offset+1)
        plt.imshow(X[i], cmap='gray')
        plt.title(f"Label: {y[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Plot first 16 images from the downsampled training set
plot_images(X_train, y_train, 25000, False)
plot_images(X_val, y_val, 25000, False)


## Part (b)
class embedding_t:
    def __init__(self):
        # initialize to appropriate sizes, fill with Gaussian entries
        mean = 0.0
        std = 0.01
        self.w = np.random.normal(loc = mean, scale = std, size = (4, 4, 8))
        self.b = np.random.normal(loc = mean, scale = std, size = (8,))

        # normalize to make the Frobenius norm of (w, b) equal to 1
        fro_norm = np.sqrt(np.sum(self.w ** 2) + np.sum(self.b ** 2))
        self.w = self.w / fro_norm
        self.b = self.b / fro_norm
    
    def zero_grad(self):
        # useful to delete the stored backprop gradients of the previous mini-batch before you start a new mini-batch
        self.dw, self.db = 0, 0

    # Shape of hl: B x 28 x 28
    def forward(self, hl):
        # Store Batch Value
        B = hl.shape[0]

        # We denote dimension of hl as B X 28 X 28 where B is the batch size
        # Step 1: Convert hl to B X 28 X 28 X 1
        hl = hl[:, :, :, None]

        # Step 2: Convert hl to B X 28 X 28 X 8
        hl = np.repeat(hl, repeats = 8, axis = -1)

        # We cache hl in forward because we need to compute it in backward
        self.hl = hl

        # Step 3: Form Sliding Windows: Shape is B x 25 x 25 x 1 x 4 x 4 x 8
        sliding_windows = np.lib.stride_tricks.sliding_window_view(hl, window_shape = (4, 4, 8), axis = (1, 2, 3))
        
        # Step 4: To capture Stride in our convolution, subset windows and we now have B x 7 x 7 x 4 x 4 x 8
        sliding_windows = sliding_windows[:, ::4, ::4, 0]

        # Step 5: Element-wise multiplication to get output of B x 7 x 7 x 8
        hl_plus_1 = (sliding_windows * self.w).sum(axis = (3, 4)) + self.b

        # Step 6: Convert to B x 392
        hl_plus_1 = hl_plus_1.reshape(B, -1)

        return hl_plus_1
    
    # dhl_plus_1 Shape: B x 392
    def backward(self, dhl_plus_1):
        # Store Batch Value
        B = dhl_plus_1.shape[0]

        # Step 1: Reshape as B x 7 x 7 x 8
        dhl_plus_1 = dhl_plus_1.reshape(B, 7, 7, 8)

        # Bring back self.hl and create sliding windows
        # Step 2: Form Sliding Windows: Shape is B x 25 x 25 x 1 x 4 x 4 x 8
        sliding_windows = np.lib.stride_tricks.sliding_window_view(self.hl, window_shape = (4, 4, 8), axis = (1, 2, 3))
        
        # Step 3: To capture Stride in our convolution, subset windows and we now have B x 7 x 7 x 4 x 4 x 8
        sliding_windows = sliding_windows[:, ::4, ::4, 0]

        # Step 4: Compute dw and db
        dhl_plus_1_reshaped = dhl_plus_1[:, :, :, None, None, :] # Reshape to B x 7 x 7 x 1 x 1 x 8
        dw = (sliding_windows * dhl_plus_1_reshaped).sum(axis = (0, 1, 2))
        db = dhl_plus_1_reshaped.sum(axis = (0, 1, 2, 3, 4))

        # Step 5: Store gradients for w and b
        self.dw, self.db = dw, db

        # Step 6: Compute dhl
        dhl = (self.w * dhl_plus_1_reshaped).sum(axis = -1) # B x 7 x 7 x 4 x 4 
        dhl = dhl.transpose(0, 1, 3, 2, 4).reshape(B, 28, 28) # B x 28 x 28
        return dhl

## Part (c)
class linear_t:
    def __init__(self):
        # initialize to appropriate sizes, fill with Gaussian entries
        mean = 0.0
        std = 0.01
        self.w = np.random.normal(loc = mean, scale = std, size = (10, 392))
        self.b = np.random.normal(loc = mean, scale = std, size = (10,))

        # normalize to make the Frobenius norm of (w, b) equal to 1
        eps = 1e-12
        fro_norm = np.sqrt(np.sum(self.w ** 2) + np.sum(self.b ** 2))
        self.w = self.w / (fro_norm + eps)
        self.b = self.b / (fro_norm + eps)
    
    def zero_grad(self):
        # useful to delete the stored backprop gradients of the previous mini-batch before you start a new mini-batch
        self.dw, self.db = 0, 0
    
    def forward(self, hl):
        # Cache hl in forward because needed for back
        self.hl = hl

        # Compute h_{l + 1}
        hl_plus_1 = hl @ self.w.T + self.b

        return hl_plus_1
    
    # Shape of dhl_plus_1: B x 10
    def backward(self, dhl_plus_1):
        # Compute dhl
        dhl = dhl_plus_1 @ self.w
        
        # Compute db
        db = dhl_plus_1.sum(axis = 0)
        self.db = db

        # Compute dw
        dw = dhl_plus_1.T @ self.hl
        self.dw = dw

        return dhl

## part (d)
class relu_t:
    def __init__(self):
        pass
    def zero_grad(self):
        pass
    def forward(self, hl):
        # Cache hl in forward because needed for back
        self.hl = hl

        # Compute h_{l + 1}
        hl_plus_1 = np.maximum(0, hl)

        return hl_plus_1
    
    def backward(self, dhl_plus_1):
        return np.where(self.hl < 0, 0, dhl_plus_1)

## part (e)
class softmax_cross_entropy_t:
    def __init__(self):
        pass

    def zero_grad(self):
        pass

    def forward(self, hl, y):
        # Cache hl in forward because needed for back
        self.hl = hl

        # Flatten y as sanity check 
        y = y.flatten()
        self.y = y

        # Step 1: Compute hl_plus_1 after Softmax
        exp_hl = np.exp(hl)
        hl_plus_1 = exp_hl / np.sum(exp_hl, axis = 1, keepdims=True)

        self.hl_plus_1 = hl_plus_1
        
        # Step 2: Compute average loss over minibatch 
        B = hl.shape[0]

        # Pick the probabilities corresponding to correct labels
        correct_probs = hl_plus_1[np.arange(B), y]
        ell = -np.mean(np.log(correct_probs + 1e-12)) #Adding 1e-12 for numerical stability

        # Step 3: Compute classification error 
        y_pred = np.argmax(hl_plus_1, axis=1)
        error = np.mean(y_pred != y)

        return ell, error

    def backward(self):
        B = self.hl.shape[0]

        # Get output from softmax
        softmax_output = self.hl_plus_1

        # Create one hot labels
        y_one_hot = np.zeros_like(softmax_output)
        y_one_hot[np.arange(B), self.y] = 1

        dhl = (softmax_output - y_one_hot) / B
        return dhl


## Part (f): Check implementation of forward and backward functionalities for all layers

### Checking Linear Layer [ Forward + Backward ]
def check_forward_linear():
    layer = linear_t()
    hl = np.random.randn(1, 392)
    out1 = layer.forward(hl)
    out2 = hl @ layer.w.T + layer.b
    np.testing.assert_allclose(out1, out2, rtol=1e-6, atol=1e-6)

# check_forward_linear()

#indices_W must be passed as a list of tuples
#indices_b must be passed as list
#indices_h must be passed as list
def check_backward_linear(k, indices_W, indices_b, indices_h):
    layer = linear_t()
    hl = np.random.randn(1, 392) # Shape: 1 x 392

    # Compute forward pass
    layer.forward(hl)

    # Set up backward pass
    W = layer.w # Shape: 10 x 392
    b = layer.b # Shape: (10,)

    # Set up dhl_plus_1
    dhl_plus_1 = np.zeros(shape = (1, 10))
    dhl_plus_1[0, k] = 1 # Shape: 1 x 10

    # Compute backward
    dhl = layer.backward(dhl_plus_1)
    dw = layer.dw # Shape: 10 x 392
    db = layer.db # Shape: (10,)

    # Verify dw
    for i, j in indices_W:
        eps = np.zeros(shape = W.shape)
        eps[i, j] = np.random.normal(loc = 0.0, scale = 1e-8)
        deriv_W = ((hl @ (W + eps).T + b) - (hl @ (W - eps).T + b))[0, k] / (2 * eps)[i, j]
        np.testing.assert_allclose(dw[i, j], deriv_W, rtol=1e-6, atol=1e-6)
    
    for i in indices_b:
        eps = np.zeros(shape = b.shape)
        eps[i] = np.random.normal(loc = 0.0, scale = 1e-8)
        deriv_b = ((hl @ W.T + (b + eps)) - (hl @ W.T + (b - eps)))[0, k] / (2 * eps)[i]
        np.testing.assert_allclose(db[i], deriv_b, rtol=1e-6, atol=1e-6)
    
    for i in indices_h:
        eps = np.zeros(shape = hl.shape)
        eps[0, i] = np.random.normal(loc = 0.0, scale = 1e-8)
        deriv_h = (((hl + eps) @ W.T + b) - ((hl - eps) @ W.T + b))[0, k] / (2 * eps)[0, i]
        np.testing.assert_allclose(dhl[0, i], deriv_h, rtol=1e-6, atol=1e-6)

def test_backward_linear_random_indices():
    rng = np.random.default_rng(seed=42) # Set seed for reproducability

    # Sample 5 values of k
    k_values = rng.choice(10, size=5, replace=False)

    for k in k_values:
        # For W (10 x 392), pick 10 random (i, j) pairs
        indices_W = [(rng.integers(0, 10), rng.integers(0, 392)) for _ in range(10)]
        
        # For b (10,), pick 3 random indices
        indices_b = rng.choice(10, size=3, replace=False).tolist()
        
        # For h (392,), pick 4 random indices
        indices_h = rng.choice(392, size=4, replace=False).tolist()

        # Run the gradient check for this k
        check_backward_linear(k, indices_W, indices_b, indices_h)

# test_backward_linear_random_indices()

### Checking ReLU Layer
def check_forward_relu():
    layer = relu_t()
    hl = np.random.randn(1, 10)
    out1 = layer.forward(hl)
    out2 = np.maximum(0, hl)
    np.testing.assert_allclose(out1, out2, rtol=1e-6, atol=1e-6)

# check_forward_relu()

def check_backward_relu(k, indices_h):
    layer = relu_t()
    hl = np.random.randn(1, 10)

    # Compute forward pass
    layer.forward(hl)

    # Set up dhl_plus_1
    dhl_plus_1 = np.zeros(shape = (1, 10))
    dhl_plus_1[0, k] = 1 # Shape: 1 x 10

    # Compute backward
    dhl = layer.backward(dhl_plus_1)

    for i in indices_h:
        eps = np.zeros(shape = hl.shape)
        eps[0, i] = np.random.normal(loc = 0.0, scale = 1e-8)
        deriv_h = ((np.maximum(0, hl + eps)) - (np.maximum(0, hl - eps)))[0, k] / (2 * eps)[0, i]
        np.testing.assert_allclose(dhl[0, i], deriv_h, rtol=1e-6, atol=1e-6)

def test_backward_relu_random_indices():
    rng = np.random.default_rng(seed=42) # Set seed for reproducability

    # Sample 5 values of k
    k_values = rng.choice(10, size=5, replace=False)

    for k in k_values:
        # For h (10,), pick 4 random indices
        indices_h = rng.choice(10, size=4, replace=False).tolist()

        # Run the gradient check for this k
        check_backward_relu(k, indices_h)

# test_backward_relu_random_indices()

### Test softmax_cross_entropy_t Function
def softmax_cross_entropy(hl, y):
    # Flatten y as sanity check 
    y = y.flatten()

    # Step 1: Compute hl_plus_1 after Softmax
    exp_hl = np.exp(hl)
    hl_plus_1 = exp_hl / np.sum(exp_hl, axis = 1, keepdims=True)
    
    # Step 2: Compute average loss over minibatch 
    B = hl.shape[0]

    # Pick the probabilities corresponding to correct labels
    correct_probs = hl_plus_1[np.arange(B), y]
    ell = -np.mean(np.log(correct_probs + 1e-12)) #Adding 1e-12 for numerical stability

    return ell

def check_backward_softmax(indices_h):
    layer = softmax_cross_entropy_t()
    hl = np.random.randn(1, 10)
    y = np.random.randint(0, 10, size=(1,))

    # Compute forward pass
    layer.forward(hl, y)

    # Compute backward
    dhl = layer.backward()

    for i in indices_h:
        eps = np.zeros(shape = hl.shape)
        eps[0, i] = np.random.normal(loc = 0.0, scale = 1e-8)
        diff = (softmax_cross_entropy(hl + eps, y) - softmax_cross_entropy(hl - eps, y))
        deriv_h = diff / (2 * eps)[0, i]
        np.testing.assert_allclose(dhl[0, i], deriv_h, rtol=1e-6, atol=1e-6)

def test_backward_softmax_random_indices():
    rng = np.random.default_rng(seed=42) # Set seed for reproducability

    # For h (10,), pick 4 random indices
    indices_h = rng.choice(10, size=5, replace=False).tolist()

    # Run the gradient check for this k
    check_backward_softmax(indices_h)
    
# test_backward_softmax_random_indices()

### Test embedding_t Function
def embedding(hl, W, b):
    # Store Batch Value
    B = hl.shape[0]

    # We denote dimension of hl as B X 28 X 28 where B is the batch size
    # Step 1: Convert hl to B X 28 X 28 X 1
    hl = hl[:, :, :, None]

    # Step 2: Convert hl to B X 28 X 28 X 8
    hl = np.repeat(hl, repeats = 8, axis = -1)

    # Step 3: Form Sliding Windows: Shape is B x 25 x 25 x 1 x 4 x 4 x 8
    sliding_windows = np.lib.stride_tricks.sliding_window_view(hl, window_shape = (4, 4, 8), axis = (1, 2, 3))
    
    # Step 4: To capture Stride in our convolution, subset windows and we now have B x 7 x 7 x 4 x 4 x 8
    sliding_windows = sliding_windows[:, ::4, ::4, 0]

    # Step 5: Element-wise multiplication to get output of B x 7 x 7 x 8
    hl_plus_1 = (sliding_windows * W).sum(axis = (3, 4)) + b

    # Step 6: Convert to B x 392
    hl_plus_1 = hl_plus_1.reshape(B, -1)

    return hl_plus_1

def check_backward_embedding(t, indices_W, indices_b, indices_h):
    layer = embedding_t()
    hl = np.random.randn(1, 28, 28)

    # Compute forward pass
    layer.forward(hl)

    # Set up backward pass
    W = layer.w # Shape: 4 x 4 x 8
    b = layer.b # Shape: (8,)

    # Set up dhl_plus_1
    dhl_plus_1 = np.zeros(shape = (1, 392))
    dhl_plus_1[0, t] = 1 # Shape: 1 x 392

    # Compute backward
    dhl = layer.backward(dhl_plus_1)
    dw = layer.dw # Shape: 4 x 4 x 8
    db = layer.db # Shape: (8,)

    # Verify dw
    for i, j, k in indices_W:
        eps = np.zeros(shape = W.shape)
        eps[i, j, k] = np.random.normal(loc = 0.0, scale = 1e-8)
        deriv_W = (embedding(hl, W + eps, b) - embedding(hl, W - eps, b))[0, t] / (2 * eps)[i, j, k]
        np.testing.assert_allclose(dw[i, j, k], deriv_W, rtol=1e-6, atol=1e-6)
    
    for i in indices_b:
        eps = np.zeros(shape = b.shape)
        eps[i] = np.random.normal(loc = 0.0, scale = 1e-8)
        deriv_b = (embedding(hl, W, b + eps) - embedding(hl, W, b - eps))[0, t] / (2 * eps)[i]
        np.testing.assert_allclose(db[i], deriv_b, rtol=1e-6, atol=1e-6)
    
    for i, j in indices_h:
        eps = np.zeros(shape = hl.shape)
        eps[0, i, j] = np.random.normal(loc = 0.0, scale = 1e-8)
        deriv_h = (embedding(hl + eps, W, b) - embedding(hl - eps, W, b))[0, t] / (2 * eps)[0, i, j]
        np.testing.assert_allclose(dhl[0, i, j], deriv_h, rtol=1e-6, atol=1e-6)

def test_backward_embedding_random_indices():
    rng = np.random.default_rng(seed=42) # Set seed for reproducability

    # Sample 5 values of k
    t_values = rng.choice(10, size=5, replace=False)

    for t in t_values:
        # For W (4 x 4 x 8), pick 10 random (i, j) pairs
        indices_W = [(rng.integers(0, 4), rng.integers(0, 4), rng.integers(0, 8)) for _ in range(10)]
        
        # For b (8,), pick 3 random indices
        indices_b = rng.choice(8, size=3, replace=False).tolist()
        
        # For h (28, 28), pick 4 random indices
        indices_h = [(rng.integers(0, 28), rng.integers(0, 28)) for _ in range(4)]

        # Run the gradient check for this k
        check_backward_embedding(t, indices_W, indices_b, indices_h)

# test_backward_embedding_random_indices()


## part (g)
### Dataset already loaded from part (a)
X_train = X_train.astype(np.float32) / 255.0
X_val = X_val.astype(np.float32) / 255.0

### Initialize all Layers
l1, l2, l3, l4 = embedding_t(), linear_t(), relu_t(), softmax_cross_entropy_t()
net = [l1, l2, l3, l4]

print(f"Part g: X_train shape -> {X_train.shape}, y_train shape: {y_train.shape}")

# Train for at least 1000 iterations
B = 32
lr = 0.1
training_loss = []
epochs = 10000
for t in range(epochs):
    # 1. sample a mini-batch of size = 32
    # each image in the mini-batch is chosen uniformly randomly from the
    # training dataset
    indices = np.random.choice(X_train.shape[0], size=B, replace=False)
    X_train_batch = X_train[indices]
    y_train_batch = y_train[indices]

    # 2. zero gradient buffer
    for l in net:
        l.zero_grad()

    # 3. Forward Pass
    h1 = l1.forward(X_train_batch)
    h2 = l2.forward(h1)
    h3 = l3.forward(h2)
    ell, error = l4.forward(h3, y_train_batch)

    # 4. Backward Pass
    dh3 = l4.backward()
    dh2 = l3.backward(dh3)
    dh1 = l2.backward(dh2)
    dx = l1.backward(dh1)

    # 5. Gather Backprop gradients
    dw1, db1 = l1.dw, l1.db
    dw2, db2 = l2.dw, l2.db

    # 6. Print some quantities for logging
    if (t + 1) % 100 == 0:
        print(f"Epoch: {t + 1}, Loss: {ell}, Error: {error}")
    training_loss.append(ell)

    # 7. one step of SGD
    l1.w = l1.w - lr*dw1
    l1.b = l1.b - lr*db1
    l2.w = l2.w - lr*dw2
    l2.b = l2.b - lr*db2

plt.figure(figsize=(8, 5))
plt.plot(training_loss, label='Training Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss vs Iteration')
plt.legend()
plt.grid(True)
plt.show()