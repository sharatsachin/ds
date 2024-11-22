# TensorFlow

## What is TensorFlow and how does it differ from other frameworks?

TensorFlow is a deep learning framework that provides:
- Static and dynamic computational graphs
- Eager execution mode
- Enterprise-level production capabilities
- Distributed training support
- TensorFlow Extended (TFX) for ML pipelines
- Keras high-level API integration

Key differences from other frameworks:
- More production-focused than PyTorch
- Better deployment options (TensorFlow Serving, TF Lite)
- Stronger enterprise adoption
- More comprehensive visualization (TensorBoard)
- Integrated mobile/edge deployment solutions

## What are tensors in TensorFlow and how do you create them?

Tensors are multi-dimensional arrays and the fundamental data structure in TensorFlow.

Common ways to create tensors:
```python
import tensorflow as tf

# From Python list/array
x = tf.constant([1, 2, 3])

# Zeros and ones
zeros = tf.zeros([2, 3])  # 2x3 tensor of zeros
ones = tf.ones([2, 3])    # 2x3 tensor of ones

# Random tensors
rand = tf.random.uniform([2, 3])     # uniform random
randn = tf.random.normal([2, 3])     # normal distribution
randint = tf.random.uniform([2, 3], minval=0, maxval=10, dtype=tf.int32)

# Range tensors
range = tf.range(0, 10, delta=1)
linspace = tf.linspace(0., 10., 5)

# From NumPy array
import numpy as np
np_array = np.array([1, 2, 3])
tensor = tf.convert_to_tensor(np_array)
```

## What are the basic tensor operations in TensorFlow?

Common tensor operations:
```python
# Arithmetic operations
x + y  # Addition
x - y  # Subtraction
x * y  # Element-wise multiplication
x / y  # Division
tf.matmul(x, y)  # Matrix multiplication
x @ y            # Matrix multiplication (Python 3)

# Mathematical operations
tf.math.abs(x)     # Absolute value
tf.math.square(x)  # Square
tf.math.sqrt(x)    # Square root
tf.math.exp(x)     # Exponential
tf.math.log(x)     # Natural logarithm

# Reshaping
tf.reshape(x, [3, 4])     # Reshape tensor
tf.squeeze(x)             # Remove dimensions of size 1
tf.expand_dims(x, axis=0) # Add dimension
tf.transpose(x)           # Transpose dimensions

# Indexing and slicing
x[0]           # First element
x[:, 1]        # Second column
x[1:3, 2:4]    # 2D slice

# Concatenation
tf.concat([x, y], axis=0)  # Concatenate along dimension
tf.stack([x, y], axis=0)   # Stack tensors
```

## How do you handle GPU acceleration in TensorFlow?

Device management in TensorFlow:
```python
# Check available devices
print(tf.config.list_physical_devices())

# GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Device placement
with tf.device('/GPU:0'):
    x = tf.constant([1, 2, 3])

# Check tensor device
x.device

# Automatic device placement
tf.config.set_soft_device_placement(True)
```

## What are the main components of a neural network in TensorFlow?

Basic neural network components using Keras API:
```python
from tensorflow import keras
from tensorflow.keras import layers

class MyModel(keras.Model):
    def __init__(self):
        super().__init__()
        # Layers
        self.conv1 = layers.Conv2D(32, 3, activation='relu')
        self.dense1 = layers.Dense(64, activation='relu')
        
        # Pooling layers
        self.maxpool = layers.MaxPooling2D()
        self.avgpool = layers.AveragePooling2D()
        
        # Normalization
        self.batchnorm = layers.BatchNormalization()
        self.dropout = layers.Dropout(0.5)
        
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.maxpool(x)
        x = self.dense1(x)
        return x

# Alternative functional API
model = keras.Sequential([
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

## How do you define loss functions and optimizers?

Common loss functions and optimizers:
```python
# Loss functions
loss = keras.losses.SparseCategoricalCrossentropy()
loss = keras.losses.BinaryCrossentropy()
loss = keras.losses.MeanSquaredError()
loss = keras.losses.MeanAbsoluteError()

# Optimizers
optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
optimizer = keras.optimizers.Adam(learning_rate=0.001)
optimizer = keras.optimizers.RMSprop(learning_rate=0.01)

# Learning rate schedules
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=10000,
    decay_rate=0.9
)
```

## How do you implement custom training loops in TensorFlow?

Basic training loop structure:
```python
@tf.function  # Optional: Compile to graph for better performance
def train_step(model, inputs, labels, optimizer, loss_fn):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(labels, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def train_model(model, train_dataset, optimizer, loss_fn, epochs):
    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in train_dataset:
            loss = train_step(model, inputs, labels, optimizer, loss_fn)
            total_loss += loss
        
        # Print epoch statistics
        avg_loss = total_loss / len(train_dataset)
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')
```

## How do you implement model evaluation?

Evaluation loop structure:
```python
@tf.function
def test_step(model, inputs, labels, loss_fn, metric):
    predictions = model(inputs, training=False)
    loss = loss_fn(labels, predictions)
    metric.update_state(labels, predictions)
    return loss

def evaluate_model(model, test_dataset, loss_fn):
    metric = keras.metrics.SparseCategoricalAccuracy()
    total_loss = 0
    
    for inputs, labels in test_dataset:
        loss = test_step(model, inputs, labels, loss_fn, metric)
        total_loss += loss
    
    avg_loss = total_loss / len(test_dataset)
    accuracy = metric.result()
    
    return avg_loss, accuracy
```

## How do you save and load models in TensorFlow?

Model persistence:
```python
# Save entire model
model.save('model_path')
model.save_weights('weights_path')

# Save in SavedModel format
tf.saved_model.save(model, 'saved_model_path')

# Checkpoints
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
checkpoint.save('checkpoint_path')

# Load model
loaded_model = keras.models.load_model('model_path')
model.load_weights('weights_path')

# Load from SavedModel
loaded = tf.saved_model.load('saved_model_path')

# Restore checkpoint
checkpoint.restore('checkpoint_path')
```

## How do you implement data loading in TensorFlow?

Data loading and datasets:
```python
# Create dataset from tensors
dataset = tf.data.Dataset.from_tensor_slices((features, labels))

# Load from files
dataset = tf.data.Dataset.from_generator(
    generator_function,
    output_types=(tf.float32, tf.int32),
    output_shapes=([28, 28], [])
)

# Data pipeline
dataset = dataset.shuffle(buffer_size=1000)
dataset = dataset.batch(32)
dataset = dataset.prefetch(tf.data.AUTOTUNE)

# Built-in datasets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
```

## What are the common data preprocessing layers in TensorFlow?

Data preprocessing layers:
```python
# Numerical features
normalizer = layers.Normalization()
normalizer.adapt(data)

# Categorical features
vectorizer = layers.TextVectorization(max_tokens=1000)
vectorizer.adapt(text_data)

# Image preprocessing
preprocessing = keras.Sequential([
    layers.Resizing(256, 256),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.Normalization()
])
```

## How do you implement transfer learning in TensorFlow?

Transfer learning implementation:
```python
# Load pre-trained model
base_model = keras.applications.ResNet50(
    weights='imagenet',
    include_top=False
)

# Freeze base model
base_model.trainable = False

# Create new model
inputs = keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(num_classes)(x)
model = keras.Model(inputs, outputs)

# Fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-4]:
    layer.trainable = False
```

## How do you handle custom loss functions and metrics?

Creating custom loss functions and metrics:
```python
class CustomLoss(keras.losses.Loss):
    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))

class CustomMetric(keras.metrics.Metric):
    def __init__(self, name='custom_metric', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        values = tf.square(y_true - y_pred)
        self.total.assign_add(tf.reduce_sum(values))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))
    
    def result(self):
        return self.total / self.count
```

## How do you use TensorBoard for visualization?

TensorBoard integration:
```python
# Create TensorBoard callback
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir='./logs',
    histogram_freq=1,
    write_graph=True
)

# Add custom scalars
file_writer = tf.summary.create_file_writer('./logs/metrics')
with file_writer.as_default():
    tf.summary.scalar('custom_metric', value, step=epoch)

# Profile model performance
tf.summary.trace_on(graph=True, profiler=True)
# Run model
with file_writer.as_default():
    tf.summary.trace_export(name="model_trace", step=0)
```