import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os

# Define the base directory where the dataset is located
base_dir = './fruits-360-original-size'  # Adjust this if your dataset is in a different directory

# Check if the directory exists
assert os.path.exists(base_dir), f"Directory '{base_dir}' does not exist"

# Define batch size and image size
batch_size = 32
image_size = (224, 224)

# Create the training and validation datasets
train_dir = os.path.join(base_dir, 'Training')
validation_dir = os.path.join(base_dir, 'Test')

train_dataset = image_dataset_from_directory(
    train_dir,
    shuffle=True,
    batch_size=batch_size,
    image_size=image_size
)

validation_dataset = image_dataset_from_directory(
    validation_dir,
    shuffle=True,
    batch_size=batch_size,
    image_size=image_size
)

# Prefetch data for performance optimization
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

# MobileNetV2 model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Load the MobileNetV2 model, excluding the top layers
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(train_dataset.class_names), activation='softmax')(x)

# Create the complete model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base model layers to prevent them from being trained
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 10
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs
)

print("Model training complete!")

# Optional: Fine-tune the model
for layer in base_model.layers[-20:]:
    layer.trainable = True

# Recompile the model with a lower learning rate
model.compile(optimizer=Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Continue training the model
fine_tune_epochs = 10
history_fine = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=fine_tune_epochs
)

print("Fine-tuning complete!")
