# ==============================================================
# model.py
# Train an emotion detection CNN using MobileNetV2 on FER-2013
# Run this script in Google Colab after extracting FER-2013 dataset
# ==============================================================

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# ==========================
# Configuration
# ==========================
TRAIN_DIR = '/content/fer2013/train'
TEST_DIR = '/content/fer2013/test'
IMG_SIZE = (96, 96)
BATCH_SIZE = 64
EPOCHS = 10
MODEL_SAVE_PATH = 'emotion_detector_v1.h5'

# ==========================
# Data Preparation
# ==========================
print("🔄 Preparing training and validation data...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode='rgb',  # convert grayscale to RGB
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    class_mode='categorical'
)

num_classes = train_generator.num_classes
print(f"✅ Dataset ready. Classes detected: {train_generator.class_indices}")

# ==========================
# Model Architecture
# ==========================
print("🧠 Building MobileNetV2-based model...")

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# Freeze base model for initial training
for layer in base_model.layers:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==========================
# Training
# ==========================
print("🚀 Starting training...")
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS
)

# ==========================
# Fine-tuning (optional)
# ==========================
print("🎯 Fine-tuning last 20 layers of MobileNetV2...")
for layer in base_model.layers[-20:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_ft = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=3
)

# ==========================
# Save Model
# ==========================
model.save(MODEL_SAVE_PATH)
print(f"💾 Model saved successfully as {MODEL_SAVE_PATH}")

# ==========================
# Plot Training Curves
# ==========================
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.show()
