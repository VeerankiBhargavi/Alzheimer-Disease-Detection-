import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.io
import os
import tensorflow as tf

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from skimage.io import imread, imshow
from skimage.transform import resize
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization, Dropout, Dense, Activation, GlobalAveragePooling2D, concatenate
from tensorflow.keras.applications.densenet import DenseNet169
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Helper function for greyscale to RGB conversion
def grey2rgb(image):
    return np.stack((image,) * 3, axis=-1)

# Data augmentation setup
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=30, zoom_range=0.2,
                                   horizontal_flip=True, vertical_flip=True, validation_split=0.2)

valid_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load train and validation datasets
train_dataset = train_datagen.flow_from_directory(directory='your_train_directory', target_size=(224, 224),
                                                  class_mode='categorical', subset='training', batch_size=128)

valid_dataset = valid_datagen.flow_from_directory(directory='your_train_directory', target_size=(224, 224),
                                                  class_mode='categorical', subset='validation', batch_size=128)

# Model Initialization
densenet_base = DenseNet169(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
resnet_base = ResNet50(input_shape=(224, 224, 3), include_top=False, weights="imagenet")

# Freeze base layers
for layer in densenet_base.layers:
    layer.trainable = False
for layer in resnet_base.layers:
    layer.trainable = False

# Combine DenseNet and ResNet features
input_layer = Input(shape=(224, 224, 3))
densenet_features = GlobalAveragePooling2D()(densenet_base(input_layer))
resnet_features = GlobalAveragePooling2D()(resnet_base(input_layer))

combined = concatenate([densenet_features, resnet_features])

# Add dense layers
x = Dropout(0.5)(combined)
x = BatchNormalization()(x)
x = Dense(2048, kernel_initializer='he_uniform', activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1024, kernel_initializer='he_uniform', activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(4, activation='softmax')(x)

# Final model
model = Model(inputs=input_layer, outputs=output_layer)
model.summary()

# Compile the model
OPT = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', metrics=[tf.keras.metrics.AUC(name='auc')], optimizer=OPT)

# Callbacks
save_dir = os.path.join(os.getcwd(), '.keras')
filepath = os.path.join(save_dir, 'best_weights.keras')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

earlystopping = EarlyStopping(monitor='val_auc', mode='max', patience=15, verbose=1)
checkpoint = ModelCheckpoint(filepath, monitor='val_auc', mode='max', save_best_only=True, verbose=1)

# Model training
model_history = model.fit(train_dataset, validation_data=valid_dataset, epochs=5, callbacks=[earlystopping, checkpoint], verbose=1)

# Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 4))
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    # AUC Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['auc'], label='Train AUC')
    plt.plot(history.history['val_auc'], label='Validation AUC')
    plt.title('Model AUC')
    plt.ylabel('AUC')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_history(model_history)

# Test data
test_dataset = test_datagen.flow_from_directory(directory='your_test_directory', target_size=(224, 224),
                                                class_mode='categorical', batch_size=128)

# Evaluate model
model.evaluate(test_dataset)

# Confusion matrix and classification report
test_labels = test_dataset.classes
predictions = model.predict(test_dataset, verbose=1)
predicted_labels = np.argmax(predictions, axis=1)

cm = confusion_matrix(test_labels, predicted_labels)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.class_indices, yticklabels=test_dataset.class_indices)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print(classification_report(test_labels, predicted_labels, target_names=test_dataset.class_indices.keys()))

# Single image prediction
image_path = 'path_to_mri_image.jpg'
img = load_img(image_path, target_size=(224, 224))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
predicted_class = np.argmax(prediction, axis=1)
class_labels = list(train_dataset.class_indices.keys())
print(f'The patient is predicted to be in the stage: {class_labels[predicted_class[0]]}')
