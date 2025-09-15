#DenseNet-EfficientNet Ensemble Skin Disease Classifier
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import DenseNet121, EfficientNetB0
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

# Parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 15

# Directory setup
train_dir = r"D:\Project_6month\skin diseases\SkinDisease\SkinDisease\train"
val_dir = r"D:\Project_6month\skin diseases\SkinDisease\SkinDisease\test"

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    shear_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# --- Calculate class weights for balancing ---
# Get class distribution from train_generator
counter = Counter(train_generator.classes)
n_classes = len(train_generator.class_indices)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(n_classes),
    y=train_generator.classes
)
class_weight_dict = dict(enumerate(class_weights))

# --- Build DenseNet121 Model ---
base_densenet = DenseNet121(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,
    weights='imagenet'
)
base_densenet.trainable = False

model_densenet = models.Sequential([
    base_densenet,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(n_classes, activation='softmax')
])
model_densenet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- Build EfficientNetB0 Model ---
base_efficientnet = EfficientNetB0(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,
    weights='imagenet'
)
base_efficientnet.trainable = False

model_efficientnet = models.Sequential([
    base_efficientnet,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(n_classes, activation='softmax')
])
model_efficientnet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- Callbacks ---
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

# --- Train DenseNet121 ---
history_dense = model_densenet.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=len(val_generator),
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weight_dict
)

# --- Train EfficientNetB0 ---
history_eff = model_efficientnet.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=len(val_generator),
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weight_dict
)

# --- Ensemble Prediction Logic ---
def ensemble_predict(models, X):
    # X is a batch of preprocessed images
    preds = np.zeros((X.shape[0], n_classes))
    for model in models:
        preds += model.predict(X, verbose=0)
    return np.argmax(preds, axis=1)

# --- Evaluate on Test Data ---
X_test, y_test = next(val_generator)
ensemble_models = [model_densenet, model_efficientnet]
ensemble_preds = ensemble_predict(ensemble_models, X_test)
true_labels = np.argmax(y_test, axis=1)
ensemble_accuracy = np.mean(ensemble_preds == true_labels)
print(f"Ensemble Validation Accuracy (batch): {ensemble_accuracy*100:.2f}%")

# --- Save Models ---
model_densenet.save('ham10000_skin_cnn_densenet121_2.h5')
model_efficientnet.save('ham10000_skin_cnn_efficientnetb0_2.h5')

# --- Plotting ---
plt.figure(figsize=(10,5))
plt.plot(history_dense.history['accuracy'], label='DenseNet Train')
plt.plot(history_dense.history['val_accuracy'], label='DenseNet Val')
plt.plot(history_eff.history['accuracy'], label='EffNet Train')
plt.plot(history_eff.history['val_accuracy'], label='EffNet Val')
plt.title('Model accuracy over epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(history_dense.history['loss'], label='DenseNet Train')
plt.plot(history_dense.history['val_loss'], label='DenseNet Val')
plt.plot(history_eff.history['loss'], label='EffNet Train')
plt.plot(history_eff.history['val_loss'], label='EffNet Val')
plt.title('Model loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
