import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

img_size = (64, 64)
batch_size = 32
epochs = 30

# Data augmentation generator for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    height_shift_range=0.1,
    width_shift_range=0.1,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    validation_split=0.2
)

# For training data (subset='training')
train_generator = train_datagen.flow_from_directory(
    'Images',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='sparse',
    subset='training',
    shuffle=True,
    seed=123
)

# For validation data (subset='validation'), no augmentation but rescaling applied
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_generator = val_datagen.flow_from_directory(
    'Images',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='sparse',
    subset='validation',
    shuffle=True,
    seed=123
)

# Model definition
model = tf.keras.Sequential([
    tf.keras.Input(shape=(64, 64, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(2, activation="softmax")  
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Callbacks for early stopping and learning rate reduction
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

# Training
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)
# Save model
model.save("CNN_model.keras")



# Load model
base_model = tf.keras.models.load_model("CNN_model.keras")

# Freeze model except last 3 layers
for layer in base_model.layers[:-3]:
    layer.trainable = False

# Adding new model
fine_tuned_model = tf.keras.Sequential([
    base_model,
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

#fine_tuned_model = Model(inputs=base_model.input, outputs=output)

fine_tuned_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


new_train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    height_shift_range=0.1,
    width_shift_range=0.1,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    validation_split=0.2
)

new_train_generator = new_train_datagen.flow_from_directory(
    'DOGS',  # ا
    target_size=img_size,
    batch_size=batch_size,
    class_mode='sparse',
    subset='training',
    shuffle=True,
    seed=123
)

new_val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

new_val_generator = new_val_datagen.flow_from_directory(
    'DOGS',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='sparse',
    subset='validation',
    shuffle=True,
    seed=123
)

#Fine-tuning 
history_finetune = fine_tuned_model.fit(
    new_train_generator,
    validation_data=new_val_generator,
    epochs=10,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

fine_tuned_model.save("CNN_model_finetuned.keras")

# Plotting accuracy and loss
plt.plot(history_finetune.history['accuracy'], label='Fine-tune Training Accuracy')
plt.plot(history_finetune.history['val_accuracy'], label='Fine-tune Validation Accuracy')
plt.plot(history_finetune.history['loss'], label='Fine-tune Training Loss')
plt.plot(history_finetune.history['val_loss'], label='Fine-tune Validation Loss')

plt.xlabel('Epoch')
plt.ylabel('Accuracy / Loss')
plt.legend()
plt.show()

# Print final accuracy
print(f"Final Fine-tune Training Accuracy: {history_finetune.history['accuracy'][-1]:.2f}")
print(f"Final Fine-tune Validation Accuracy: {history_finetune.history['val_accuracy'][-1]:.2f}")
