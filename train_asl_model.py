# train_asl_model.py

import os
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

DATASET_DIR = "asl_dataset"
CHECKPOINT_DIR = "asl_project/models/checkpoints"
FINAL_MODEL_PATH = "asl_project/models/asl_model_final.keras"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 50  

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(train_gen.num_classes, activation="softmax")
])

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".keras")]
checkpoint_files.sort()
if checkpoint_files:
    last_checkpoint = checkpoint_files[-1]
    print(f"Found checkpoint: {last_checkpoint}")
    model.load_weights(os.path.join(CHECKPOINT_DIR, last_checkpoint))
    match = re.search(r"epoch(\d+)", last_checkpoint)
    if match:
        initial_epoch = int(match.group(1))
    else:
        initial_epoch = 0
    print(f"Resuming from epoch {initial_epoch + 1}")
else:
    print("No checkpoint found. Starting fresh training...")
    initial_epoch = 0

checkpoint_cb = ModelCheckpoint(
    filepath=os.path.join(CHECKPOINT_DIR, "asl_model_epoch{epoch:02d}_val{val_accuracy:.2f}.keras"),
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

earlystop_cb = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=[checkpoint_cb, earlystop_cb],
    initial_epoch=initial_epoch
)

model.save(FINAL_MODEL_PATH)
print(f"Training is done. The model is saved to {FINAL_MODEL_PATH}")