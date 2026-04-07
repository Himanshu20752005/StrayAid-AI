"""
train_all_models.py
-------------------
Run this from:  Stray-Aid/
Command       :  python train_all_models.py

Trains all four models and copies them to Backend/models/.
"""

import os, sys

BASE      = os.path.dirname(os.path.abspath(__file__))
OUT_DIR   = os.path.join(BASE, "Backend", "models")
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Animal Classifier  →  animal_model.pth
# ─────────────────────────────────────────────────────────────────────────────
def train_animal_classifier():
    import torch, torch.nn as nn
    from torchvision import datasets, transforms, models
    from torch.utils.data import DataLoader

    print("\n" + "="*60)
    print("  TRAINING: Animal Classifier  (cat / cow / dog)")
    print("="*60)

    DATA_DIR = os.path.join(BASE, "training_Animal_Classifer", "dataset")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    loader  = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    print(f"  Classes  : {dataset.classes}")
    print(f"  Samples  : {len(dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device   : {device}")

    model = models.resnet18(weights="IMAGENET1K_V1")
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.0003)

    EPOCHS = 10
    for epoch in range(EPOCHS):
        total_loss = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}/{EPOCHS}  Loss: {total_loss:.4f}")

    out = os.path.join(OUT_DIR, "animal_model.pth")
    torch.save(model.state_dict(), out)
    print(f"\n  ✅ Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Cat Disease Model  →  best_cat_model.keras
# ─────────────────────────────────────────────────────────────────────────────
def train_cat_disease():
    import tensorflow as tf

    print("\n" + "="*60)
    print("  TRAINING: Cat Disease Classifier")
    print("="*60)

    DATA_DIR   = os.path.join(BASE, "training_Cat_decease_Detection_system", "dataset")
    IMG_SIZE   = (224, 224)
    BATCH_SIZE = 32
    EPOCHS     = 20
    LR         = 1e-3
    OUT_PATH   = os.path.join(OUT_DIR, "best_cat_model.keras")

    AUTOTUNE = tf.data.AUTOTUNE

    def preprocess(img, label):
        img = tf.cast(img, tf.float32)
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        return img, label

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="training",
        seed=42, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=True)
    val_ds   = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="validation",
        seed=42, image_size=IMG_SIZE, batch_size=BATCH_SIZE)

    class_names = train_ds.class_names
    NUM_CLASSES = len(class_names)
    print(f"  Classes ({NUM_CLASSES}): {class_names}")

    train_ds = (train_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
                        .prefetch(AUTOTUNE))
    val_ds   = (val_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
                      .prefetch(AUTOTUNE))

    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])

    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    base.trainable = False

    inputs  = tf.keras.Input(shape=(224, 224, 3))
    x = augment(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model   = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[
        tf.keras.callbacks.ModelCheckpoint(OUT_PATH, monitor="val_accuracy",
                                           save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                                             patience=3, verbose=1),
    ])
    print(f"\n  ✅ Saved → {OUT_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Cow Disease Model  →  best_cow_model.keras
# ─────────────────────────────────────────────────────────────────────────────
def train_cow_disease():
    import tensorflow as tf

    print("\n" + "="*60)
    print("  TRAINING: Cow Disease Classifier")
    print("="*60)

    DATA_DIR   = os.path.join(BASE, "training_Cow_decease_Detection_system", "dataset")
    IMG_SIZE   = (224, 224)
    BATCH_SIZE = 32
    EPOCHS     = 15
    LR         = 1e-3
    OUT_PATH   = os.path.join(OUT_DIR, "best_cow_model.keras")

    AUTOTUNE = tf.data.AUTOTUNE

    def preprocess(img, label):
        img = tf.cast(img, tf.float32)
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        return img, label

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="training",
        seed=42, image_size=IMG_SIZE, batch_size=BATCH_SIZE)
    val_ds   = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="validation",
        seed=42, image_size=IMG_SIZE, batch_size=BATCH_SIZE)

    cow_class_names = train_ds.class_names
    NUM_COW_CLASSES = len(cow_class_names)
    print(f"  Classes ({NUM_COW_CLASSES}): {cow_class_names}")

    train_ds = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val_ds   = val_ds.map(preprocess,   num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])

    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    base.trainable = False

    inputs  = tf.keras.Input(shape=(224, 224, 3))
    x = augment(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(NUM_COW_CLASSES, activation="softmax")(x)
    model   = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[
        tf.keras.callbacks.ModelCheckpoint(OUT_PATH, monitor="val_accuracy",
                                           save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                         restore_best_weights=True),
    ])
    print(f"\n  ✅ Saved → {OUT_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Dog Disease Model  →  best_model_Dog.keras
# ─────────────────────────────────────────────────────────────────────────────
def train_dog_disease():
    import tensorflow as tf, numpy as np

    print("\n" + "="*60)
    print("  TRAINING: Dog Disease Classifier")
    print("="*60)

    BASE_DIR   = os.path.join(BASE, "training_Dog_decease_Detection_system", "dataset")
    IMG_SIZE   = (224, 224)
    BATCH_SIZE = 32
    EPOCHS     = 25
    FINE_EPOCHS= 10
    LR         = 1e-3
    FINE_LR    = 1e-5
    OUT_PATH   = os.path.join(OUT_DIR, "best_model_Dog.keras")

    CLASSES = ["demodicosis","Dermatitis","Fungal_infections",
               "Healthy","Hypersensitivity","ringworm"]
    NUM_CLASSES = len(CLASSES)

    AUTOTUNE = tf.data.AUTOTUNE

    def preprocess(img, label):
        img = tf.cast(img, tf.float32)
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        return img, label

    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomBrightness(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ], name="augmentation")

    def load_split(directory, do_augment=False):
        ds = tf.keras.utils.image_dataset_from_directory(
            directory, class_names=CLASSES, image_size=IMG_SIZE,
            batch_size=BATCH_SIZE, shuffle=do_augment, seed=42,
            label_mode="categorical")
        ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE)
        if do_augment:
            ds = ds.map(lambda x, y: (augment(x, training=True), y),
                        num_parallel_calls=AUTOTUNE)
        return ds.cache().prefetch(AUTOTUNE)

    train_ds = load_split(os.path.join(BASE_DIR, "train"), do_augment=True)
    valid_ds = load_split(os.path.join(BASE_DIR, "valid"))
    test_ds  = load_split(os.path.join(BASE_DIR, "test"))

    # Build model (EfficientNetB3)
    base = tf.keras.applications.EfficientNetB3(
        include_top=False, weights="imagenet",
        input_shape=(*IMG_SIZE, 3), pooling=None)
    base.trainable = False

    inputs  = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model   = tf.keras.Model(inputs, outputs, name="DogDiseaseDetector")

    callbacks_p1 = [
        tf.keras.callbacks.ModelCheckpoint(OUT_PATH, monitor="val_accuracy",
                                           save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=7,
                                         restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=3, min_lr=1e-7, verbose=1),
    ]

    # Phase 1
    print("\n  PHASE 1 — Training head (base frozen)")
    model.compile(optimizer=tf.keras.optimizers.Adam(LR),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=valid_ds,
              epochs=EPOCHS, callbacks=callbacks_p1)

    # Phase 2 – fine-tune last 30 layers
    print("\n  PHASE 2 — Fine-tuning last 30 base layers")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(FINE_LR),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=valid_ds, epochs=FINE_EPOCHS, callbacks=[
        tf.keras.callbacks.ModelCheckpoint(OUT_PATH, monitor="val_accuracy",
                                           save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                         restore_best_weights=True, verbose=1),
    ])
    print(f"\n  ✅ Saved → {OUT_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train all StrayAid-AI models")
    parser.add_argument("--only", choices=["animal","cat","cow","dog"],
                        help="Train only one model (default: train all)")
    args = parser.parse_args()

    if args.only == "animal" or args.only is None:
        train_animal_classifier()

    if args.only == "cat" or args.only is None:
        train_cat_disease()

    if args.only == "cow" or args.only is None:
        train_cow_disease()

    if args.only == "dog" or args.only is None:
        train_dog_disease()

    print("\n" + "="*60)
    print("  ALL MODELS TRAINED & SAVED TO Backend/models/")
    print("="*60)
    print(f"  → {os.path.join(OUT_DIR, 'animal_model.pth')}")
    print(f"  → {os.path.join(OUT_DIR, 'best_cat_model.keras')}")
    print(f"  → {os.path.join(OUT_DIR, 'best_cow_model.keras')}")
    print(f"  → {os.path.join(OUT_DIR, 'best_model_Dog.keras')}")
