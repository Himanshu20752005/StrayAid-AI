"""
train.py
--------
Train a Dog-Disease Detection CNN using Transfer Learning (EfficientNetB3).

Classes : demodicosis | Dermatitis | Fungal_infections |
          Healthy | Hypersensitivity | ringworm

Usage:
    pip install tensorflow scikit-learn matplotlib seaborn
    python train.py
    python train.py --epochs 30 --batch_size 16 --base_dir "path/to/dataset"
"""

import os
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
DEFAULT_DATASET = (
    r"C:\Users\Shailendra\Downloads\Final_Proj\Stray-Aid"
    r"\training_Dog_decease_Detection_system\dataset"
)

# ── Hyper-parameters (can be overridden via CLI) ─────────────────────────────
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
EPOCHS      = 25          # Phase 1 (frozen base)
FINE_EPOCHS = 10          # Phase 2 (unfrozen last layers)
LEARNING_RATE     = 1e-3
FINE_LEARNING_RATE = 1e-5
DROPOUT     = 0.4

CLASSES = [
    "demodicosis",
    "Dermatitis",
    "Fungal_infections",
    "Healthy",
    "Hypersensitivity",
    "ringworm",
]
NUM_CLASSES = len(CLASSES)


# ─────────────────────────────────────────────────────────────────────────────
#  1.  DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────
def build_data_loaders(base_dir: str, batch_size: int):
    """Return (train_ds, valid_ds, test_ds) as tf.data pipelines."""
    import tensorflow as tf

    AUTOTUNE = tf.data.AUTOTUNE

    def preprocess(img, label):
        img = tf.cast(img, tf.float32)
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        return img, label

    train_dir = os.path.join(base_dir, "train")
    valid_dir = os.path.join(base_dir, "valid")
    test_dir  = os.path.join(base_dir, "test")

    # ── Augmentation (only for training) ────────────────────────────────────
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomBrightness(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ], name="augmentation")

    def load_split(directory, augment=False):
        ds = tf.keras.utils.image_dataset_from_directory(
            directory,
            class_names=CLASSES,
            image_size=IMG_SIZE,
            batch_size=batch_size,
            shuffle=augment,
            seed=42,
            label_mode="categorical",
        )
        ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE)
        if augment:
            ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                        num_parallel_calls=AUTOTUNE)
        return ds.cache().prefetch(AUTOTUNE)

    train_ds = load_split(train_dir, augment=True)
    valid_ds = load_split(valid_dir, augment=False)
    test_ds  = load_split(test_dir,  augment=False)

    return train_ds, valid_ds, test_ds


# ─────────────────────────────────────────────────────────────────────────────
#  2.  MODEL
# ─────────────────────────────────────────────────────────────────────────────
def build_model(num_classes: int, dropout: float):
    """EfficientNetB3 with a custom classification head."""
    import tensorflow as tf

    base = tf.keras.applications.EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3),
        pooling=None,
    )
    base.trainable = False   # freeze during Phase 1

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout / 2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="DogDiseaseDetector")
    return model, base


# ─────────────────────────────────────────────────────────────────────────────
#  3.  TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def compile_and_train(model, base_model, train_ds, valid_ds,
                      epochs, fine_epochs, lr, fine_lr):
    import tensorflow as tf

    # ── Callbacks ────────────────────────────────────────────────────────────
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        "best_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True,
        verbose=1,
    )
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1,
    )

    # ── Phase 1 : Train head only ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 1 — Training classification head (base frozen)")
    print("=" * 60)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="categorical_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    model.summary()

    hist1 = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=epochs,
        callbacks=[ckpt, early_stop, lr_scheduler],
    )

    # ── Phase 2 : Fine-tune last 30 layers of base ───────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 2 — Fine-tuning (unfreezing last 30 base layers)")
    print("=" * 60)

    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(fine_lr),
        loss="categorical_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )

    ckpt2 = tf.keras.callbacks.ModelCheckpoint(
        "best_model_finetuned.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )
    early_stop2 = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    hist2 = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=fine_epochs,
        callbacks=[ckpt2, early_stop2, lr_scheduler],
    )

    return hist1, hist2


# ─────────────────────────────────────────────────────────────────────────────
#  4.  EVALUATION & PLOTS
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(model, test_ds):
    """Print classification report and confusion matrix."""
    import tensorflow as tf
    from sklearn.metrics import classification_report, confusion_matrix

    print("\n" + "=" * 60)
    print("  EVALUATION on TEST SET")
    print("=" * 60)

    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=CLASSES))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title("Confusion Matrix — Dog Disease Detection", fontsize=14, pad=12)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()
    print("  Saved → confusion_matrix.png")


def plot_history(hist1, hist2):
    """Plot training curves from both phases."""
    # Concatenate the two phase histories
    acc  = hist1.history["accuracy"]  + hist2.history["accuracy"]
    val_acc  = hist1.history["val_accuracy"]  + hist2.history["val_accuracy"]
    loss = hist1.history["loss"] + hist2.history["loss"]
    val_loss = hist1.history["val_loss"] + hist2.history["val_loss"]

    epochs_range = range(1, len(acc) + 1)
    phase2_start = len(hist1.history["accuracy"]) + 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(epochs_range, acc,     label="Train Accuracy",      color="#2196F3")
    axes[0].plot(epochs_range, val_acc, label="Validation Accuracy",  color="#FF5722",
                 linestyle="--")
    axes[0].axvline(phase2_start, color="gray", linestyle=":", label="Fine-tune start")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Loss
    axes[1].plot(epochs_range, loss,     label="Train Loss",      color="#2196F3")
    axes[1].plot(epochs_range, val_loss, label="Validation Loss",  color="#FF5722",
                 linestyle="--")
    axes[1].axvline(phase2_start, color="gray", linestyle=":", label="Fine-tune start")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle("Dog Disease Detection — Training Curves", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved → training_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
#  5.  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train Dog Disease Detection Model")
    parser.add_argument("--base_dir",    default=DEFAULT_DATASET)
    parser.add_argument("--epochs",      type=int,   default=EPOCHS)
    parser.add_argument("--fine_epochs", type=int,   default=FINE_EPOCHS)
    parser.add_argument("--batch_size",  type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",          type=float, default=LEARNING_RATE)
    parser.add_argument("--fine_lr",     type=float, default=FINE_LEARNING_RATE)
    parser.add_argument("--dropout",     type=float, default=DROPOUT)
    args = parser.parse_args()

    # ── GPU config ────────────────────────────────────────────────────────
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"  GPUs available: {[g.name for g in gpus]}")
    else:
        print("  No GPU found — running on CPU (training will be slower).")

    print(f"\n  Dataset  : {args.base_dir}")
    print(f"  Classes  : {CLASSES}")
    print(f"  Img size : {IMG_SIZE}")
    print(f"  Batch    : {args.batch_size}")
    print(f"  Epochs   : Phase1={args.epochs}  Phase2={args.fine_epochs}")

    # ── Load data ─────────────────────────────────────────────────────────
    train_ds, valid_ds, test_ds = build_data_loaders(args.base_dir, args.batch_size)

    # ── Build model ───────────────────────────────────────────────────────
    model, base_model = build_model(NUM_CLASSES, args.dropout)

    # ── Train ─────────────────────────────────────────────────────────────
    hist1, hist2 = compile_and_train(
        model, base_model, train_ds, valid_ds,
        args.epochs, args.fine_epochs, args.lr, args.fine_lr,
    )

    # ── Save final model ──────────────────────────────────────────────────
    model.save("dog_disease_model_final.keras")
    print("\n  Saved → dog_disease_model_final.keras")

    # ── Evaluate ──────────────────────────────────────────────────────────
    evaluate_model(model, test_ds)
    plot_history(hist1, hist2)

    print("\n  Done! Outputs: best_model.keras | best_model_finetuned.keras")
    print("               dog_disease_model_final.keras")
    print("               confusion_matrix.png | training_curves.png")


if __name__ == "__main__":
    main()