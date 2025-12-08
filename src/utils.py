import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def get_project_root() -> Path:
    """Returns the root directory of the project."""
    return Path(__file__).resolve().parent.parent

def plot_history(history, save_path: Path):
    """Plots training accuracy and loss and saves the figure."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.savefig(save_path)
    plt.close()

def print_evaluation_metrics(model, val_ds, class_names, save_path: Path):
    """
    Calculates metrics (Precision, Recall, F1) and saves a Confusion Matrix plot.
    """
    y_true = []
    y_pred = []

    print("\nGenerating evaluation report...")
    
    # Iterate over validation dataset to get predictions
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    # 1. Print Text Report (Rubric requirement: Precision, Recall, F1-Score)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("\n" + "="*60)
    print("FINAL MODEL EVALUATION METRICS")
    print("="*60)
    print(report)
    print("="*60)

    # 2. Generate and Save Confusion Matrix Plot (Professional formatting)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    plt.figure(figsize=(8, 8))
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()
    print(f"\nVisual Confusion Matrix saved to {save_path}")