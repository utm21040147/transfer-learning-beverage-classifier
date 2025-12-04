from pathlib import Path
from src.classifier import BeverageClassifier
from src.utils import plot_history, get_project_root

def main():
    # 1. Setup Paths
    root_dir = get_project_root()
    data_dir = root_dir / 'data'
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    models_dir = root_dir / 'models'
    
    # Create models directory if it does not exist
    models_dir.mkdir(exist_ok=True)

    # 2. Verify Data Existence
    if not train_dir.exists() or not val_dir.exists():
        print(f"Error: Data directories not found at {data_dir}")
        print("Please ensure 'data/train' and 'data/val' exist with images.")
        return

    # 3. Initialize Classifier
    classifier = BeverageClassifier()
    
    # 4. Load Datasets
    print("Loading datasets...")
    train_ds = classifier.create_dataset(train_dir)
    val_ds = classifier.create_dataset(val_dir)

    # 5. Train Model
    print("Starting training with Transfer Learning...")
    history = classifier.train(train_ds, val_ds, epochs=25)

    # 6. Save Results
    model_path = models_dir / 'beverage_model.keras'
    plot_path = models_dir / 'training_results.png'
    
    classifier.save_model(model_path)
    plot_history(history, plot_path)
    
    print(f"Training complete. Model saved to {model_path}")

if __name__ == "__main__":
    main()