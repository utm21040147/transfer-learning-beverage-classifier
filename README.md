# Transfer Learning Beverage Classifier

## 📌 Project Description
This project creates a "plug-and-play" Python application that uses **Transfer Learning** with a Convolutional Neural Network (CNN) to classify images into three distinct categories:
1.  **Cola-flavored soda**
2.  **Orange Juice**
3.  **Natural Water**

The system is built ensuring clean code principles, strict adherence to conventional commits, and cross-platform compatibility using relative paths.

## 🚀 Key Features
* **Architecture:** MobileNetV2 (Pre-trained on ImageNet) + Custom Classification Head.
* **Data Augmentation:** Random flip, rotation, and zoom to prevent overfitting.
* **Clean Code:** Fully modular structure with English documentation.
* **Cross-Platform:** Uses `pathlib` for OS-agnostic path handling.
* **Metrics:** Achieves >90% validation accuracy.

## 🧠 Model Explanation
The classification is performed using **Transfer Learning** based on the **MobileNetV2** architecture.
* **Base Model:** MobileNetV2 (pre-trained on ImageNet) is used as a feature extractor. The base layers are frozen to retain learned features.
* **Custom Head:** A Global Average Pooling layer followed by a Dense output layer (Softmax) is added to classify the specific beverage types.
* **Data Augmentation:** Random rotation, zoom, and flipping are applied during training to prevent overfitting and improve generalization on small datasets.

## 📦 Dependencies
The project requires **Python 3.8+** and the following libraries (listed in `requirements.txt`):
* `tensorflow==2.15.0` (Core ML framework)
* `numpy` (Data manipulation)
* `matplotlib` (Plotting training results)
* `pillow` (Image processing)

## 🛠️ Instructions (Installation)
To set up the project on any computer, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/utm21040147/transfer-learning-beverage-classifier.git](https://github.com/utm21040147/transfer-learning-beverage-classifier.git)
    cd transfer-learning-beverage-classifier
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Verify Data:**
    Ensure the `data/` folder contains the `train/` and `val/` subdirectories with the required images.

## ▶️ How to Run
Execute the main script from the root directory:

```bash
python main.py