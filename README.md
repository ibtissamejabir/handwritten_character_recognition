# ✍️ Handwritten Character Recognition with Deep Learning

A deep learning project that recognizes handwritten digits and letters using a Convolutional Neural Network (CNN) built with PyTorch. The project combines model training, evaluation, and an interactive web application that allows users to draw a character and receive the model's top predictions.

## 🚀 Project Overview

Handwritten character recognition is a computer vision task that involves classifying handwritten symbols from images. This project uses the **EMNIST ByClass dataset** to train a CNN capable of recognizing digits and letters.

The trained model is integrated into a Streamlit application where users can draw a character on a canvas, visualize the preprocessed image, and explore the model's top five predictions with their confidence scores.

## ✨ Features

- **Deep Learning:** Custom CNN architecture implemented using PyTorch.
- **Multi-Class Classification:** Recognizes digits and letters across 46 output classes.
- **Data Augmentation:** Applies random rotations, scaling, and translations to improve generalization.
- **Model Evaluation:** Measures accuracy, loss, precision, recall, and F1-score.
- **Interactive Drawing Canvas:** Draw handwritten characters directly in the web application.
- **Top-5 Predictions:** Displays the five most likely classes and their predicted probabilities.
- **Image Preprocessing:** Converts drawings into normalized 28 × 28 grayscale images.
- **Model Checkpointing:** Saves the best-performing model based on validation accuracy.
- **GPU Acceleration:** Supports CUDA-enabled GPUs when available.

## 📊 Results

The model was trained and evaluated on the EMNIST ByClass dataset.

| Metric | Result |
|---|---:|
| Test accuracy | **90.82%** |
| Best validation accuracy | **90.86%** |
| Test loss | 0.3304 |
| Training epochs | 25 |
| Number of output classes | 46 |
| Model parameters | 2,348,910 |

The dataset contains 697,932 training examples and 116,323 test examples. The training data was split into training and validation subsets, with approximately 10% reserved for validation.

These results are from the recorded notebook run and may vary when training is repeated.

## 🧠 Model Architecture

The model is a custom Convolutional Neural Network designed to extract visual features from handwritten characters.

The architecture consists of:

1. **Convolutional feature extraction:** Three convolutional blocks with 64, 128, and 256 channels.
2. **Batch normalization and ReLU:** Improve training stability and introduce non-linearity.
3. **Max pooling:** Reduces the spatial dimensions of feature maps.
4. **Dropout:** Helps reduce overfitting.
5. **Fully connected layers:** Map the extracted features to the final 46-class prediction.

### Training configuration

- **Optimizer:** AdamW
- **Loss function:** Cross-Entropy Loss with label smoothing
- **Learning rate scheduler:** OneCycleLR
- **Batch size:** 256
- **Epochs:** 25
- **Random seed:** 42
- **Data augmentation:** Rotation, scaling, and translation
- **Mixed-precision training:** Enabled when using CUDA

## 🗂️ Dataset

This project uses the **EMNIST ByClass** dataset, available through `torchvision.datasets.EMNIST`.

The dataset contains handwritten digits and uppercase and lowercase English letters. The original 62 classes are mapped to 46 output classes by merging selected uppercase and lowercase letter pairs with similar visual shapes.

Examples include `C/c`, `I/i`, `O/o`, and `S/s`.

This mapping reduces the classification complexity but means that the model does not always distinguish between uppercase and lowercase forms.

The dataset is downloaded automatically by the training notebook.

## 🛠️ Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- OpenCV
- Streamlit
- Streamlit Drawable Canvas
- Matplotlib
- Scikit-learn
- Pillow

## 📁 Project Structure

```text
handwritten-character-recognition/
│
├── handwritten_character_recognition.ipynb
├── app.py
├── emnist_cnn_case_merged_best.pth
├── requirements.txt
├── .gitignore
└── README.md
```

The notebook contains the data preparation, model training, validation, and evaluation workflow.

The Streamlit application loads the trained checkpoint and provides an interactive interface for making predictions.

The `.pth` file is the saved model checkpoint. Make sure the checkpoint is available locally before running the application.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

Replace the URL with your actual GitHub repository URL.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

Create a `requirements.txt` file containing:

```text
torch
torchvision
numpy
opencv-python-headless
Pillow
streamlit
streamlit-drawable-canvas
matplotlib
scikit-learn
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

For GPU training, install the appropriate PyTorch version for your CUDA environment using the official [PyTorch installation guide](https://pytorch.org/get-started/locally/).

### 4. Prepare the trained model

Place the trained checkpoint in your project directory:

```text
emnist_cnn_case_merged_best.pth
```

The checkpoint is generated by the notebook during training.

**Important:** Update the checkpoint path in `app.py` to use a relative path or a configurable environment variable rather than a machine-specific absolute path.

### 5. Launch the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal to access the application.

## 🖥️ How to Use the Application

1. Launch the Streamlit application.
2. Draw a single digit or letter on the drawing canvas.
3. View the preprocessed 28 × 28 image used by the model.
4. Examine the top five predicted classes.
5. Compare the prediction scores to understand which classes the model considers most likely.

For best results, draw one character at a time using a clear, reasonably thick stroke.

## 🔍 Evaluation and Challenges

The model achieves over 90% accuracy on the test dataset, but some visually similar characters remain difficult to distinguish.

Examples include:

- `0` and `O`
- `1`, `I`, and `L`
- `5` and `S`
- Uppercase and lowercase letters with similar shapes

The confusion matrix and classification report in the notebook help identify these weaknesses and provide directions for future improvements.

## 🔮 Future Improvements

- Improve recognition of visually similar digits and letters.
- Experiment with alternative CNN architectures and hyperparameters.
- Add image centering and bounding-box-based preprocessing.
- Compare performance across individual classes.
- Improve robustness to different handwriting styles.
- Deploy the application using Streamlit Community Cloud or another hosting platform.
- Add automated tests and a reproducible training configuration.

## 🎯 Learning Outcomes

This project demonstrates practical experience with:

- Computer vision and image classification.
- CNN architecture design using PyTorch.
- Dataset preprocessing and label mapping.
- Data augmentation and model regularization.
- GPU-accelerated model training.
- Evaluation using classification metrics and confusion matrices.
- Integrating a trained deep learning model into an interactive web application.

## 👩‍💻 Author

**Ibtissame Jabir**

Artificial Intelligence Engineering Graduate | Machine Learning | Deep Learning | Computer Vision

[GitHub](https://github.com/ibtissamejabir) · [LinkedIn](https://www.linkedin.com/in/ibtissame-jabir/)

---

*Built as a hands-on deep learning project to explore handwritten character classification and interactive AI applications.*
