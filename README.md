# Sign_Language_Detection
This repository contains the implementation of a Sign Language Detection system using Deep Learning techniques. The project aims to classify American Sign Language (ASL) alphabets (A-Y, excluding J) using Convolutional Neural Networks (CNNs) built with PyTorch and TensorFlow/Keras.
# 📋 Overview
The project demonstrates:
  - Preprocessing of the Sign Language MNIST dataset for training, validation, and testing.
  - Implementation of a custom CNN model for image classification.
  - Model training with performance evaluation, including accuracy metrics, loss curves, and classification reports.
  - Visualization of sample predictions and model architecture.
    
# 🛠 Features
  - Data Augmentation: Enhance dataset variety for robust model training.
  - Custom CNN Architecture: Includes layers like Conv2D, MaxPooling, Dropout, and Batch Normalization for optimal performance.
  - Evaluation Metrics: Provides detailed analysis through confusion matrices, classification reports, and accuracy/loss plots.
  - Model Visualization: Visualize the model architecture using visualkeras and TensorFlow.

# 🚀 Technologies Used
  - Languages: Python
  - Libraries: PyTorch, TensorFlow/Keras, Matplotlib, Seaborn, VisualKeras, Pandas, NumPy, Scikit-learn.

# 📂 Dataset
  - The project uses the Sign Language MNIST dataset, a modified version of the original MNIST dataset, adapted for American Sign Language alphabets.
  - Dataset Link: Sign Language MNIST on Kaggle

# 🎯 How It Works
1.Dataset Preprocessing:
  - Data normalization and reshaping.
  - Adjusting class labels for compatibility.
2.Model Training:
  - A PyTorch-based CNN architecture with dropout and ReLU activations.
  - Batch processing with custom DataLoader configurations.
3.Evaluation:
  - Compare training and validation performance over epochs.
  - Evaluate the test dataset with a detailed classification report.
4.Visualization:
  - Plot training/validation loss and accuracy curves.
  - Display random test predictions with true and predicted labels.

#🧑‍💻 Getting Started
Prerequisites:
  - Install required Python packages using:
    pip install -r requirements.txt
Running the Project:
1.Clone the repository:
    git clone https://github.com/yourusername/sign-language-detection.git
2.Navigate to the project directory:
    cd sign-language-detection
3.Execute the script:
    python main.py

# 🔍 Results
  - Accuracy: 95%
  - Model performs exceptionally well in classifying ASL alphabets, as depicted by evaluation metrics and visualization.

# 🙌 Acknowledgments
Special thanks to my mentors and peers for their support during this project.

# 🤝 Contributing
Contributions are welcome! Please feel free to fork this repository and submit a pull request.

