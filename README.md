# 🖼️ Image Recognition and Classification Using CNNs

## 🌟 Overview
This project implements a **Convolutional Neural Network (CNN)** to classify images into three categories:  
**Water Bottles**, **Chairs**, and **Bags**.

The goal was to explore deep learning techniques for image recognition and object classification. Built with **TensorFlow** and **Keras**, this model leverages the power of CNNs to extract spatial features and achieve high accuracy in classification tasks.

This project was part of the code associated with **ENACT**, a mobility cane my team and I designed to assist users in identifying objects in their environment. The cane features a built-in camera that captures images and sends them to this model for classification. The program then deciphers the object in real-time, providing essential feedback to the user. This functionality is aimed at enhancing accessibility and mobility for individuals who rely on assistive devices. 🚶‍♂️✨  

---

## 🔑 Key Features
- **Model**: Built and trained a multi-layer CNN for image classification.
- **Dataset**:
  - Started with **300 images for training**.
  - Augmented and expanded to **3,000 images** for better accuracy.
- **Technologies Used**:
  - TensorFlow for ML training.
  - Keras for building and training the model.
  - NumPy for numerical operations.
- **Categories**:
  - Water Bottles, Chairs, and Bags.

---

## 🛠️ Tech Stack & Tools
- **Programming Language**: Python 🐍  
- **Hardware**: Raspberry Pi 🍓  
- **Cloud Platform**: Google Colab 🌐  

---

## 📊 Results
Training this model was an adventure! While the initial dataset was small, augmenting it significantly improved performance. Here's what I achieved:  
- **Training Images**: 3,000
- **Test Accuracy**: 96%  

---

### Challenges
While the model performs well in controlled conditions, it faces the following limitations:  
1. **Background Sensitivity**:  
   The CNN is influenced by the background of the objects. For instance, a **water bottle on a hardwood floor** might be recognized correctly, but the same bottle on a **marble floor** could cause misclassification.  
2. **Dataset Bias**:  
   Despite augmenting the dataset, the model still benefits from more diverse images to improve robustness against real-world variations.  

Addressing these challenges is crucial to making this model more adaptable and reliable in real-world scenarios.

---

## 🚀 Future Goals
- Expand the dataset with images in diverse environments and lighting conditions.- Expand the categories and dataset for broader use cases.  
- Explore **transfer learning** with pre-trained models like ResNet or MobileNet to improve generalization.

---
