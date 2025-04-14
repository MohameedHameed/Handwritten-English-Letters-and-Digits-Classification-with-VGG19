# 🔤 Handwritten English Letters and Digits Classification with VGG19

This project classifies handwritten English **uppercase and lowercase letters** as well as **digits (0–9)** using a fine-tuned **VGG19** convolutional neural network. It also includes a simple **Flask web app** for real-time predictions.

---

## 🧠 Overview

- 🔡 **Classes:** A–Z, a–z, 0–9 (62 total)
- 🧠 **Model:** VGG19 (Transfer Learning)
- ⚙️ **Frameworks:** TensorFlow, Keras, Flask
- 🎯 **Goal:** Accurate character recognition from handwritten images

---
## 📁 Project Structure
/img/ ├── train/ ├── val/ └── test/ (each containing subfolders for each character class) /training.ipynb ← Training notebook
/main.py ← Flask web app for inference
/VGG19_v1_09_0.809.h5 ← Trained model file /templates/index.html ← Web app HTML interface

## 🚀 How to Run

### 1. 🏋️‍♂️ Train the Model
Open and run `training.ipynb` to train the VGG19 model on your dataset.
This uses the structure in `/img/train`, `/img/val`, and `/img/test`.

### 2. 🌐 Run the Web App

```bash
pip install flask tensorflow numpy
python main.py
Then open http://localhost:5000 in your browser and upload an image to classify.

## 📷 Example Classes
Digits: 0–9
Uppercase: A–Z
Lowercase: a–z

## 👨‍💻 Author
Developed by Mohammed Hameed
📬 For contact or collaboration, feel free to open an issue or fork the repo.
