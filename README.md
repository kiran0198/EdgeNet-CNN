# ❄️ EdgeNet-CNN
### Edge Detection Using a Convolutional Neural Network Built Completely From Scratch

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

</p>

---

## 📖 Overview

**EdgeNet-CNN** is a lightweight Convolutional Neural Network for image edge detection implemented **entirely from scratch using NumPy**.

Unlike traditional deep learning projects that rely on frameworks such as PyTorch or TensorFlow, this project implements every major component manually to demonstrate how CNNs work internally.

The project includes a beautiful Streamlit interface for real-time image edge detection.

---

# ✨ Features

✅ CNN implemented completely from scratch

✅ Forward propagation

✅ Backpropagation

✅ Custom Convolution Layer

✅ ReLU Activation

✅ Sigmoid Activation

✅ Binary Cross Entropy Loss

✅ Stochastic Gradient Descent Optimizer

✅ im2col / col2im implementation

✅ Gradient Checking

✅ Image Prediction

✅ Streamlit Web App

---

# 🧠 Architecture

```
Input Image
      │
      ▼
Conv2D (3×3)
      │
      ▼
ReLU
      │
      ▼
Conv2D (3×3)
      │
      ▼
Sigmoid
      │
      ▼
Edge Map
```

---

# 📂 Project Structure

```text
EdgeNet-CNN
│
├── dataset/
│
├── layers/
│   ├── conv2d.py
│   ├── relu.py
│   └── sigmoid.py
│
├── losses/
│   └── bce.py
│
├── optim/
│   └── sgd.py
│
├── utils/
│   ├── im2col.py
│   ├── col2im.py
│   ├── padding.py
│   └── dataloader.py
│
├── model.py
├── train.py
├── predict.py
├── app.py
├── style.css
├── snow.html
├── best_model.pkl
├── requirements.txt
└── README.md
```

---

# 🚀 Installation

```bash
git clone https://github.com/YOUR_USERNAME/EdgeNet-CNN.git

cd EdgeNet-CNN

pip install -r requirements.txt
```

---

# ▶️ Training

```bash
python train.py
```

---

# 🔍 Prediction

```bash
python predict.py
```

---

# 🌐 Streamlit Demo

```bash
streamlit run app.py
```

---

# 📊 Dataset

This project is trained using the **BSDS500 (Berkeley Segmentation Dataset)**.

Download the dataset and place it inside

```
dataset/
```

---

# 📷 Results

| Original | Edge Prediction |
|----------|-----------------|
| *(Original)* | *(Edge Prediction)* |

---

# 🎯 Learning Objectives

This project demonstrates:

- CNN implementation from scratch
- Matrix-based convolution
- Gradient computation
- Backpropagation
- Edge detection
- Neural network optimization
- Computer Vision fundamentals

---

# 🛠 Technologies

- Python
- NumPy
- OpenCV
- Streamlit

---

# 👨‍💻 Author

**Kiran Kumar Sahu**

M.Tech (Computer Science & Engineering)

AI • Deep Learning • Computer Vision

GitHub: [https://github.com/kiran0198](https://github.com/kiran0198)

---

# ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub.

It helps others discover the project and motivates future improvements.

---

# 📄 License

This project is released under the MIT License.
