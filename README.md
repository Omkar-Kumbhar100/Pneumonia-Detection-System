# 🫁 Pneumonia Detection using ResNet-18

A Deep Learning project that detects Pneumonia from Chest X-ray images using a pretrained ResNet-18 model with Transfer Learning.

---

## 📌 Features

- Chest X-ray image classification
- Predicts:
  - NORMAL
  - PNEUMONIA
- Confidence score
- Streamlit Web Application
- Model evaluation using Accuracy, Precision, Recall and F1-score

---

## 🧠 Model

- ResNet-18 (Pretrained on ImageNet)
- Transfer Learning
- PyTorch

---

## 📂 Dataset

Chest X-Ray Pneumonia Dataset

Classes:
- NORMAL
- PNEUMONIA

---

## 🛠 Technologies Used

- Python
- PyTorch
- Torchvision
- Streamlit
- NumPy
- Matplotlib
- Scikit-learn

---

## 🚀 How to Run



Install dependencies

```bash
pip install -r requirements.txt
```

Train the model

```bash
python train.py
```

Evaluate

```bash
python evaluate.py
```

Predict

```bash
python predict.py
```

Run Streamlit App

```bash
streamlit run app.py
```

---

## 📊 Output

The model predicts:

- ✅ NORMAL
- ✅ PNEUMONIA

along with confidence score and probability chart.

---



## 👨‍💻 Author

Omkar Kumbhar.