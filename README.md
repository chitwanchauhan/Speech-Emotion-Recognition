# 🎙️ Speech Emotion Recognition using LSTM and MFCC

This project focuses on recognizing human emotions from speech signals using a deep learning approach. The system uses **MFCC (Mel-frequency cepstral coefficients)** for feature extraction and an **LSTM (Long Short-Term Memory)** model for classification. The goal is to classify speech into different emotions such as *happy*, *sad*, *angry*, *fear*, *disgust*, and *neutral*.

---

## 📂 Dataset

The model uses the **TESS (Toronto Emotional Speech Set)** dataset, which consists of female voice recordings labeled with different emotions.

📥 **Download it here**:  
🔗 [TESS Dataset on Kaggle](https://www.kaggle.com/datasets/manikantagade/tess-dataset)

> 📌 Note: Make sure to exvract the dataset and update your local directory path accordingly in the code.

---

## 🧠 Model Architecture

The model uses:
- **MFCC** features (40 coefficients)
- **LSTM** layer for sequential learning
- **Dense** layers with ReLU activation
- **Dropout** for regularization
- **Softmax** for final emotion classification

```python
Sequential([
    LSTM(123, input_shape=(40,1)),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(7, activation='softmax')
])




📈 Training Details
Epochs: 100

Batch size: 512

Loss function: Categorical Crossentropy

Optimizer: Adam

Validation split: 20%

Training and validation accuracy/loss are visualized using matplotlib.
