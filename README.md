# Speech-Emotion-Recognition

This project implements a Speech Emotion Recognition (SER) system using machine learning and deep learning techniques. It classifies emotions from speech audio samples using Mel-frequency cepstral coefficients (MFCCs) and a deep learning model.

---

## 📌 Features

- Audio preprocessing with `librosa`
- Emotion classification using a neural network
- Visualizations of waveforms and MFCCs
- Dataset: [RAVDESS](https://zenodo.org/record/1188976)
- Implemented and tested in Google Colab

---

## 🧠 Emotions Recognized

- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- Librosa
- NumPy / Pandas
- Matplotlib / Seaborn

---

## 📁 Dataset

We use the [RAVDESS dataset](https://zenodo.org/record/1188976), which contains 24 professional actors (12 male, 12 female) vocalizing two lexically-matched statements in a neutral North American accent.

Download and extract it into the `audio_files/` directory.

---

## 🧪 Model Training Steps

1. **Extract Features**: Using MFCC, Chroma, Mel spectrogram
2. **Label Encoding**: Categorical labels for output
3. **Train/Test Split**: 80/20 split
4. **Model Architecture**: Fully connected neural network
5. **Evaluation**: Accuracy on test data

---

## 🚀 How to Run

1. Clone this repo:
    ```bash
    git clone https://github.com/chitwanchauhan/Speech-Emotion-Recognition.git
    cd Speech-Emotion-Recognition
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the notebook:
    Open `notebook5569b2ff1e.ipynb` in Jupyter or Google Colab

---

## 📊 Results

The model achieves up to **85–90% accuracy** depending on the number of emotion classes and data samples used.

---

## 📈 Sample Output

- Confusion Matrix
- Accuracy/Loss plots
- MFCC waveform plots

---

## 🧑‍💻 Author

**Chitwan Chauhan**  
[GitHub Profile](https://github.com/chitwanchauhan)

---

## 📄 License

This project is licensed under the MIT License.
