
#  Speech Emotion Recognition using LSTM and MFCC

This project focuses on recognizing human emotions from speech signals using a deep learning approach. The system uses **MFCC (Mel-frequency cepstral coefficients)** for feature extraction and an **LSTM (Long Short-Term Memory)** model for classification. The goal is to classify speech into different emotions such as *happy*, *sad*, *angry*, *fear*, *disgust*, and *neutral*.

---

## Dataset

The model uses the **TESS (Toronto Emotional Speech Set)** dataset, which consists of female voice recordings labeled with different emotions.

 **Download it here**:  
 [TESS Dataset on Kaggle](https://www.kaggle.com/datasets/manikantagade/tess-dataset)

>  Note: Make sure to extract the dataset and update your local directory path accordingly in the code.

---

## Features

- Upload WAV audio files for emotion analysis
- Playback uploaded audio files directly in the app
- Predicts emotions with a single click
- Clean and intuitive user interface

## How It Works

1. User uploads a WAV audio file
2. The app saves the file temporarily
3. User clicks "Predict Emotion" button
4. The app analyzes the audio and displays the predicted emotion

## Setup Instructions

### Prerequisites

- Python 3.7+
- pip package manager

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/speech-emotion-recognition.git
   cd speech-emotion-recognition
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. Open your browser and navigate to `http://localhost:8501`

## File Structure

```
## Project Structure
speech-emotion-recognition/
├── app.py                  # Main Streamlit application
├── training/
│ ├── train_model.py        # Script to train the emotion recognition model
│ └── TESS dataset          # Toronto Emotional Speech Set dataset
├── model/
│ └── ser.model.h5          # Pretrained speech emotion recognition model
├── utils/
│ └── predict.py            # Emotion prediction logic
├── requirements.txt        # Python dependencies
└── README.md 
```

## Requirements

The application requires the following Python packages (should be listed in requirements.txt):

```
streamlit
[other dependencies your predict.py uses]
```

## Usage Notes

- Currently only supports WAV audio files
- For best results, use clear speech recordings
- The model works best with recordings of 3-10 seconds in length

![image](https://github.com/user-attachments/assets/d670f370-2b5b-4694-9ab9-7cd9fa9f15a5)
![image](https://github.com/user-attachments/assets/e2d97779-8c9d-489c-9234-7e559b92fb41)
![image](https://github.com/user-attachments/assets/de037a96-ed55-489c-ad1d-54811727874f)



