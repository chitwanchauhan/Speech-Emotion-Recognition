import os
import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# 1. Load audio file paths and labels
paths = []
labels = []

for dirname, _, filenames in os.walk('C:\Chitwan\LNCT\Projects\SER_Fullstack\TESS Toronto emotional speech set data'):  # <-- change this to your folder
    for filename in filenames:
        if filename.endswith('.wav'):
            paths.append(os.path.join(dirname, filename))
            label = filename.split('_')[-1].split('.')[0]
            labels.append(label.lower())

print("✅ Dataset is loaded.")

# 2. Create DataFrame
print(f"Total audio files found: {len(paths)}")


df = pd.DataFrame({'speech': paths, 'label': labels})
print(df.head())
print(df['label'].value_counts())

# 3. Extract features
def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

X_mfcc = df['speech'].apply(lambda x: extract_mfcc(x))
X = np.array(X_mfcc.tolist())  # Convert list of arrays to 2D array
X = np.expand_dims(X, -1)  # Add one more dimension for LSTM

# 4. One-hot encode labels
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(df[['label']])
y = np.array(y)

# 5. Build model
model = Sequential([
    LSTM(128, input_shape=(40, 1), return_sequences=False),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(y.shape[1], activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 6. Train model
print("🚀 Training started...")
history = model.fit(X, y, epochs=50, batch_size=128, validation_split=0.2, verbose=1)
print("✅ Training completed.")

# 7. Save model
model.save("model/ser_model.h5")
print("💾 Model saved as 'ser_model.h5' in 'model/' folder.")

# 8. Optional: Plot accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.show()
