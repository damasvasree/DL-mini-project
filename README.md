# 🏃 Human Activity Recognition using 1D-CNN + LSTM

![Python](https://img.shields.io/badge/Python-3.10--3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-92.06%25-success)
![License](https://img.shields.io/badge/License-MIT-green)

A deep learning system that recognizes **six everyday human activities** from smartphone **accelerometer and gyroscope** signals, using a hybrid **1D-CNN + LSTM** network, with an interactive **Streamlit web app**.

> Mini project for **Deep Learning (24UAD511)**, B.Tech Artificial Intelligence and Data Science, KGiSL Institute of Technology.

---

## 📌 Overview

Human Activity Recognition (HAR) identifies what a person is doing from wearable or smartphone sensor data. Instead of hand-crafted features, this project learns patterns directly from raw signals:

- **1D-CNN** extracts local movement patterns from the sensor signals.
- **LSTM** learns how those patterns change over time.
- **Softmax** outputs a probability for each activity.

**Activities recognized:** Walking · Walking Upstairs · Walking Downstairs · Sitting · Standing · Laying

---

## ✨ Features

- 🎯 **Live prediction:** pick a sample and see the predicted activity, confidence and sensor graphs
- 🎲 Random sample button and CSV upload for your own sensor windows
- 🧪 **Test all samples** in one click with per-activity accuracy
- 📊 **Model performance** dashboard: classification report and confusion matrix
- 🌙 Dark themed, colourful UI built with Streamlit and Altair

---

## 📸 Screenshots

| Live Prediction | Model Performance |
|---|---|
| ![Live Prediction](screenshots/live_prediction.png) | ![Performance](screenshots/performance.png) |

---

## 📂 Dataset

[UCI Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)

- 30 volunteers (age 19 to 48) wearing a smartphone on the waist
- Accelerometer and gyroscope sampled at 50 Hz
- Windows of 2.56 s (**128 time steps**) with 50% overlap
- This project uses 6 raw channels: body acc X/Y/Z and gyro X/Y/Z

| Split | Samples | Input shape |
|---|---|---|
| Train | 7,352 | 7352 × 128 × 6 |
| Test | 2,947 | 2947 × 128 × 6 |

20% of the training data is used for validation during training.

---

## 🧠 Model Architecture

| Layer | Output Shape | Parameters |
|---|---|---|
| Input | 128 × 6 | 0 |
| Conv1D (64, kernel 3, ReLU) | 126 × 64 | 1,216 |
| MaxPooling1D (2) | 63 × 64 | 0 |
| Conv1D (128, kernel 3, ReLU) | 61 × 128 | 24,704 |
| MaxPooling1D (2) | 30 × 128 | 0 |
| LSTM (64) | 64 | 49,408 |
| Dropout (0.5) | 64 | 0 |
| Dense (64, ReLU) | 64 | 4,160 |
| Dense (6, Softmax) | 6 | 390 |
| **Total** | | **79,878** |

**Training:** Adam optimizer, categorical cross-entropy, batch size 64, 30 epochs.
**Preprocessing:** StandardScaler (fitted on training data only), labels shifted to 0 to 5 and one-hot encoded.

---

## 📈 Results

Evaluated on the 2,947 held-out test samples (volunteers not seen during training):

| Metric | Value |
|---|---|
| **Test accuracy** | **92.06%** |
| Macro precision / recall / F1 | 0.92 / 0.92 / 0.92 |

| Activity | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| WALKING | 0.99 | 0.95 | 0.97 | 496 |
| WALKING_UPSTAIRS | 0.98 | 0.94 | 0.96 | 471 |
| WALKING_DOWNSTAIRS | 0.89 | 0.98 | 0.93 | 420 |
| SITTING | 0.83 | 0.92 | 0.87 | 491 |
| STANDING | 0.90 | 0.93 | 0.92 | 532 |
| LAYING | 0.94 | 0.82 | 0.88 | 537 |

Almost all errors happen **inside a group** (walking variants with each other, and sitting/standing/laying with each other). Only 3 of 234 errors mix a moving and a static activity. The static postures are harder to separate because gravity is removed from the body-acceleration signals used here.

> Numbers come from the project report. A freshly retrained model may differ by a small amount.

---

## 🗂️ Project Structure

```
human-activity-recognition/
├── app.py                  # Streamlit web app
├── train_model.py          # Download data, train, evaluate, save model
├── requirements.txt
├── models/
│   ├── human_activity_cnn_lstm.keras   # trained model
│   ├── human_activity_scaler.pkl       # fitted StandardScaler
│   └── demo_samples.npz                # demo windows for the app
├── .streamlit/config.toml  # theme and server settings
├── screenshots/
├── LICENSE
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/human-activity-recognition.git
cd human-activity-recognition
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
Use Python 3.10 to 3.12.

### 4. Run the app
```bash
streamlit run app.py
```
Open **http://localhost:8501** in your browser.

---

## 🔁 Train the model yourself (optional)

The `models/` folder already contains a trained model. To retrain:

```bash
python train_model.py
```
This downloads the UCI HAR dataset, trains for 30 epochs, prints the test accuracy and classification report, and saves the 3 files into `models/`.

**On Google Colab:** upload `train_model.py`, run `!python train_model.py` (use a T4 GPU for faster training). The 3 output files are downloaded automatically.

---

## 📤 Using your own data (CSV upload)

In the app, choose **Upload CSV** and upload a file with **128 rows × 6 columns** in this order:

```
acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
```
Header is optional. Use the same units as the UCI HAR body-acceleration (g) and gyroscope (rad/s) signals.

---

## ⚠️ Limitations

- Uses only 6 channels (body accelerometer and gyroscope); gravity-inclusive acceleration is not used
- Validation split is the last 20% of the training data, not volunteer-wise
- No early stopping or hyperparameter search; results from a single run
- Trained on 30 volunteers with the phone on the waist; other users or phone positions were not tested
- Evaluated offline on pre-segmented windows, not on live streaming data

**Ethics:** activity data is personal. Real deployments need informed consent, secure storage and preferably on-device processing. This is a decision-support tool, not for medical or safety decisions.

---

## 🔮 Future Scope

- Real-time prediction from a live smartphone sensor stream
- More activities (running, cycling, jumping) and transitions
- Gravity-inclusive total acceleration and extra sensors
- Bi-LSTM, GRU and Transformer models
- Early stopping, volunteer-wise validation, data augmentation, cross-validation
- TensorFlow Lite conversion for on-device mobile deployment
- Explainability: which channels and time steps drive each prediction

---

## 👥 Team

| Roll No. | Name |
|---|---|
| 711724UAD111 | Anitha K |
| 711724UAD118 | Bavithra S |
| 711724UAD120 | Damasvasree N |
| 711724UAD136 | Hanisha V |
| 711724UAD154 | Madhumitha S |

KGiSL Institute of Technology, Department of Artificial Intelligence and Data Science

---

## 📚 References

1. D. Anguita et al., "A public domain dataset for human activity recognition using smartphones," ESANN, 2013.
2. C. A. Ronao and S.-B. Cho, "Human activity recognition with smartphone sensors using deep learning neural networks," Expert Systems with Applications, 2016.
3. F. J. Ordóñez and D. Roggen, "Deep convolutional and LSTM recurrent neural networks for multimodal wearable activity recognition," Sensors, 2016.
4. S. Hochreiter and J. Schmidhuber, "Long short-term memory," Neural Computation, 1997.
5. J. R. Kwapisz et al., "Activity recognition using cell phone accelerometers," ACM SIGKDD Explorations Newsletter, 2011.

## 📄 License

Released under the [MIT License](LICENSE).
