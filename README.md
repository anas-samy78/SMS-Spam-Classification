# ✉️ SMS Spam Classification with Deep Learning

An end-to-end NLP project that classifies SMS messages as **Spam** or **Not Spam (Ham)**. It covers the whole pipeline: data preprocessing, building a vocabulary, using pre-trained **GloVe** embeddings, training and comparing several recurrent architectures (**RNN / LSTM / GRU / BiLSTM**) in **PyTorch**, and serving the best model through an interactive **Streamlit** web app.

Live Demo: https://sms-spam-classification-anas-samy.streamlit.app/

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Workflow](#-project-workflow)
- [Models Explored](#-models-explored)
- [Final Model Architecture](#-final-model-architecture)
- [Results](#-results)
- [Web App](https://sms-spam-classification-anas-samy.streamlit.app/)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Limitations & Future Work](#-limitations--future-work)
- [Author](#-author)

---

## 🔎 Overview

Spam messages waste time and can be used for phishing and fraud. This project builds a text classifier that reads a raw SMS and predicts whether it is spam, together with the probability of each class.

The project was built **from scratch**, not only the web interface:

1. Data loading and cleaning
2. Text tokenization and vocabulary construction
3. Embedding matrix creation from pre-trained GloVe vectors
4. Training and evaluating multiple deep learning architectures
5. Comparing the models and selecting the best one
6. Saving the trained model and deploying it in a web app

---

## ✨ Features

- Spam / Not Spam prediction with **confidence scores** for both classes
- Pre-trained **GloVe** word embeddings for better semantic understanding
- Comparison of several recurrent architectures (RNN, LSTM, GRU, BiLSTM)
- Handling of variable-length messages with **packed sequences**
- Clean, responsive **Streamlit** interface with one-click example messages
- Model cached in memory for fast, repeated predictions

---

## 🛠 Tech Stack

| Area | Tools |
|------|-------|
| Language | Python |
| Deep Learning | PyTorch |
| Embeddings | GloVe (pre-trained word vectors) |
| Data Handling | NumPy, Pandas |
| Evaluation | scikit-learn |
| Web App | Streamlit (custom HTML/CSS styling) |

---

## 🔄 Project Workflow

```
Raw SMS data
     │
     ▼
Cleaning & tokenization
     │
     ▼
Vocabulary building (<PAD>, <UNK> tokens)
     │
     ▼
GloVe embedding matrix
     │
     ▼
Train / validation / test split
     │
     ▼
Train multiple models (RNN, LSTM, GRU, BiLSTM)
     │
     ▼
Evaluate & compare
     │
     ▼
Save best model (.pth checkpoint)
     │
     ▼
Streamlit web app
```

### 1. Data preprocessing
- Load the SMS dataset and map labels to `0` (Not Spam) and `1` (Spam).
- Tokenize each message into words.
- Build a vocabulary that maps each word to an integer index, with special tokens for padding (`<PAD>`) and unknown words (`<UNK>`).

### 2. GloVe embeddings
- Load the pre-trained GloVe vectors.
- Build an **embedding matrix** aligned with the project vocabulary, so each word index points to its GloVe vector.
- Words missing from GloVe fall back to a default vector.
- The matrix initializes the model's `nn.Embedding` layer.

### 3. Training
- Cross-entropy loss with an Adam optimizer.
- Variable-length batches handled with `pack_padded_sequence`.
- Each architecture was trained and evaluated on the same splits for a fair comparison.

---

## 🧪 Models Explored

Several architectures were implemented and compared:

| Model | Description |
|-------|-------------|
| **RNN** | Baseline vanilla recurrent network |
| **LSTM** | Handles longer dependencies with gated memory cells |
| **GRU** | Lighter gated architecture with fewer parameters |
| **BiLSTM** | Reads the sequence in both directions, capturing context before and after each word |

All models use the same GloVe-initialized embedding layer, so the comparison focuses on the recurrent architecture.

---

## 🧠 Final Model Architecture

The deployed model is a **Bidirectional LSTM**:

```
Input (word indices)
   │
   ▼
Embedding layer (initialized with GloVe)
   │
   ▼
Bidirectional LSTM (packed sequences)
   │
   ▼
Concatenate final forward + backward hidden states
   │
   ▼
Fully connected layer
   │
   ▼
Softmax → [P(Not Spam), P(Spam)]
```

```python
class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, embedding_matrix):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
```

The final hidden states of the forward and backward passes are concatenated before the classification layer. The saved checkpoint (`bilstm_spam.pth`) contains the model weights, the vocabulary, the embedding matrix, and the hyperparameters.

---

## 📊 Results

> Replace the placeholder values below with your actual metrics.

| Model | Accuracy | Precision | Recall | F1-score |
|-------|----------|-----------|--------|----------|
| RNN | – | – | – | – |
| LSTM | – | – | – | – |
| GRU | – | – | – | – |
| **BiLSTM** | – | – | – | – |

**Why BiLSTM?** Add a short note here on why you selected it (for example: best F1-score on the test set).

---

## 🌐 Web App

The Streamlit app lets you:

- Paste or type an SMS message (up to 500 characters)
- Click **Check Message** to get a prediction
- See the predicted label and the probability of each class
- Try ready-made **example messages** with one click

The UI uses custom CSS: a fixed navigation bar, card layout, gradient buttons, and a responsive design for mobile screens.

---

## 📁 Project Structure

> Adjust the names to match your repository.

```
.
├── sms_spam_web_app.py        # Streamlit web app
├── bilstm_spam.pth            # Saved model checkpoint
├── notebooks/
│   └── sms_spam_training.ipynb   # Data prep, GloVe, training, evaluation
├── data/
│   └── spam.csv               # SMS dataset
├── assets/
│   └── app_screenshot.png     # Screenshot used in this README
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. (Optional) create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

Example `requirements.txt`:

```
torch
streamlit
numpy
pandas
scikit-learn
```

> Use a recent Streamlit version (`pip install -U streamlit`). The app relies on keyed containers and button icons, which older releases do not support.

---

## ▶️ Usage

Make sure `bilstm_spam.pth` is in the same folder as the app, then run:

```bash
streamlit run sms_spam_web_app.py
```

The app opens at `http://localhost:8501`.

### Using the model in Python

```python
model, vocab = load_model()
p_not_spam, p_spam = predict_spam(model, vocab, "Congratulations! You won a free prize.")
print(f"Spam probability: {p_spam:.2%}")
```

---

## 🚧 Limitations & Future Work

- Tokenization is simple whitespace splitting. Handling punctuation, casing, and URLs more carefully could improve accuracy.
- The dataset is small and in English, so the model may not generalize to other languages or newer spam styles.
- Possible improvements:
  - Add an attention mechanism on top of the BiLSTM
  - Try transformer-based models (e.g., DistilBERT)
  - Handle class imbalance with class weights or oversampling
  - Deploy the app online (Streamlit Community Cloud, Hugging Face Spaces)

---

## 👤 Author

**Your Name**
- GitHub: [anas-samy78](https://github.com/anas-samy78)
- LinkedIn: (https://www.linkedin.com/in/anas-samy-907072320/)

---

## 📄 License

This project is licensed under the MIT License. Feel free to use and modify it.
