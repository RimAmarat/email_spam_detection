# 📧 Spam Detection using Naive Bayes

This is a machine learning project that uses **Natural Language Processing (NLP)** and a **Naive Bayes classifier** to detect spam emails. The dataset used is a CSV file containing labeled emails as spam (`1`) or not spam (`0`).

## 🚀 Features

- Text preprocessing with NLTK
- Feature extraction using `CountVectorizer`
- Train/test split (80/20)
- Naive Bayes classification (`MultinomialNB`)
- Evaluation with accuracy, confusion matrix, and classification report

## 📁 Dataset

The dataset used: `spam_or_not_spam.csv`  
This CSV should include at least two columns:
- `email` (text content)
- `label` (0 for not spam, 1 for spam)

Upload it manually when prompted in Google Colab.

## 📊 Model Overview

- **Algorithm**: Multinomial Naive Bayes
- **Vectorization**: CountVectorizer using custom tokenizer (with punctuation and stopwords removal)
- **Accuracy**: Evaluated on both training and test datasets

## 🧠 Dependencies

- `nltk` for natural language processing
- `scikit-learn` for machine learning tools
- `pandas` and `numpy` for data handling

See the [`requirements.txt`](./requirements.txt) for full list.

## 🛠️ Setup Instructions

### ✅ Using Google Colab

1. Open the [notebook](https://colab.research.google.com/drive/1tE-1aEUiC4OAEFZghhEoUYqNeHYMxLiK?usp=sharing) in Google Colab.
2. Save a copy to GoogleColab.
3. Upload the dataset file `spam_or_not_spam.csv` when prompted.
4. Run all cells in sequence.

### 💻 Local Setup

1. Clone the repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Python script using:

```bash
python SpamDetection.py
```

## 🧹 Text Preprocessing

The text preprocessing function removes:
- Punctuation
- English stopwords

```python
def text_processing(text):
    rm_punc = [char for char in text if char not in string.punctuation]
    rm_punc = ''.join(rm_punc)
    clean_words = [word for word in rm_punc.split() if word.lower() not in stopwords.words('english')]
    return clean_words
```

## 📈 Evaluation Metrics

The model prints:
- **Classification Report** (Precision, Recall, F1-score)
- **Confusion Matrix**
- **Accuracy Score**

---

## 🧾 License

This project is for educational and research purposes.

## 🙌 Acknowledgements

- [NLTK](https://www.nltk.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [Google Colab](https://colab.research.google.com/)
