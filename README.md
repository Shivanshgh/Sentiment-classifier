# Sentiment & Text Classifier

A Python-based machine learning project for classifying text into categories such as **positive, negative, spam, and ham**.

## 🚀 Features

- Text cleaning and normalization
- URL and special-character removal
- Tokenization using NLTK
- Stop-word removal
- Word lemmatization
- TF-IDF feature extraction
- Unigram and bigram feature support
- Logistic Regression classification
- Train/test split with stratification
- Accuracy and classification report evaluation
- Model saving and loading using Joblib
- Prediction on new text inputs

## 🛠️ Technologies Used

- **Python**
- **NLTK**
- **Scikit-learn**
- **Joblib**
- **TF-IDF**
- **Logistic Regression**

## 📌 How It Works

The project first preprocesses the input text by converting it to lowercase, removing URLs and special characters, tokenizing the text, removing English stop words, and applying lemmatization.

The processed text is then converted into numerical features using **TF-IDF vectorization** with unigram and bigram features.

A **Logistic Regression** classifier is trained on these features to classify the text.

The trained machine learning pipeline is saved using **Joblib** and can later be loaded to make predictions on new text.

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
```

Move into the project directory:

```bash
cd YOUR-REPOSITORY
```

Install the required dependencies:

```bash
pip install nltk scikit-learn joblib
```

## ▶️ Run the Project

Run the Python file:

```bash
python text_classification.py
```

The program will:

1. Build the dataset
2. Clean and preprocess the text
3. Split the dataset into training and testing sets
4. Train the Logistic Regression model
5. Display accuracy and classification results
6. Save the trained model
7. Predict categories for new text

## 🧪 Example Input

```text
"This email offers you a great deal on new phones"
"I had an amazing dinner last night"
"Please call me back, it's urgent"
```

The model predicts the category for each input and displays the result.

## 📈 Model

The project uses:

**TF-IDF Vectorizer**

Converts text into numerical features based on the importance of words and word combinations.

**Logistic Regression**

Uses the extracted TF-IDF features to classify text into different categories.

## 🔮 Future Improvements

- Train the model on a larger real-world dataset
- Improve classification accuracy
- Add more sentiment categories
- Perform hyperparameter tuning
- Compare different machine learning algorithms
- Add confusion matrix visualization
- Build a Streamlit web interface
- Integrate a larger NLP dataset
- Explore transformer-based NLP models

## 👨‍💻 Author

**Shivansh Ghildiyal**
