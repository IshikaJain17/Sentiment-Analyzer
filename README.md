# Sentiment Analyzer

This project is a Sentiment Analyzer application that uses machine learning techniques to classify text reviews as positive or negative. It employs two classifiers: Naive Bayes and Support Vector Machine (SVM). The application features a graphical user interface (GUI) built with Tkinter for easy interaction.

## Features

- Text preprocessing with tokenization, stopword removal, and lemmatization.
- Sentiment classification using Naive Bayes and SVM classifiers.
- Confidence scores for predictions.
- User-friendly GUI for inputting reviews and displaying results.

## Requirements

- Python 3.x
- pandas
- nltk
- scikit-learn
- tk (Tkinter)

## Installation

1. Clone the repository or download the source code.
2. Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

3. Download the necessary NLTK data:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

## Usage

Run the `SentimentalAnalysis.py` script:

```bash
python SentimentalAnalysis.py
```

Enter your review text in the GUI and click the "Predict Sentiment" button to see the classification results from both classifiers.

## Dataset

The project uses a dataset file named `RawData.csv` which should be placed in the same directory as the script.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
