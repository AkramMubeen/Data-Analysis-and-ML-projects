import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess(text):
  pattern_non_alnum = re.compile(r'[^a-z0-9\s]', re.IGNORECASE)
  pattern_digits = re.compile(r'\d')
  # Convert to lowercase
  text = text.lower()

  # Remove non-alphanumeric characters
  text = pattern_non_alnum.sub(' ', text)

  # Remove digits
  text = pattern_digits.sub('', text)

  # Remove extra white spaces
  text = re.sub(r'\s+', ' ', text).strip()

  # Tokenize
  tokens = word_tokenize(text)

  # Removing stopwords
  stop_words = set(stopwords.words('english'))
  tokens = [token for token in tokens if token not in stop_words]

  # Lemmatization
  lemmatizer = WordNetLemmatizer()
  tokens = [lemmatizer.lemmatize(token) for token in tokens]

  # Join tokens back into a single string
  text = ' '.join(tokens)

  return text