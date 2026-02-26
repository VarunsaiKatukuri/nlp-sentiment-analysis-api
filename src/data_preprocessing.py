import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    # here we are changing the texts to lower case
    text = text.lower()

    #here we are removing the html tags
    #for removing html tags we should use Regex patterns
    text = re.sub(r"<.*?>", "", text)

    #removing the punctuations
    text = text.translate(str.maketrans("", "", string.punctuation))

    #removing if there are any numbers
    text = re.sub(r"\d+", "", text)

    #tokenizing
    tokens = text.split()

    #removing stopwords and adding lemmatization
    cleaned_tokens = [
        lemmatizer.lemmatize(word) for word in tokens
        if word not in stop_words
    ]

    return " ".join(cleaned_tokens)

