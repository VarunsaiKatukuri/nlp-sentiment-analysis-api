from sklearn.feature_extraction.text import TfidfVectorizer

def create_tfidf_vectorizer(max_features=10000):
    vectorizer = TfidfVectorizer(
        max_features = max_features,
        ngram_range = (1,2), #here we are using unigrams + bigrams
        stop_words = None #this is optional as we already removed em
    )
    return vectorizer

def fit_transform_tfidf(vectorizer, texts):
    X = vectorizer.fit_transform(texts)
    return X

def transform_tfidf(vectorizer, texts):
    X = vectorizer.transform(texts)
    return X