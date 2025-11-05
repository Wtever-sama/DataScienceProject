'''process source data from text files into vectors, which can be used for training and testing'''
from stop_symbol import stopwords_

# try downloading NLTK stopwords if available
try:
    from nltk.corpus import stopwords
    stopwords_nltk = set(stopwords.words('english'))

    # merge stopwords
    stopwords = stopwords_nltk.union(stopwords_)
except LookupError:
    # if failed, use stopwords_ only
    stopwords = set(stopwords_)
    import nltk
    # try downloading data, but don't require it to succeed
    try:
        nltk.download('punkt', quiet=True)
    except:
        pass

def _strip_punctuation(text: str)-> str:
    """
    Strip punctuation from a string
    """
    # replace `word_tokenize with` simple split and filter
    words = text.lower().split()
    filtered_words = [w for w in words if w not in stopwords and w.isalnum()]
    return " ".join(filtered_words)

from sklearn import datasets
from sklearn.model_selection import train_test_split, StratifiedKFold
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

import os
from tqdm import tqdm


def _load_src_data(baseball_dir: str, hockey_dir: str)-> tuple[list[str], list[str]]:
    '''Load source data from baseball and hockey dirs'''
    baseball_data = []
    hockey_data = []
    for f in tqdm(os.listdir(baseball_dir), desc="Loading baseball data"):
        file_path = os.path.join(baseball_dir, f)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        except UnicodeDecodeError:
            # try other encoding if UTF-8 fails
            with open(file_path, 'r', encoding='latin-1') as file:
                text = file.read()
        baseball_data.append(_strip_punctuation(text))
        
    for f in tqdm(os.listdir(hockey_dir), desc="Loading hockey data"):
        file_path = os.path.join(hockey_dir, f)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        except UnicodeDecodeError:
            # try other encoding if UTF-8 fails
            with open(file_path, 'r', encoding='latin-1') as file:
                text = file.read()
        hockey_data.append(_strip_punctuation(text))
    return baseball_data, hockey_data


from sklearn.feature_extraction.text import TfidfVectorizer


def _vectorize_data(X: list[str])-> tuple:
    '''Vectorize data using CountVectorizer'''
    tf = TfidfVectorizer()
    X_tf = tf.fit_transform(X)
    return X_tf

def _train_test_split(X, y)-> tuple:
    '''Split data into train and test sets'''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # stratify to keep class balance
    return X_train, X_test, y_train, y_test

def data_full()-> tuple:
    '''Load full data from both classes'''
    baseball_data, hockey_data = _load_src_data('Dataset_classification/baseball', 'Dataset_classification/hockey')
    X_src = baseball_data + hockey_data
    X_full = _vectorize_data(X_src)
    y_full = ['baseball'] * len(baseball_data) + ['hockey'] * len(hockey_data)
    X_train, X_test, y_train, y_test = _train_test_split(X_full, y_full)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = data_full()
    print(X_train.shape, X_test.shape, len(y_train), len(y_test))