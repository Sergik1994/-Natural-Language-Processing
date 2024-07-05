import string
import re

with open('stop_words.txt') as f:
    stop_words = f.read()
    stop_words_txt = stop_words_txt.split(',')

def data_preprocessing(text: str) -> str:
    """preprocessing string: lowercase, removing html-tags, punctuation, 
                            stopwords, digits

    Args:
        text (str): input string for preprocessing

    Returns:
        str: preprocessed string
    """    

    text = text.lower()
    text = re.sub('<.*?>', '', text) # html tags
    text = ''.join([c for c in text if c not in string.punctuation])# Remove punctuation
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = [word for word in text.split() if not word.isdigit()]
    text = ' '.join(text)
    return text