"""
Purpose: Utility functions for nlp.
Author(s): Bobby (Robert) Lumpkin
"""


from langdetect import detect
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import nltk.data
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
import numpy as np
import re
import string
import sys


def find_download_nltk(content, quiet=True):
    """
    Download nltk data only if don't already have it.
    """
    try:
        nltk.data.find(content)
    except LookupError:
        content_name = content.split("/")[-1]
        nltk.download(content_name, quiet=quiet)


find_download_nltk("tokenizers/punkt")
find_download_nltk("corpora/wordnet")
find_download_nltk("taggers/averaged_perceptron_tagger")
find_download_nltk("corpora/stopwords")
find_download_nltk("omw-1.4")


def detect_wrapper(text: str):
    """
    Catch langdetect.detect exceptions as `unknown`.
    """
    try:
        lang = detect(text)
    except Exception: 
        lang = "unknown"
    return lang


def remove_urls(text: str):
    """
    Remover urls from a string.

    Parameters
    ----------
    text: A string from which to remove urls.

    Returns
    ----------
    'text' with urls removed.
    """
    pattern = re.compile(r"https?://(\S+|www)\.\S+")
    matches = pattern.findall(text)
    text = pattern.sub(r"", text)
    return text


def remove_twitter_lingo(text: str):
    """
    Remove retweet, tag and hashtag symbols.

    Parameters
    ----------
    text: A string from which to remove 'twitter lingo'.

    Returns
    ----------
    'text' with twitter 'lingo removed'.
    """
    # Eliminate 'RT'
    pattern = re.compile(r"rt @\S+")
    matches = pattern.findall(text)
    text = pattern.sub(r"", text)

    # Eiliminate tags
    pattern = re.compile(r"@\S+")
    matches = pattern.findall(text)
    text = pattern.sub(r"", text)

    # Eliminate hashtags
    pattern = re.compile(r"#\S+")
    matches = pattern.findall(text)
    text = pattern.sub(r"", text)
    return text


def remove_emojis(text: str):
    """
    Remove emojis from a string.

    Parameters
    ----------
    text: A string from which to remove emojis.

    Returns
    ----------
    'text' with emojis removed.
    """
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    return text


def remove_nonprintable_chars(text: str):
    """
    Remove non-printable characters.

    Parameters
    ----------
    text: A string from which to remove non-printable characters.

    Returns
    ----------
    'text' with non-printable characters removed.
    """
    new_text = "".join([char for char in text if char in string.printable])

    return new_text


def get_stop_words(additional_stopwords: list = None):
    """
    Get a list of stopwords.

    Parameters
    ----------
    additional_stopwords: An optional list of stopwords to use in
        addition to `nltk.corupus.stopwords.words("english")`.

    Returns
    ----------
    A list of stopwords containing
        `nltk.corupus.stopwords.words("english")` and any additionally
        provided words.
    """
    words = stopwords.words("english")

    if additional_stopwords is None:
        return words
    else:
        words.extend(additional_stopwords)
        return words


def remove_words_from_list(text_list: list, words_list: list):
    """
    Remove words from a list.

    Parameters
    ----------
    text_list: A list of words from which to remove items in
        'words_list'.
    words_list: A list of words to remove from 'text_list'.

    Returns
    ----------
    'text_list' with words in 'words_list' removed.
    """
    return_list = [w for w in text_list if w not in words_list]
    return return_list


def remove_words_from_str(text: str, words_list: list):
    """
    Remove words from a string.

    Parameters
    ----------
    text: A string from which to remove words in 'words_list'.

    Returns
    ----------
    'text' with words in 'words_list' removed.
    """
    return_text = ' '.join([
        word for word in text.split() if word not in words_list
    ])
    return return_text


def remove_punctuation_from_list(text_list: list):
    """
    Remove punctuation from a list of characters.

    Parameters
    ----------
    text_list: A list of characters from which to remove punctuation.

    Returns
    ----------
    'text_list' with punctuation removed.
    """
    remove_punc = [w for w in text_list if w not in string.punctuation]
    return remove_punc


def remove_punctuation_from_str(text: str):
    """
    Remove punctuation from a string.

    Parameters
    ----------
    text: A string from which to remove punctuation.

    Returns
    ----------
    'text' with punctuation removed.
    """
    text = word_tokenize(text)
    text = ' '.join([word for word in text if word not in string.punctuation])
    return text


def remove_punctuation_from_list_regex(text_list: list):
    """
    Remove punctuation from a list of characters, using regex.

    Parameters
    ----------
    text_list: A list of characters from which to remove punctuation.

    Returns
    ----------
    'text_list' with punctuation removed.
    """
    for idx, ele in enumerate(text_list):
        new_ele = re.sub("[^\w*^\s*^\d*]", "", ele)
        new_ele = re.sub("\s+", "", new_ele)
        text_list[idx] = new_ele
    return_list = [w for w in text_list if w != ""]
    return return_list


def remove_punctuation_from_str_regex(text: str):
    """
    Remove punctuation from a string, using regex.

    Parameters
    ----------
    text: A string from which to remove punctuation.

    Returns
    ----------
    'text' with punctuation removed.
    """
    text = re.sub("[^\w*^\s*^\d*]", " ", text)
    text = re.sub("\s+", " ", text)
    text = ''.join(re.findall("[\w\s\d]*", text))
    return text


def get_wordnet_pos(word):
    """
    Get POS for 'nltk.stem.WordNetLemmatizer'.

    Parameters
    ----------
    word: A word (str) to get the POS of.

    Returns
    ----------
    The POS of 'word' for 'nltk.stem.WordNetLemmatizer'. One of 
        - wordnet.ADJ
        - wordnet.NOUN
        - wordnet.VERB
        - wordnet.ADV
    """
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def apply_word_lemmatizer(text_list, lemmatizer=None):
    """
    Apply a lemmatizer to a list of words.

    Parameters
    ----------
    text_list: A list of words to be lemmatized.
    lemmatizer: An instance of 'nltk.stem.WordNetLemmatizer' to use for
        lemmatization.

    Returns
    ----------
    'text' with elements lemmatized using 'lemmatizer'.
    """
    if not lemmatizer:
        lemmatizer = WordNetLemmatizer()
    lem_text = [
        lemmatizer.lemmatize(i, pos=get_wordnet_pos(i)) 
        for i in text_list
    ]
    return lem_text


def apply_word_stemmer(text_list, stemmer=None):
    """
    Apply a word-stemmer to a list of words.

    Parameters
    ----------
    text_list: A list of words to be stemmed.
    stemmer: An instance of 'nltk.stem.snowball.SnowballStemmer' to use
        for stemming.

    Returns
    ----------
    'text_list' with elements stemmed using 'stemmer'.
    """
    if not stemmer:
        stemmer = SnowballStemmer("english")
    stem_text = [stemmer.stem(i) for i in text_list]
    return stem_text


def join_text_list(text_list):
    """
    Join a list of words together into a string.
    
    Parameters
    ----------
    text_list: A list of strings to be joined.

    Returns
    ----------
    A single string -- the product of joining elements of 'text_list'
        together using a separator of ' '.
    """
    return " ".join(text_list)