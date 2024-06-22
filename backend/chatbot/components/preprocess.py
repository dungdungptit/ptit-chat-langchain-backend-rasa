from pyvi import ViTokenizer
from pyvi.ViTokenizer import tokenize
import pandas as pd
import re
import string
# Create your views here.


def clean_text(text):
    text = re.sub('<.*?>', '', text).strip()
    text = re.sub('(\s)+', r'\1', text)
    return text

def sentence_segment(text):
    sents = re.split("([.?!])?[\n]+|[.?!] ", text)
    return sents

def word_segment(sent): # chuyển câu thành từ
    sent = tokenize(sent)
    return sent

def normalize_text(text):
    listpunctuation = string.punctuation.replace('_', '')
    for i in listpunctuation:
        text = text.replace(i, ' ')
    return text.lower()


def remove_numbers(text_in):
  for ele in text_in.split(): 
    if ele.isdigit():
        text_in = text_in.replace(ele, "@")
  for character in text_in:
    if character.isdigit():
        text_in = text_in.replace(character, "@")
  return text_in


def remove_special_characters(text):
  chars = re.escape(string.punctuation)
  return re.sub(r'['+chars+']', '', text)

 
def preprocess(text_in):  
    text = clean_text(text_in)
    text = remove_special_characters(text)
    text = remove_numbers(text) 
    return text

# data_stopwords = pd.read_csv('../FastText_v1/stopwords.csv')
# list_stopwords = data_stopwords['stopwords'].values.tolist()
# def remove_stopword(text):
#     text = ' '.join([i for i in text.split() if i not in list_stopwords])
#     return text

def process_text(text):
    text = clean_text(text)
    text = remove_special_characters(text)
    text = remove_numbers(text)
    text = word_segment(text)
    text = normalize_text(text)
    # text = remove_stopword(text)
    return text