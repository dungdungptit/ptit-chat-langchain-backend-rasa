from __future__ import annotations
from typing import Any, Dict, List, Optional, Text

from rasa.engine.graph import ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.shared.nlu.training_data.message import Message
from underthesea import word_tokenize

from pyvi import ViTokenizer
from pyvi.ViTokenizer import tokenize
import pandas as pd
import re
import string

@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_TOKENIZER, is_trainable=False
)
class VietnameseTokenizer(Tokenizer):
    """Creates features for entity extraction."""

    @staticmethod
    def not_supported_languages() -> Optional[List[Text]]:
        """The languages that are not supported."""
        return ["zh", "ja", "th"]

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Returns the component's default config."""
        return {
            # This *must* be added due to the parent class.
            "intent_tokenization_flag": False,
            # This *must* be added due to the parent class.
            "intent_split_symbol": "_",
            # This is a, somewhat silly, config that we pass
        }

    def __init__(self, config: Dict[Text, Any]) -> None:
        """Initialize the tokenizer."""
        super().__init__(config)

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> VietnameseTokenizer:
        return cls(config)

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        text = message.get(attribute)
        text = process_text(text)
        words = text.split()

        words = [w for w in words if w]

        # if we removed everything like smiles `:)`, use the whole text as 1 token
        if not words:
            words = [text]
        # the ._convert_words_to_tokens() method is from the parent class.
        tokens = self._convert_words_to_tokens(words, text)

        return self._apply_token_pattern(tokens)
    
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