'''
===========================================================================================
Natural Language Processing Package
===========================================================================================
Script Reviewed by COGNAS
===========================================================================================
'''
from bs4 import BeautifulSoup
import unidecode
import re
import numpy as np
import pandas as pd
import nltk
import logging
import json
from src.lib.utils.util import Util

TXT_TOKENIZATION_COLUMN = 'TXT_TOKENS'

class NLPUtils:

    def __init__(self):
        '''Constructor for this class'''
        self.word2int_dict = None

    @staticmethod
    def clean_html(dataframe:pd =None, columns:list =None) -> pd:
        for col in columns:
            dataframe[col] = dataframe[col].apply(NLPUtils.clean_html_pandas_apply)
        return dataframe

    @staticmethod
    def clean_html_pandas_apply(txt:str) -> str:
        txt = BeautifulSoup(txt,'lxml')
        return txt.text

    @staticmethod
    def convert_to_unicode(dataframe: pd =None, columns: list =None) -> pd:
        for col in columns:
            dataframe[col] = dataframe[col].apply(NLPUtils.convert_to_unicode_pandas_apply)
        return dataframe

    @staticmethod
    def convert_to_unicode_pandas_apply(txt) -> str:
        txt = unidecode.unidecode(txt)
        return txt

    @staticmethod
    def convert_to_lower(dataframe:pd=None, columns:list=None) -> pd:
        for col in columns:
            dataframe[col] = dataframe[col].apply(NLPUtils.convert_to_lower_pandas_apply)
        return dataframe

    @staticmethod
    def convert_to_lower_pandas_apply(txt:str) -> str:
        txt = txt.lower()
        return txt

    @staticmethod
    def split_units_from_numbers(dataframe: pd = None, columns: list = None) -> pd:
        for col in columns:
            dataframe[col] = dataframe[col].apply(NLPUtils.split_units_from_numbers_pandas_apply)
        return dataframe

    @staticmethod
    def split_units_from_numbers_pandas_apply(txt: str) -> str:
        # https://en.wikipedia.org/wiki/Metric_units
        pattern = re.compile(
            '(\d+)(litros|ml|l|hz|mhz|ghz|hp|ph|kb|mb|gb|tb|kbps|mbps|bps|bar|mmhg|pa|mwh|kwh|kw|w|va|kva|k|kg|g|gr|pol|km|m|cm|mm|km2|m2|cm2|mm2|km3|m3|cm3|mm3|in|ft|rad|rads|db|v|volts|dpi|h|seg)(?![A-zÀ-ÿ0-9])',
            re.S)
        txt = re.sub(pattern, r'\1 \2 ', txt)
        return txt

    @staticmethod
    def clean_special_char(dataframe:pd =None, columns:list =None) -> pd:
        for col in columns:
            dataframe[col] = dataframe[col].apply(NLPUtils.clean_special_char_pandas_apply)
        return dataframe

    @staticmethod
    def clean_special_char_pandas_apply(txt:str) -> str:
        char_from = '!"#$&()*/:;<=>?@[\\]^`{|}~\''
        char_to = ' ' * len(char_from)

        # Tabela de conversao
        table = str.maketrans(char_from, char_to)
        txt = txt.translate(table)

        return txt

    @staticmethod
    def build_sentence_tokenizer(dataframe:pd = None, column:str = None) -> pd:

        name_col = column + "_sent"
        dataframe[name_col] = dataframe[column].apply(NLPUtils.build_sentence_tokenizer_pandas_apply)

        return dataframe,name_col

    @staticmethod
    def build_sentence_tokenizer_pandas_apply(txt:str) -> list:
        sentences = nltk.tokenize.sent_tokenize(txt)
        return sentences

    @staticmethod
    def build_word_tokenizer(dataframe:pd=None, column:str=None) -> pd:

        # tokenizate to list
        name_col = column + "_tk"
        dataframe[name_col] = dataframe[column].apply(NLPUtils.build_word_tokenizer_pandas_apply)
            
        return dataframe, name_col

    @staticmethod
    def build_word_tokenizer_pandas_apply(txt:str)->list:

        # tokenizer with whitespace
        tokens = txt.split()

        # Excluding special chars at the ending
        tokens = [word[:-2] if word.endswith('.,') else word for word in tokens]
        tokens = [word[:-1] if word.endswith('.') else word for word in tokens]
        tokens = [word[:-1] if word.endswith(',') else word for word in tokens]
        #tokens = [word[:-1] if word.endswith('-') else word for word in tokens]

        # change , for . for numbers
        tokens = [re.sub('(\d+),(\d+)', r'\1.\2', word) for word in tokens]

        # number to digits
        tokens = [re.sub('(\d)', r' \1 ', word) for word in tokens]

        # ??
        tokens = [re.sub('^(\d+)$', r' \1 ', word) for word in tokens]

        # joining post processing
        txt = " ".join(tokens)

        # tokenizer refactor
        tokens = txt.split()

        return tokens

    @staticmethod
    def build_freqdist_tokens(dataframe:pd=None, column:str=None):
        tokens_all = Util.get_list_from_pandas_list_rows(dataframe=dataframe, column=column)
        freqdist = nltk.FreqDist(tokens_all)
        #freqdist = freqdist.most_common()
        return freqdist


    @staticmethod
    def convert_json_to_txt(dataframe:pd=None, column:str=None) -> pd:
        dataframe[column] = dataframe[column].apply(NLPUtils.convert_json_to_txt_pandas_apply)
        return dataframe

    @staticmethod
    def convert_json_to_txt_pandas_apply(txt: str = None):

        txt_data = ""
        empty_values = ['-', 'nan']

        try:
            data = json.loads(txt)
            for key in data:
                value = data.get(key)

                # normalize empty values
                if value in empty_values:
                    value = ""

                # maximum char
                if len(str(value))<100:
                    txt_data = txt_data + " " + str(key) + " " + str(value)
        except:
            logging.error("Invalid json")

        return txt_data


    def encode_word2int(self, dataframe=None, column=None, max_length=0):

        # Count distinct words
        dataframe[column].str.lower().str.split()
        words = set()
        dataframe[column].str.lower().str.split().swifter.apply(words.update)
        unique_words_count = len(words)

        self.int2word_dict = dict((i, w) for i, w in enumerate(words))
        self.word2int_dict = dict((w, i) for i, w in enumerate(words))

        # Create a dictionary
        #words = list(words)
        #index = range(0,unique_words_count)

        #zipbObj = zip(words, index)
        #self.word2int_dict = dict(zipbObj)

        #zipbObj = zip(index, words)
        #self.int2word_dict = dict(zipbObj)

        # Hash each word in row
        dataframe['txt_encoded'] = dataframe[column].swifter.apply(self.word2int_from_dict)
        encoded_samples = pad_sequences(dataframe['txt_encoded'], maxlen=max_length, padding='post')

        # Reshape for RNN [samples, time steps, features]
        encoded_samples = np.reshape(a=encoded_samples,newshape=(dataframe.shape[0],max_length,1))

        return encoded_samples

    def word2int_from_dict(self, txt):

        txt_list = txt.split()
        encoded = [self.word2int_dict[word] for word in txt_list]

        return encoded

    def convert_pandas_tokens_to_list(self, dataframe=None, column=None):
        corpus = []
        for index, row in dataframe.iterrows():
            line_list = row[column].split()
            corpus.append(line_list)
        return corpus




