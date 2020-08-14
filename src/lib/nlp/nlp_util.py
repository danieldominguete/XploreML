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
    def build_word_tokenizer(dataframe:pd=None, column:str=None, return_list:bool=True) -> pd:

        # tokenizate to list
        name_col = column + "_tk"
        dataframe[name_col] = dataframe[column].apply(NLPUtils.build_word_tokenizer_pandas_apply,args=(return_list,))
            
        return dataframe, name_col

    @staticmethod
    def build_word_tokenizer_pandas_apply(txt:str, return_list:bool)->list:

        # tokenizer with whitespace
        txt = str(txt)
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

        if return_list:
            # tokenizer refactor to list
            txt = txt.split()

        return txt

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

    @staticmethod
    def encode_word2int(dataframe=None, columns=None, max_length=0):

        from tensorflow.keras.preprocessing.sequence import pad_sequences
        input_var = []
        int2word_dict_list = [] 
        word2int_dict_list = []
        
        for var in columns:
            
            # Count distinct words
            dataframe[var] = dataframe[var].apply(lambda x: str(x).lower())
            words = set()
            dataframe[var].str.lower().str.split().apply(words.update)
            unique_words_count = len(words)
            logging.info("Word2int encode var: " + var + " => unique words: " + str(unique_words_count))
    
            int2word_dict = dict((i, w) for i, w in enumerate(words))
            word2int_dict = dict((w, i) for i, w in enumerate(words))
    
    
            # Hash each word in row
            dataframe[var + '_int_encoded'] = dataframe[var].apply(NLPUtils.word2int_from_dict, args=(word2int_dict,))
            encoded_samples = pad_sequences(dataframe[var + '_int_encoded'], maxlen=max_length, padding='post')
    
            # Converting to pandas
            var_list = []
            for i in range(max_length):
                var_name = var + "_" + str(i)
                dataframe[var_name] = encoded_samples[:,i]
                var_list.append(var_name)
            
            input_var.append(var_list)
            int2word_dict_list.append(int2word_dict)
            word2int_dict_list.append(word2int_dict)



        return dataframe, input_var, int2word_dict_list, word2int_dict_list

    @staticmethod
    def word2int_from_dict(txt,word2int_dict):

        txt_list = txt.split()
        encoded = [word2int_dict[word] for word in txt_list]

        return encoded

    def convert_pandas_tokens_to_list(self, dataframe=None, column=None):
        corpus = []
        for index, row in dataframe.iterrows():
            line_list = row[column].split()
            corpus.append(line_list)
        return corpus




