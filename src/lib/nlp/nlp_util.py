'''
===========================================================================================
Natural Language Processing Package
===========================================================================================
Script Reviewed by COGNAS
===========================================================================================
'''
from bs4 import BeautifulSoup
import unidecode
import numpy as np
import pandas as pd
import logging

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

    def build_word_tokenizer(self, dataframe=None, txt_inputs=None):

        i = 0
        for txt_feat in txt_inputs:
            logging.info('Processing txt feature ' +  str(txt_feat))

            #concatenate columns
            dataframe["_temp"] = dataframe[txt_feat].apply(lambda x: ' '.join(x.astype(str)), axis=1)

            # tokenizate to list
            dataframe[TXT_TOKENIZATION_COLUMN + '_' + str(i)] = dataframe["_temp"].apply(self.build_word_tokenizer_pandas_apply)

            # excluding temp column
            dataframe.drop("_temp", axis=1, inplace=True)
            i = i + 1
            
        return dataframe

    def build_word_tokenizer_pandas_apply(self, txt):
        tokens = txt.split()
        tokens = ' '.join(tokens)
        return tokens

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




