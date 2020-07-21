'''
===========================================================================================
Word2Vec Building Package
===========================================================================================
Script Reviewed by COGNAS
===========================================================================================
'''
from lib_xplore.nlp.nlp_processing import NLPProcessing
from gensim.models.word2vec import Word2Vec as W2V_gensim
from multiprocessing import cpu_count
import sys, logging, os
import numpy as np
import json

class Word2Vec:

    def __init__(self, model_parameters=None):
        '''Constructor for this class'''
        self.model_parameters = model_parameters
        self.corpus = None
        self.model = None
        self.word_index = None

    def set_corpus(self, dataframe=None, columns=None):
        nlp = NLPProcessing()

        # concatenate columns
        dataframe["_temp"] = dataframe[columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)

        # pandas list to list
        self.corpus = nlp.convert_pandas_tokens_to_list(dataframe=dataframe,
                                                        column="_temp")
        dataframe.drop("_temp", axis=1, inplace=True)

        return True

    def fit(self, ):

        # paralell computing
        qtd_threads = cpu_count()

        # Gensim implementation
        if self.model_parameters.framework == 'gensim':

            if self.model_parameters.topology == 'CBOW':
                sg = 0
            elif self.model_parameters.topology == 'skipgram':
                sg = 1
            else:
                logging.error('Ops ' + str(sys.exc_info()[0]) + ' occured!')
                raise

            # Init model
            self.model = W2V_gensim(size=self.model_parameters.embedding_dim,
                                    workers=qtd_threads,
                                    window=self.model_parameters.window_max_distance,
                                    min_count=self.model_parameters.tokens_min_count,
                                    sg=sg)

            # Build vocab
            self.model.build_vocab(sentences=self.corpus)

            # Training model
            self.model.train(sentences=self.corpus,
                             total_examples=self.model.corpus_count,
                             epochs=self.model_parameters.epochs)
            return self.model

    def save_model(self, working_folder=None, prefix=None):

        # Filename
        model_filename_path = working_folder
        model_filename = prefix + 'best_'

        filename = os.path.join(model_filename_path, model_filename + "model.h5")
        self.model.save(filename)

        # Word dictionary
        model_filename_word_dict = os.path.join(model_filename_path, model_filename + "wdic.json")

        # building dictionary
        word_index = {word: vocab.index for word, vocab in self.model.wv.vocab.items()}
        with open(model_filename_word_dict, 'w') as json_file:
            json.dump(word_index, json_file)

        logging.info('Model saved:' + filename)
        logging.info('Word dictionary saved: ' + model_filename_word_dict)

        return filename

    def load_model(self, filename):

        self.model = W2V_gensim.load(filename)

        return filename

    def get_embedding_parameters(self):

        return True

    def get_embedding_matrix(self):

        matrix_vector = self.model.wv[self.model.wv.vocab]
        word_index = {word: vocab.index for word, vocab in self.model.wv.vocab.items()}

        WV_WORDS = matrix_vector.shape[0]
        WV_DIM = matrix_vector.shape[1]

        # initialize the matrix with random numbers
        wv_matrix = np.zeros((WV_WORDS, WV_DIM))
        for word, i in word_index.items():
            try:
                embedding_vector = self.get_vector_from_word(word=word)
                if embedding_vector is not None:
                    # words not found in embedding index will be all-zeros.
                    wv_matrix[i] = embedding_vector
            except:
                pass
        return wv_matrix

    def get_vector_from_word(self, word):
        return self.model[word]