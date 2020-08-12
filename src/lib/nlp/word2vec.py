"""
===========================================================================================
Word2Vec Model Building Class
===========================================================================================
Script Reviewed by COGNAS
===========================================================================================
"""

from gensim.models.word2vec import Word2Vec
import logging, os
import numpy as np
import pandas as pd
import json
from src.lib.model.xmodel import XModel


class XWord2Vec(XModel):
    def init(self) -> bool:

        # init parameters
        if self._param.topology == "cbow":
            self.sg = 0
        else:
            self.sg = 1

        self._model = Word2Vec(
            size=self._param.encode_dim,
            workers=os.cpu_count(),
            window=self._param.window_max_distance,
            min_count=0,
            sg=self.sg,
            compute_loss=True,
            sample=0.0,
        )

        return True

    def fit(self, dataframe: pd = None, corpus_col: str = None):

        # init model
        self.init()

        # convert pandas column to list of sentences
        corpus = list(dataframe[corpus_col])

        # build vocabulary dictionary
        self._model.build_vocab(sentences=corpus)

        # todo callbacks
        # monitor = MonitorCallback(save_model=save_checkpoints, path_prefix=output_checkpoints_folder,
        #                          output_epochs_per_checkpoint=output_epochs_per_checkpoint, prefix=self.model_name)

        # training model
        self._model.train(
            sentences=corpus, total_examples=self._model.corpus_count, epochs=self._param.epochs
        )
        # save results
        self.save_results()

        return True

    def save_model(self):

        # model h5
        filename_path = self._run_folder_path + self._prefix_name + "w2v_model.h5"
        self.model.save(filename_path)

        # Word dictionary
        filename_word_dict_path = self._run_folder_path + self._prefix_name + "w2v_wdic.json"

        # word dictionary
        word_index = {word: vocab.index for word, vocab in self._model.wv.vocab.items()}
        with open(filename_word_dict_path, "w") as json_file:
            json.dump(word_index, json_file)

        logging.info("Model saved:" + filename_path)
        logging.info("Word dictionary saved: " + filename_word_dict_path)

        return True

    def load_model(self, filename):

        return Word2Vec.load(filename)

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

    def eval_predict(self):

        # model evaluation

        return True

    def save_results(self) -> bool:

        error_training = self._model.get_latest_training_loss()
        logging.info('Final Training Loss: ' + str(error_training))

        # hyperparameters (numbers and string)
        self._history["params"] = dict(self._param)

        # metrics (list numbers only)
        # self._history['metrics'] = {'teste': [100]}

        return True
