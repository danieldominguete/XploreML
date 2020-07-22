'''
===========================================================================================
K-nearest Neighbors Model Building Class
===========================================================================================
Script Reviewed by COGNAS
===========================================================================================
'''
import heapq
import pandas as pd
import numpy as np
import logging
from src.lib.model.xmodel import XModel
from sklearn.neighbors import KNeighborsClassifier

class XKNearestNeighbors(XModel):

    def fit(self, data_input, data_target):

        # init model
        model = KNeighborsClassifier(n_neighbors=self.param.n_neighbors, metric=self.param.metric, p=self.param.metric_power)
        
        # fit model
        self.model = model.fit(data_input, data_target)

        # report results
        self.report_modeling()

        return self.model

    def eval_regression_predict(self, data_input:pd)-> pd:
        logging.error("K nearest neighbors regression is not available")

    def eval_classification_predict(self, data_input:pd, int_to_cat_dict_target)->pd:

        data_predict = self.model.predict(data_input)

        # convert coding output to label category (1 max)
        id_max = np.argmax(data_predict, axis=1)
        value_max = np.max(data_predict, axis=1)

        prediction = []
        for sample in id_max:
            prediction.append(int_to_cat_dict_target[sample])

        # convert ndarray to pandas dataframe
        data = pd.DataFrame(data=prediction)

        return data
    
    def report_modeling(self) -> bool:

        logging.info("======================================================================")
        logging.info('Hyperparameters:')
        logging.info('number of neighbors : ' + str(self.param.n_neighbors))
        logging.info('distance metric : ' + str(self.param.metric))
        logging.info('metric power : ' + str(self.param.metric_power))
        #logging.info('fit_intersection : {a:.3f}'.format(a=self.param.fit_intersection))

        return True

