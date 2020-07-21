"""
===========================================================================================
Polynomial Regression Model Building Class
===========================================================================================
Script Reviewed by COGNAS
===========================================================================================
"""

import pandas as pd
import logging
from src.lib.model.xmodel import XModel
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


class XPolynomialRegression(XModel):
    def fit(self, data_input, data_target):

        # init model
        poly_features = PolynomialFeatures(degree=self.param.degree)
        X_poly = poly_features.fit_transform(data_input)
        self.model = LinearRegression()

        # fit model
        self.model.fit(X_poly, data_target)

        # report results
        self.report_modeling()

        return self.model

    def eval_predict(self, data_input: pd) -> pd:

        poly_features = PolynomialFeatures(degree=self.param.degree)
        X_poly = poly_features.fit_transform(data_input)
        data_predict = self.model.predict(X_poly)

        # convert ndarray to pandas dataframe
        data = pd.DataFrame(data=data_predict)

        return data

    def report_modeling(self) -> bool:

        logging.info("======================================================================")
        logging.info("Hyperparameters:")
        logging.info('degree : {a:.3f}'.format(a=self.param.degree))

        return True
