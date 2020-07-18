'''
===========================================================================================
Model Evaluation Package
===========================================================================================
Script Reviewed by COGNAS
===========================================================================================
'''
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import logging
import pandas as pd
import numpy as np
from lib_xplore.data_handle.data_plotting import DataPlotting

class RegressionModelEvaluation:

    def __init__(self, Y_target=None, Y_predict=None, subset_label=None, output_type=None, history=None):
        '''Constructor for this class'''

        self.Y_target = Y_target
        self.Y_predict = Y_predict
        self.subset_label = subset_label
        self.output_type = output_type
        self.history = history

    def get_mean_absolute_error(self):
        return mean_absolute_error(y_true=self.Y_target, y_pred=self.Y_predict)

    def get_explained_variance_score(self):
        return explained_variance_score(y_true=self.Y_target, y_pred=self.Y_predict)

    def get_max_error_score(self):
        return max_error(y_true=self.Y_target, y_pred=self.Y_predict)

    def get_mse_error_score(self):
        return mean_squared_error(y_true=self.Y_target, y_pred=self.Y_predict)

    def get_mdae_error_score(self):
        return median_absolute_error(y_true=self.Y_target, y_pred=self.Y_predict)

    def get_r2_score(self):
        return r2_score(y_true=self.Y_target, y_pred=self.Y_predict)

    def print_evaluation_scores(self):
        logging.info("Mean Absolute Error: {a:.3f}".format(a=self.get_mean_absolute_error()))
        logging.info("Median Absolute Error: {a:.3f}".format(a=self.get_mdae_error_score()))
        logging.info("Mean Squared Error: {a:.3f}".format(a=self.get_mse_error_score()))
        logging.info("R2: {a:.3f}".format(a=self.get_r2_score()))
        logging.info("Explained Variance: {a:.3f}".format(a=self.get_explained_variance_score()))
        logging.info("Max Absolute Error: {a:.3f}".format(a=self.get_max_error_score()))

    def plot_evaluation_scores(self):

        result_train = pd.DataFrame(columns=['Output_Target', 'Output_Pred'])
        result_train['Output_Target'] = self.Y_target[:,0]
        result_train['Output_Pred'] = self.Y_predict[:,0]

        title = self.subset_label + " Dataset"
        dv = DataPlotting(dataframe=result_train)

        # Scatter plot with target x predicted
        dv.plot_scatter_2d(X_name='Output_Target', Y_name='Output_Pred', title=title,
                           marginal_distribution=True)

        # Sort regression
        dv.plot_chart_sort_regression(pred='Output_Pred', y='Output_Target')

class ClassificationModelEvaluation:

    def __init__(self, Y_target=None, Y_predict=None, subset_label=None, output_type=None, Y_labels=None, history=None):
        '''Constructor for this class'''
        self.Y_target = Y_target
        self.Y_predict = Y_predict
        self.subset_label = subset_label
        self.output_type = output_type
        self.Y_labels = Y_labels
        self.history = history

    def get_accuracy_score(self):
        return accuracy_score(y_true=self.Y_target, y_pred=self.Y_predict)

    def print_evaluation_scores(self):

        if self.output_type == "binary_category":
            cm = self.get_confusion_matrix()
            #print(cm)

        if self.output_type == "multi_category_unilabel":
            cm = self.get_confusion_matrix()
            #print(cm)

        logging.info("Accuracy: {a:.3f}".format(a=self.get_accuracy_score()))


    def get_confusion_matrix(self):
        cm = confusion_matrix(self.Y_target, self.Y_predict, normalize='all', labels=self.Y_labels)
        return cm

    def plot_evaluation_scores(self):

        result_train = pd.DataFrame(columns=['Output_Target', 'Output_Pred'])

        result_train['Output_Target'] = self.Y_target
        result_train['Output_Pred'] = self.Y_predict

        title = self.subset_label + " Dataset"
        dv = DataPlotting(dataframe=result_train)

        # history of training process
        if self.history is not None:
            dv.plot_history_training(self.history, loss=True, accuracy=True)

        # Scatter plot with target x predicted
        # dv.plot_scatter_2d(X_name='Output_Target', Y_name='Output_Pred', title=title,
        #                    marginal_distribution=True)

        if self.output_type == "binary_category":
            cm = self.get_confusion_matrix()
            dv.plot_confusion_matrix(cm=cm, names=self.Y_labels)

            dv.plot_roc(pred=result_train['Output_Pred'], y=result_train['Output_Target'])

        if self.output_type == "multi_category_unilabel":
            cm = self.get_confusion_matrix()
            dv.plot_confusion_matrix(cm=cm, names=self.Y_labels)

            #dv.plot_roc(pred=result_train['Output_Pred'], y=result_train['Output_Target'])