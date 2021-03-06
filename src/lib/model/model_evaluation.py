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
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
import logging
import numpy as np
import pandas as pd
from src.lib.data_visualization.data_plotting import DataPlotting

class RegressionModelEvaluation:

    def __init__(self, Y_target:pd=None, Y_predict:pd=None, Y_reliability:pd=None, subset_label:str=None, regression_type:str=None, train_history=None):
        '''Constructor for this class'''

        self.Y_target = Y_target
        self.Y_predict = Y_predict
        self.Y_reliability = Y_reliability
        self.subset_label = subset_label
        self.regression_type = regression_type
        self.train_history = train_history
        self.history = {"params": {}, "metrics": {}, "files": {}}

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

    def plot_training_results(self, view:bool=False, save:bool=False, path:str=None, prefix:str=None)->bool:

        result_train = pd.DataFrame(columns=['Output_Target', 'Output_Pred'])
        result_train['Output_Target'] = self.Y_target.iloc[:,0]
        result_train['Output_Pred'] = self.Y_predict.iloc[:,0]

        dv = DataPlotting(dataframe=result_train, view_plots=view, save_plots=save, folder_path=path, prefix=prefix)

        # history of training process
        if self.history is not None:
            dv.plot_history_training(self.train_history)

        # Scatter plot with target x predicted
        dv.plot_scatter_2d(X_name='Output_Target', Y_name='Output_Pred', title="Training",
                           marginal_distribution=True)

        # Sort regression
        dv.plot_chart_sort_regression(pred='Output_Pred', y='Output_Target')

        return True

    def plot_test_results(self, view:bool=False, save:bool=False, path:str=None, prefix:str=None)->bool:

        result_train = pd.DataFrame(columns=['Output_Target', 'Output_Pred'])
        result_train['Output_Target'] = self.Y_target.iloc[:,0]
        result_train['Output_Pred'] = self.Y_predict.iloc[:,0]

        dv = DataPlotting(dataframe=result_train, view_plots=view, save_plots=save, folder_path=path, prefix=prefix)

        # Scatter plot with target x predicted
        dv.plot_scatter_2d(X_name='Output_Target', Y_name='Output_Pred', title="Test",
                           marginal_distribution=True)

        # Sort regression
        dv.plot_chart_sort_regression(pred='Output_Pred', y='Output_Target')

        return True

    def include_param_history(self, dict):
        self.history["params"].update(dict)
        return True

    def include_metric_history(self, dict):
        self.history["metrics"].update(dict)
        return True

    def include_files_history(self, dict):
        self.history["files"].update(dict)
        return True

    def get_prediction_report(self):

        df_report = pd.DataFrame()
        df_report["target"] = self.Y_target.iloc[:,0]
        df_report["predict"] = self.Y_predict.iloc[:,0]
        df_report["reliability"] = self.Y_reliability.iloc[:,0]
        df_report["error"] = df_report.apply(lambda x: 0 if x["target"] == x["predict"] else 1, axis=1)
        return df_report

    def execute(self):

        # MSE
        name = self.subset_label + "mse"
        value = self.get_mse_error_score()
        logging.info(name + ": {a:.3f}".format(a=value))
        self.include_metric_history({name:[value]})

        # MAE
        name = self.subset_label + "mae"
        value = self.get_mean_absolute_error()
        logging.info(name + ": {a:.3f}".format(a=value))
        self.include_metric_history({name: [value]})

        # R2
        name = self.subset_label + "R2"
        value = self.get_r2_score()
        logging.info(name + ": {a:.3f}".format(a=value))
        self.include_metric_history({name: [value]})

        return True


class ClassificationModelEvaluation:

    def __init__(self, Y_target:pd=None, Y_predict:pd=None, Y_reliability:pd=None, subset_label:str=None, classification_type:str=None, Y_int_to_cat_labels:dict=None, Y_cat_to_int_labels:dict=None, train_history=None):
        '''Constructor for this class'''
        self.Y_target = Y_target
        self.Y_predict = Y_predict
        self.Y_reliability = Y_reliability
        self.subset_label = subset_label
        self.classification_type = classification_type
        self.Y_labels = list(Y_int_to_cat_labels[0].values())
        self.Y_int_to_cat_labels = Y_int_to_cat_labels[0]
        self.Y_cat_to_int_labels = Y_cat_to_int_labels[0]
        self.train_history = train_history
        self.history = {"params": {}, "metrics": {}, "files": {}}

    def get_accuracy_score(self):
        return accuracy_score(y_true=self.Y_target, y_pred=self.Y_predict)

    def get_balanced_accuracy_score(self):
        return balanced_accuracy_score(y_true=self.Y_target, y_pred=self.Y_predict)

    def get_confusion_matrix(self):
        y_true = self.Y_target.to_numpy()
        y_pred = self.Y_predict.to_numpy()
        labels = self.Y_labels
        cm = confusion_matrix(y_true=y_true,y_pred=y_pred , normalize='all', labels=labels)
        return cm

    def get_confusion_classes(self):

        # id classes
        dict_int_to_string = dict((i, c) for i, c in enumerate(self.Y_labels))

        # confusion matrix
        confusion_matrix = self.get_confusion_matrix()

        # dataframe with classes and their biggest confusion class
        cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:,np.newaxis]
        cm[np.isnan(cm)] = 0

        # excluding the right classification
        np.fill_diagonal(a=cm, val=0)

        # get biggest confusion values
        cm_max_class = np.argmax(a=cm, axis=1)
        cm_max_value = np.max(a=cm, axis=1)

        target_int = list(range(0, cm.shape[0], 1))
        target_class = [dict_int_to_string[i] for i in target_int]
        confusion_class = [dict_int_to_string[i] for i in cm_max_class]

        df_report = pd.DataFrame()
        df_report['target'] = target_class
        df_report['confusion'] = confusion_class
        df_report['value'] = cm_max_value

        # Zero values represents not exist target in prediction values or classes with no confusion
        df_report = df_report[df_report['value'] != 0]

        return df_report


    def plot_training_results(self, view:bool=False, save:bool=False, path:str=None, prefix:str=None)->bool:

        result_train = pd.DataFrame(columns=['Output_Target', 'Output_Pred'])

        result_train['Output_Target'] = self.Y_target.iloc[:,0]
        result_train['Output_Pred'] = self.Y_predict.iloc[:,0]

        dv = DataPlotting(dataframe=result_train, view_plots=view, save_plots=save, folder_path=path, prefix=prefix)

        # history of training process
        if self.history is not None:
            dv.plot_history_training(self.train_history)

        if self.classification_type == "binary_category":
            cm = self.get_confusion_matrix()
            dv.plot_confusion_matrix(cm=cm, names=self.Y_labels)

            # converting binary labels to 0,1
            result_train['Output_Target'].replace(self.Y_cat_to_int_labels,inplace=True)
            result_train['Output_Pred'].replace(self.Y_cat_to_int_labels, inplace=True)
            dv.plot_roc(pred=result_train['Output_Pred'], y=result_train['Output_Target'])

        if self.classification_type == "multi_category_unilabel":
            cm = self.get_confusion_matrix()
            dv.plot_confusion_matrix(cm=cm, names=self.Y_labels)

        return True

    def plot_test_results(self, view:bool=False, save:bool=False, path:str=None, prefix:str=None)->bool:

        result_train = pd.DataFrame(columns=['Output_Target', 'Output_Pred'])

        result_train['Output_Target'] = self.Y_target.iloc[:,0]
        result_train['Output_Pred'] = self.Y_predict.iloc[:,0]

        dv = DataPlotting(dataframe=result_train, view_plots=view, save_plots=save, folder_path=path, prefix=prefix)

        if self.classification_type == "binary_category":
            cm = self.get_confusion_matrix()
            dv.plot_confusion_matrix(cm=cm, names=self.Y_labels)

            # converting binary labels to 0,1
            result_train['Output_Target'].replace(self.Y_cat_to_int_labels,inplace=True)
            result_train['Output_Pred'].replace(self.Y_cat_to_int_labels, inplace=True)
            dv.plot_roc(pred=result_train['Output_Pred'], y=result_train['Output_Target'])

        if self.classification_type == "multi_category_unilabel":
            cm = self.get_confusion_matrix()
            dv.plot_confusion_matrix(cm=cm, names=self.Y_labels)

        return True

    def plot_reliability_sensitivity(self, report:pd, view:bool=False, save:bool=False, path:str=None, prefix:str=None, title:str=None)->bool:

        dv = DataPlotting(dataframe=report, view_plots=view, save_plots=save, folder_path=path, prefix=prefix)
        dv.plot_multiple_lines(x_column='reliability', y_columns=['coverage', 'accuracy'], title=title, xlabel='reliability', ylabel='value')

        return True

    def include_param_history(self, dict):
        self.history["params"].update(dict)
        return True

    def include_metric_history(self, dict):
        self.history["metrics"].update(dict)
        return True

    def include_files_history(self, dict):
        self.history["files"].update(dict)
        return True

    def get_prediction_report(self):

        df_report = pd.DataFrame()
        df_report["target"] = self.Y_target.iloc[:,0]
        df_report["predict"] = self.Y_predict.iloc[:,0]
        df_report["reliability"] = self.Y_reliability.iloc[:,0]
        df_report["error"] = df_report.apply(lambda x: 0 if x["target"] == x["predict"] else 1, axis=1)
        return df_report

    def get_reliability_sensitivity(self):

        triggers_list = np.arange(0.025,1.0,0.025)
        accuracy_list = []
        coverage_list = []

        prediction_report = self.get_prediction_report()

        for value in triggers_list:
            prediction_report['filtered'] = np.where((prediction_report['reliability'] < value), -1, prediction_report['error'])
            correct_predictions = prediction_report[prediction_report['filtered'] == 0].shape[0]
            error_predictions = prediction_report[prediction_report['filtered'] == 1].shape[0]
            nan_predictions = prediction_report[prediction_report['filtered'] == -1].shape[0]

            coverage = (correct_predictions + error_predictions)/(correct_predictions+error_predictions+nan_predictions)
            if coverage > 0:
                accuracy = (correct_predictions)/(correct_predictions+error_predictions)
            else:
                accuracy = 1

            #logging.info("Reliability: {a:.3f}".format(a=value) + " = Coverage: {a:.3f}".format(a=coverage) + " Accuracy: {a:.3f}".format(a=accuracy))
            accuracy_list.append((accuracy))
            coverage_list.append(coverage)

        df_report = pd.DataFrame()
        df_report['reliability'] = triggers_list
        df_report['coverage'] = coverage_list
        df_report['accuracy'] = accuracy_list

        return df_report

    def get_score_by_classes(self):

        classes_report = classification_report(y_true=self.Y_target,
                                               y_pred=self.Y_predict,
                                               target_names=self.Y_labels,
                                               output_dict=True)
        df_classes_report = pd.DataFrame(classes_report).transpose()

        return df_classes_report

    def execute(self):

        # Accuracy
        name = self.subset_label + "accuracy"
        value = self.get_accuracy_score()
        logging.info(name + ": {a:.3f}".format(a=value))
        self.include_metric_history({name:[value]})

        # Balanced Accuracy
        name = self.subset_label + "bal_accuracy"
        value = self.get_balanced_accuracy_score()
        logging.info(name + ": {a:.3f}".format(a=value))
        self.include_metric_history({name:[value]})

        return True


