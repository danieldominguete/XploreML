'''
===========================================================================================
MLFlow Package
===========================================================================================
Webview: $ mlflow ui
===========================================================================================
Script by COGNAS
===========================================================================================
'''
import logging
import mlflow
from mlflow.tracking import MlflowClient, get_tracking_uri, set_tracking_uri, is_tracking_uri_set


class MLFlowManagement:

    def __init__(self, param=None):
        """Constructor for this class"""
        self.param = param
        self.client = None
        self.run_id = None

    def setup_mlflow_tracking(self, URI, experiment_name, run_name):

        # select URI for server tracking
        set_tracking_uri(uri=URI)
        if is_tracking_uri_set():
            logging.debug('MLFlow URI: ' + str(get_tracking_uri()))

        # CRUD interface
        self.client = MlflowClient(tracking_uri=get_tracking_uri())

        # Experiment setup
        if self.client.get_experiment_by_name(name=experiment_name) is None:
            exp_id = self.client.create_experiment(name=experiment_name)
        else:
            exp = self.client.get_experiment_by_name(name=experiment_name)
            exp_id = exp.experiment_id

        # Run setup
        mlflow.start_run(experiment_id=exp_id, run_name=run_name)
        self.run_id = mlflow.active_run().info.run_id
        data = self.client.get_run(mlflow.active_run().info.run_id).data
        logging.info('MLFlow tracking started - Experiment: ' + str(experiment_name) + " - Run: " + str(
            data.tags["mlflow.runName"]))

    def log_param(self, key, value):
        self.client.log_param(key=key, value=value, run_id=self.run_id)

    def log_params(self, params):
        for key in params.keys():
            self.client.log_param(key=key, value=params[key], run_id=self.run_id)

    def log_metric(self, key, value):
        self.client.log_metric(key=key, value=value, run_id=self.run_id)

    def log_metrics(self, metrics):
        for key in metrics.keys():
            step = 0
            for value in metrics[key]:
                self.client.log_metric(key=key, value=value, step=step, run_id=self.run_id)
                step += 1

    def log_artifact(self, file):
        self.client.log_artifact(local_path=file, run_id=self.run_id)

    def log_artifacts(self, files_dict):
        for key in files_dict.keys():
            self.client.log_artifact(local_path=files_dict[key], run_id=self.run_id)

    def log_artifacts_folder(self, local_dir):
        self.client.log_artifact(local_dir=local_dir, run_id=self.run_id)

    # def publish_classification_eval(self, model_eval=None, mode=None):
    #
    #     if mode == 'train':
    #         log_param("train_acc", model_eval.get_accuracy_score())
    #
    #     if mode == 'test':
    #         log_param("test_acc", model_eval.get_accuracy_score())
    #
    def publish_history(self, history):

        self.log_params(params=history["params"])
        self.log_metrics(metrics=history["metrics"])
        self.log_artifacts(files_dict=history["files"])

        # save plot: https://towardsdatascience.com/tracking-ml-experiments-using-mlflow-7910197091bb
