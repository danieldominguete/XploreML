"""
===========================================================================================
Static2Class : Model building from static data to class prediction
===========================================================================================
"""
# =========================================================================================
# Importing the libraries
import logging
import os, sys, inspect
from dotenv import load_dotenv, find_dotenv
import argparse

# Include root folder to path
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(currentdir)))
sys.path.insert(0, parentdir)

from src.lib.utils.util import Util
from src.lib.data_schemas.static2class_parameters import Static2ClassParameters
from src.lib.data_schemas.environment_parameters import EnvironmentParameters
from src.lib.data_processing.data_processing import DataProcessing
from src.lib.environment.environment import Environment
from src.lib.data_schemas.k_nearest_neighbors_parameters import XKNearestNeighborsParameters
from src.lib.model.k_nearest_neighbors import XKNearestNeighbors
from src.lib.data_schemas.logistic_regression_parameters import XLogisticRegressionParameters
from src.lib.model.logistic_regression import XLogisticRegression
from src.lib.data_schemas.svm_parameters import XSVMParameters
from src.lib.model.svm import XSVM
from src.lib.data_schemas.decision_tree_parameters import XDecisionTreeParameters
from src.lib.model.decision_tree import XDecisionTree
from src.lib.data_schemas.random_forest_parameters import XRandomForestParameters
from src.lib.model.random_forest import XRandomForest
from src.lib.data_schemas.neural_FFNN_parameters import XNeuralFeedForwardParameters
from src.lib.model.neural_FFNN import XNeuralFeedForward
from src.lib.model.model_evaluation import ClassificationModelEvaluation


class BuildStatic2ValueMain:
    def __init__(self, parameters_file):

        """Constructor for this class"""
        self.parameters_file = parameters_file

    def model_selection(self, data_config, data_param, environment):

        # Validade and load model hyperparameters
        if data_param.model_type == "k_nearest_neighbors":
            model_param = XKNearestNeighborsParameters(
                **data_config.get("k_nearest_neighbors_parameters")
            )
            model = XKNearestNeighbors(param=model_param, application=data_param.application, application_type=data_param.classification_type)

        elif data_param.model_type == "logistic_regression":
            model_param = XLogisticRegressionParameters(
                **data_config.get("logistic_regression_parameters")
            )
            model = XLogisticRegression(param=model_param, application=data_param.application, application_type=data_param.classification_type)


        elif data_param.model_type == "svm":
            model_param = XSVMParameters(
                **data_config.get("svm_parameters")
            )
            model = XSVM(param=model_param, application=data_param.application, application_type=data_param.classification_type)

        elif data_param.model_type == "decision_tree":
            model_param = XDecisionTreeParameters(
                **data_config.get("decision_tree_parameters")
            )
            model = XDecisionTree(param=model_param, application=data_param.application, application_type=data_param.classification_type)

        elif data_param.model_type == "random_forest":
            model_param = XRandomForestParameters(
                **data_config.get("random_forest_parameters")
            )
            model = XRandomForest(param=model_param, application=data_param.application, application_type=data_param.classification_type)

        elif data_param.model_type == "neural_feedforward":
            model_param = XNeuralFeedForwardParameters(
                **data_config.get("neural_feedforward_parameters")
            )
            model = XNeuralFeedForward(param=model_param,
                                       application=data_param.application,
                                       application_type=data_param.classification_type,
                                       env=environment)

        else:
            logging.error('Model type not valid.')

        return model

    def run(self):

        # =========================================================================================
        # Environment variables
        load_dotenv(find_dotenv())
        PYTHON_WARNINGS = os.getenv("PYTHON_WARNINGS")

        # ===========================================================================================
        # Script Setup

        # Loading json file
        data_config = Util.load_parameters_from_file(path_file=self.parameters_file)

        # Validate parameters and load environment class
        env_param = EnvironmentParameters(**data_config.get("environment_parameters"))
        env = Environment(param=env_param)

        # Validade parameters and load data processing class
        data_param = Static2ClassParameters(**data_config.get("static2class_parameters"))
        ds = DataProcessing(param=data_param)

        # ===========================================================================================
        # Setup environment
        env.init_script(script_name=os.path.basename(__file__), warnings_level=PYTHON_WARNINGS)

        # ===========================================================================================
        # Loading data
        logging.info("======================================================================")
        logging.info("Fit and Transform Training Data:")
        data_train_input, data_train_target = ds.load_dataset(subset='train')


        logging.info("======================================================================")
        logging.info("Fit and Transform Training Data:")
        (
            data_train_input,
            data_train_target,
            input_var_dict,
            target_var_dict,
            numerical_input_encoder_list,
            categorical_input_encoder_int_list,
            categorical_input_encoder_hot_list,
            categorical_input_encoder_bin_list,
            categorical_input_int_to_cat_dict_list,
            categorical_input_cat_to_int_dict_list,
            txt_int_to_word_dict_list_input,
            txt_word_to_int_dict_list_input,
            numerical_output_encoder_list,
            categorical_output_encoder_int_list,
            categorical_output_encoder_hot_list,
            categorical_output_encoder_bin_list,
            int_to_cat_dict_list_output_list,
            cat_to_int_dict_list_output_list,
        ) = ds.fit_transform_train_data(
            data_train_input=data_train_input,
            data_train_target=data_train_target
        )

        logging.info("======================================================================")
        logging.info("Building Model:")

        model = self.model_selection(data_config=data_config, data_param=data_param, environment=env)

        model.fit(
            data_input=data_train_input,
            data_target=data_train_target,
            input_var_dict=input_var_dict,
            target_var_dict=target_var_dict,
            target_cat_dict=cat_to_int_dict_list_output_list
        )

        logging.info("======================================================================")
        logging.info("Building predictions:")

        data_train_predict = model.eval_predict(
            data_input=data_train_input,
            input_var_dict=input_var_dict,
            int_to_cat_dict_target=int_to_cat_dict_list_output_list[0])

        logging.info("======================================================================")
        logging.info("Training Results")

        model_eval_train = ClassificationModelEvaluation(
            Y_target=data_train_target[data_param.output_target],
            Y_predict=data_train_predict[['predict']],
            Y_reliability=data_train_predict[['reliability']],
            subset_label="eval_train_",
            classification_type=data_param.classification_type,
            Y_int_to_cat_labels=int_to_cat_dict_list_output_list[0],
            Y_cat_to_int_labels=cat_to_int_dict_list_output_list[0],
            train_history=model.history,
        )

        # checking metrics
        model_eval_train.execute()
        # ===========================================================================================
        # Saving files
        logging.info("======================================================================")
        logging.info("Saving Training Results:")

        # prediction report
        prediction_report = model_eval_train.get_prediction_report()
        Util.save_dataframe(data=prediction_report, folder_path=env.run_folder,
                            prefix=env.prefix_name + "pred_train_report")

        # classes performance
        classes_report = model_eval_train.get_score_by_classes()
        Util.save_dataframe(data=classes_report, folder_path=env.run_folder,
                            prefix=env.prefix_name + "class_train_report")

        # classes confusion
        confusion_report = model_eval_train.get_confusion_classes()
        Util.save_dataframe(data=confusion_report, folder_path=env.run_folder,
                            prefix=env.prefix_name + "confusion_train_report")

        # Reliability sensitivity
        reliability_report = model_eval_train.get_reliability_sensitivity()
        Util.save_dataframe(data=reliability_report, folder_path=env.run_folder,
                            prefix=env.prefix_name + "reliability_train_report")

        # ===========================================================================================
        # ploting results
        if env_param.view_plots or env_param.save_plots:
            logging.info("======================================================================")
            logging.info("Plotting training result graphs")

            model_eval_train.plot_training_results(
                view=env_param.view_plots,
                save=env_param.save_plots,
                path=env.run_folder,
                prefix=env.prefix_name + "train_",
            )

            logging.info("Plotting reliability sensitivity")
            model_eval_train.plot_reliability_sensitivity(report=reliability_report, view=env_param.view_plots,
                                                          save=env_param.save_plots, path=env.run_folder,
                                                          prefix=env.prefix_name + "reliability_",
                                                          title='Train Sensitivity')

        # ===========================================================================================
        # Evaluating test dataset
        # ===========================================================================================
        # Loading data
        logging.info("======================================================================")
        logging.info("Loading Test Data:")

        # exclude data_train for memory optimization
        del (data_train_input)
        del (data_train_target)
        del (data_train_predict)

        # loading test data
        data_test_input, data_test_target = ds.load_dataset(subset='test')

        logging.info("======================================================================")
        logging.info("Transform Test Data:")
        (
            data_test_input,
            data_test_target,
        ) = ds.transform_test_data(
            data_test_input=data_test_input,
            data_test_target=data_test_target,
            input_var_dict=input_var_dict,
            target_var_dict=input_var_dict,
            numerical_input_encoder_list=numerical_input_encoder_list,
            categorical_input_encoder_int_list=categorical_input_encoder_int_list,
            categorical_input_encoder_hot_list=categorical_input_encoder_hot_list,
            categorical_input_encoder_bin_list=categorical_input_encoder_bin_list,
            categorical_int_to_cat_dict_list_input=categorical_input_int_to_cat_dict_list,
            categorical_cat_to_int_dict_list_input=categorical_input_cat_to_int_dict_list,
            txt_int_to_word_dict_list_input=txt_int_to_word_dict_list_input,
            txt_word_to_int_dict_list_input=txt_word_to_int_dict_list_input,
            numerical_output_encoder_list=numerical_output_encoder_list,
            categorical_output_encoder_int_list=categorical_output_encoder_int_list,
            categorical_output_encoder_hot_list=categorical_output_encoder_hot_list,
            categorical_output_encoder_bin_list=categorical_output_encoder_bin_list,
            int_to_cat_dict_list_output_list=int_to_cat_dict_list_output_list,
            cat_to_int_dict_list_output_list=cat_to_int_dict_list_output_list,
        )

        logging.info("======================================================================")
        logging.info("Test Results")
        data_test_predict = model.eval_predict(data_input=data_test_input,
                                               input_var_dict=input_var_dict,
                                               int_to_cat_dict_target=int_to_cat_dict_list_output_list[0])

        model_eval_test = ClassificationModelEvaluation(
            Y_target=data_test_target[data_param.output_target],
            Y_predict=data_test_predict[['predict']],
            Y_reliability=data_test_predict[['reliability']],
            subset_label="eval_test_",
            classification_type=data_param.classification_type,
            Y_int_to_cat_labels=int_to_cat_dict_list_output_list[0],
            Y_cat_to_int_labels=cat_to_int_dict_list_output_list[0],
            train_history=None
        )

        # checking metrics
        model_eval_test.execute()

        # ===========================================================================================
        # Saving files
        logging.info("======================================================================")
        logging.info("Saving Testing Results:")

        # prediction report
        prediction_report = model_eval_test.get_prediction_report()
        Util.save_dataframe(data=prediction_report, folder_path=env.run_folder,
                            prefix=env.prefix_name + "pred_test_report")

        # classes performance
        classes_report = model_eval_test.get_score_by_classes()
        Util.save_dataframe(data=classes_report, folder_path=env.run_folder,
                            prefix=env.prefix_name + "class_test_report")

        # classes confusion
        confusion_report = model_eval_test.get_confusion_classes()
        Util.save_dataframe(data=confusion_report, folder_path=env.run_folder,
                            prefix=env.prefix_name + "confusion_test_report")

        # Reliability sensitivity
        reliability_report = model_eval_test.get_reliability_sensitivity()
        Util.save_dataframe(data=reliability_report, folder_path=env.run_folder,
                            prefix=env.prefix_name + "reliability_test_report")

        # ===========================================================================================
        # ploting results
        if env_param.view_plots or env_param.save_plots:
            logging.info("======================================================================")
            logging.info("Plotting test result graphs")

            model_eval_train.plot_test_results(
                view=env_param.view_plots,
                save=env_param.save_plots,
                path=env.run_folder,
                prefix=env.prefix_name + "test_",
            )

            logging.info("Plotting reliability sensitivity")
            model_eval_train.plot_reliability_sensitivity(report=reliability_report, view=env_param.view_plots,
                                                          save=env_param.save_plots, path=env.run_folder,
                                                          prefix=env.prefix_name + "reliability_",
                                                          title='Test Sensitivity')

        # ===========================================================================================
        # Register tracking info
        if env.param.tracking:
            env.publish_results(history=ds.history)
            env.publish_results(history=model.history)
            env.publish_results(history=model_eval_train.history)
            env.publish_results(history=model_eval_test.history)
            env.tracking.log_artifacts_folder(local_dir=env.run_folder)

        # ===========================================================================================
        # Script Performance
        env.close_script()
        # ===========================================================================================


# ===========================================================================================
# ===========================================================================================
# Main call from terminal
if __name__ == "__main__":
    """
    Call from terminal command
    """

    # getting script arguments
    parser = argparse.ArgumentParser(
        description="XploreML - Script Main"
    )
    parser.add_argument(
        "-f", "--config_file_json", help="Json config file for script execution", required=True
    )

    args = parser.parse_args()

    # Running script main
    try:
        processor = BuildStatic2ValueMain(parameters_file=args.config_file_json)
        processor.run()
    except:
        logging.error("Ops " + str(sys.exc_info()[0]) + " occured!")
        raise
# ===========================================================================================
# ===========================================================================================
