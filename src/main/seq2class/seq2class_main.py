"""
===========================================================================================
Seq2Class : Model building from sequence data to class prediction
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
from src.lib.data_schemas.seq2class_parameters import Seq2ClassParameters
from src.lib.data_schemas.environment_parameters import EnvironmentParameters
from src.lib.data_processing.data_processing import DataProcessing
from src.lib.environment.environment import Environment
from src.lib.model.model_evaluation import ClassificationModelEvaluation
from src.lib.data_schemas.neural_RNN_parameters import XNeuralRecurrentParameters
from src.lib.model.neural_RNN import XNeuralRecurrent


class BuildSeq2ClassMain:
    def __init__(self, parameters_file):

        """Constructor for this class"""
        self.parameters_file = parameters_file

    def model_selection(self, data_config, data_param, environment):

        # Validade and load model hyperparameters
        if data_param.model_type == "neural_recurrent":
            model_param = XNeuralRecurrentParameters(
                **data_config.get("neural_recurrent_parameters")
            )
            model = XNeuralRecurrent(param=model_param,
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
        data_param = Seq2ClassParameters(**data_config.get("seq2class_parameters"))
        ds = DataProcessing(param=data_param)

        # ===========================================================================================
        # Setup environment
        env.init_script(script_name=os.path.basename(__file__), warnings_level=PYTHON_WARNINGS)

        # ===========================================================================================
        # Loading data
        logging.info("======================================================================")
        logging.info("Loading Training Data:")
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
            subset_label="Train",
            classification_type=data_param.classification_type,
            Y_int_to_cat_labels=int_to_cat_dict_list_output_list[0],
            Y_cat_to_int_labels=cat_to_int_dict_list_output_list[0],
            history=None,
        )

        model_eval_train.print_evaluation_scores()
        #env.tracking.publish_c_eval(model_eval=model_eval_train, mode="train")

        if env_param.view_plots or env_param.save_plots:
            logging.info("======================================================================")
            logging.info("Plotting training result graphs")

            if env_param.save_plots:
                logging.info("Plots will save in " + env.run_folder)

            if env_param.view_plots:
                logging.info("Plots will view in window popup")

            model_eval_train.plot_evaluation_scores(
                view=env_param.view_plots,
                save=env_param.save_plots,
                path=env.run_folder,
                prefix=env.prefix_name + "train_",
            )

        # ===========================================================================================
        # Loading data
        logging.info("======================================================================")
        logging.info("Loading Test Data:")

        # exclude data_train for memory optimization
        del(data_train_input)
        del(data_train_target)
        del(data_train_predict)

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
            input_var_dict = input_var_dict,
            target_var_dict = input_var_dict,
            numerical_input_encoder_list = numerical_input_encoder_list,
            categorical_input_encoder_int_list = categorical_input_encoder_int_list,
            categorical_input_encoder_hot_list = categorical_input_encoder_hot_list,
            categorical_input_encoder_bin_list = categorical_input_encoder_bin_list,
            categorical_int_to_cat_dict_list_input = categorical_input_int_to_cat_dict_list,
            categorical_cat_to_int_dict_list_input = categorical_input_cat_to_int_dict_list,
            txt_int_to_word_dict_list_input = txt_int_to_word_dict_list_input,
            txt_word_to_int_dict_list_input = txt_word_to_int_dict_list_input,
            numerical_output_encoder_list = numerical_output_encoder_list,
            categorical_output_encoder_int_list = categorical_output_encoder_int_list,
            categorical_output_encoder_hot_list = categorical_output_encoder_hot_list,
            categorical_output_encoder_bin_list = categorical_output_encoder_bin_list,
            int_to_cat_dict_list_output_list = int_to_cat_dict_list_output_list,
            cat_to_int_dict_list_output_list = cat_to_int_dict_list_output_list,
        )

        logging.info("======================================================================")
        logging.info("Test Results")
        data_test_predict = model.eval_predict(data_input=data_test_input,
                                               input_var_dict=input_var_dict,
                                               int_to_cat_dict_target=int_to_cat_dict_list_output_list[0])

        model_eval_test = ClassificationModelEvaluation(
            Y_target=data_test_target[data_param.output_target],
            Y_predict=data_test_predict[['predict']],
            subset_label="Test",
            classification_type=data_param.classification_type,
            Y_int_to_cat_labels=int_to_cat_dict_list_output_list[0],
            Y_cat_to_int_labels=cat_to_int_dict_list_output_list[0],
            history=None,
        )

        model_eval_test.print_evaluation_scores()
        #env.tracking.publish_regression_eval(model_eval=model_eval_test, mode="test")

        if env_param.view_plots or env_param.save_plots:
            logging.info("======================================================================")
            logging.info("Plotting test result graphs")

            if env_param.save_plots:
                logging.info("Plots will save in " + env.run_folder)

            if env_param.view_plots:
                logging.info("Plots will view in window popup")

            model_eval_test.plot_evaluation_scores(
                view=env_param.view_plots,
                save=env_param.save_plots,
                path=env.run_folder,
                prefix=env.prefix_name + "test_",
            )

        # ===========================================================================================
        # Saving model
        logging.info("======================================================================")
        logging.info("Saving Results:")

        # ===========================================================================================
        # Register tracking info
        if env.param.tracking:
            env.publish_results(history=ds.history)
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
        description="XploreML - Script Main for Dataset Vizualization"
    )
    parser.add_argument(
        "-f", "--config_file_json", help="Json config file for script execution", required=True
    )

    args = parser.parse_args()

    # Running script main
    try:
        processor = BuildSeq2ClassMain(parameters_file=args.config_file_json)
        processor.run()
    except:
        logging.error("Ops " + str(sys.exc_info()[0]) + " occured!")
        raise
# ===========================================================================================
# ===========================================================================================
