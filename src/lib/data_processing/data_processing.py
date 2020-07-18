"""
===========================================================================================
Database Processing Package
===========================================================================================
"""
import string
import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import sys
from src.lib.nlp.nlp_util import NLPUtils
from src.lib.utils.util import Util
from src.lib.data_visualization.data_plotting import DataPlotting

TXT_TOKENIZATION_COLUMN = "TXT_TOKENS"


class DataProcessing:
    def __init__(self, param):
        """Constructor for this class"""
        self.param = param
        self.history = {"params": {}, "metrics": {}, "files": {}}
        self.samples_lifecycle = []

        # registering params
        self.include_param_history(dict=self.param)

    def load_data(self)->pd:
        logging.info("======================================================================")
        logging.info("Loading data ...")

        # flatting txt columns
        if len(self.param.txt_variables)>0:
            txt_variables_flat = Util.flat_lists(sublist=self.param.txt_variables)
        else:
            txt_variables_flat = []

        # checking other variables
        features = 0
        if len(self.param.numerical_variables) > 0:
            features = features + 1
        if len(self.param.categorical_variables) > 0:
            features = features + 1

        if features > 0:
            columns = Util.join_lists(
                l1=self.param.numerical_variables,
                l2=self.param.categorical_variables,
                l3=txt_variables_flat,
            )
        else:
            columns = txt_variables_flat

        # exclude duplicate variables
        columns = Util.get_unique_list(columns)

        # loading local data file
        if self.param.data_source == "localhost_datafile":
            try:
                data = self.load_csv_database(
                    filepath=self.param.data_file_path,
                    separator=self.param.separator,
                    selected_columns=columns,
                    perc_sample=self.param.perc_load,
                    select_sample=self.param.mode_load,
                    order="asc",
                )
                self.samples_lifecycle.append(data.shape[0])
            except:
                logging.error("Ops " + str(sys.exc_info()[0]) + " occured!")
                raise

        return data

    def load_train_data(self)->pd:
        logging.info("======================================================================")
        logging.info("Loading training data ...")

        # flatting txt columns
        if len(self.param.txt_inputs)>0:
            txt_variables_flat = Util.flat_lists(sublist=self.param.txt_variables)
        else:
            txt_variables_flat = []

        # checking other variables
        features = 0
        if len(self.param.numerical_inputs) > 0:
            features = features + 1
        if len(self.param.categorical_inputs) > 0:
            features = features + 1

        if features > 0:
            columns = Util.join_lists(
                l1=self.param.numerical_inputs,
                l2=self.param.categorical_inputs,
                l3=txt_variables_flat,
                l4=self.param.output_target
            )

            columns_input = Util.join_lists(
                l1=self.param.numerical_inputs,
                l2=self.param.categorical_inputs,
                l3=txt_variables_flat
            )
        else:
            columns = txt_variables_flat

        # exclude duplicate variables
        columns = Util.get_unique_list(columns)
        columns_input = Util.get_unique_list(columns_input)

        # loading input data
        if self.param.data_source == "localhost_datafile":
            try:
                data = self.load_csv_database(
                    filepath=self.param.data_train_file_path,
                    separator=self.param.separator,
                    selected_columns=columns,
                    perc_sample=self.param.perc_load,
                    select_sample=self.param.mode_load,
                    order="asc",
                )

            except:
                logging.error("Ops " + str(sys.exc_info()[0]) + " occured!")
                raise

        data_input = data[columns_input]
        data_target = data[self.param.output_target]

        return data_input, data_target

    def prep_rawdata(self, data: pd) -> pd:

        logging.info("======================================================================")
        logging.info("Starting data processing ...")

        # Register original sample count
        n_original_rows = data.shape[0]

        logging.info("Original number of samples: {a:.3f}".format(a=(n_original_rows)))

        # flatting txt columns - concatenated selection cleaning
        input_txt_features_flat = Util.flat_lists(sublist=self.param.txt_variables)
        input_txt_features_flat = Util.get_unique_list(input_txt_features_flat)

        # ========================================
        # Missing data processing
        logging.info("======================================================================")
        logging.info("Processing missing data:")

        # ========================================
        # Processing missing data for input number features
        if len(self.param.numerical_variables) > 0:
            logging.info("======================================================================")
            logging.info(
                "Processing numerical variables with: " + self.param.missing_number_inputs
            )

            if self.param.missing_number_inputs == "delete":
                data = self.missing_delete(data=data, columns=self.param.numerical_variables)

            logging.info("Dataframe shape = " + str(data.shape))
            self.samples_lifecycle.append(data.shape[0])

        # ========================================
        # Processing missing data for categorical number features
        if len(self.param.categorical_variables) > 0:
            logging.info("======================================================================")
            logging.info(
                "Processing input categorical features with: "
                + self.param.missing_categorical_inputs
            )

            if self.param.missing_categorical_inputs == "delete":
                data = self.missing_delete(data=data, columns=self.param.categorical_variables)

            logging.info("Dataframe shape = " + str(data.shape))
            self.samples_lifecycle.append(data.shape[0])

        # ========================================
        # Processing missing data for input txt features
        if len(self.param.txt_variables) > 0:
            logging.info("======================================================================")
            logging.info("Processing input txt features with: " + self.param.missing_txt_inputs)

            if self.param.missing_txt_inputs == "delete":
                data = self.missing_delete(data=data, columns=input_txt_features_flat)

            logging.info("Dataframe shape = " + str(data.shape))
            self.samples_lifecycle.append(data.shape[0])

        # ========================================
        # Processing missing data for output features
        if len(self.param.output_target) > 0:
            logging.info("======================================================================")
            logging.info("Processing output features with: " + self.param.missing_outputs)

            if self.param.missing_outputs == "delete":
                data = self.missing_delete(data=data, columns=self.param.output_target)

            logging.info("Dataframe shape = " + str(data.shape))
            self.samples_lifecycle.append(data.shape[0])

        # ========================================
        # Delete repeated samples
        # ========================================
        if self.param.delete_repeated_samples:
            logging.info("======================================================================")
            logging.info("Deleting repeated samples")
            data = self.delete_repeated_rows(data=data)
            logging.info("Dataframe shape = " + str(data.shape))
            self.samples_lifecycle.append(data.shape[0])

        # ========================================
        # Delete outliers samples
        # ========================================
        if self.param.remove_outliers:
            logging.info("======================================================================")
            logging.info(
                "Deleting numerical outliers samples with : " + self.param.outliers_method
            )
            if self.param.outliers_method == "k_std":
                data = self.remove_outliers_std(
                    data=data,
                    columns_name=self.param.numerical_variables,
                    k_factor=self.param.k_numerical_outlier_factor,
                )
            else:
                logging.error("Outlier method not recognized")

            logging.info("Dataframe shape = " + str(data.shape))
            self.samples_lifecycle.append(data.shape[0])

        # ========================================
        # Processing txt features
        if len(self.param.txt_variables) > 0:
            logging.info("======================================================================")
            logging.info("Processing txt variables...")

            # convert to unicode txt
            logging.info("Converting to unicode...")
            data = NLPUtils.convert_to_unicode(dataframe=data, columns=input_txt_features_flat)

            # cleaning html tags
            logging.info("Cleaning html tags...")
            data = NLPUtils.clean_html(dataframe=data, columns=input_txt_features_flat)

            # lower casing
            logging.info("Converting to lower case...")
            data = NLPUtils.convert_to_lower(dataframe=data, columns=input_txt_features_flat)

            # exclude special chars
            logging.info("Excluding special chars...")
            data = NLPUtils.clean_special_char(dataframe=data, columns=input_txt_features_flat)

        # ========================================
        # Registering infos for tracking
        self.include_metric_history(dict={"samples_lifecycle": self.samples_lifecycle})

        return data

    # delete missing values
    def missing_delete(self, data: pd, columns: list) -> pd:
        data.dropna(subset=columns, axis=0, how="any", inplace=True)
        return data

    # Replace missing values to median
    def missing_median(self, data, columns):
        med = data[columns].median()
        data[columns] = data[columns].fillna(med)
        return data

    # Replace missing values to default value
    def missing_default(self, data, name, default_value):
        data[name] = data[name].fillna(default_value)
        return data

    # Remove rows with column value is +/- k * std value
    def remove_outliers_std(self, data: pd, columns_name: list, k_factor: float) -> pd:
        for col in columns_name:
            drop_rows = data.index[
                (np.abs(data[col] - data[col].mean()) >= (k_factor * data[col].std()))
            ]
            data.drop(drop_rows, axis=0, inplace=True)
            logging.info(
                "Column " + str(col) + " dropped " + str(len(drop_rows)) + " outliers samples."
            )
        return data

    def save_dataframe(self, data: pd = None, folder_path: str = None, prefix: str = None) -> bool:

        data.to_csv(folder_path + prefix + "data.tsv", index=False, sep="\t", encoding="utf-8")
        logging.info("File saved in: " + folder_path + prefix + "data.tsv")

        return True

    def save_datasets(self, data_train: pd = None, data_test: pd = None, folder_path: str = None, prefix: str = None) -> bool:

        data_train.to_csv(folder_path + prefix + "train.tsv", index=False, sep="\t", encoding="utf-8")
        logging.info("File saved in: " + folder_path + prefix + "train.tsv")

        data_test.to_csv(folder_path + prefix + "test.tsv", index=False, sep="\t", encoding="utf-8")
        logging.info("File saved in: " + folder_path + prefix + "test.tsv")

        return True

    def delete_repeated_rows(self, data: pd) -> pd:
        data.drop_duplicates(inplace=True)
        return data

    def include_param_history(self, dict):
        self.history["params"].update(dict)
        return True

    def include_metric_history(self, dict):
        self.history["metrics"].update(dict)
        return True

    def include_files_history(self, dict):
        self.history["files"].update(dict)
        return True

    def descriptive_analysis(
        self,
        data: pd,
        view_plots: bool = False,
        save_plots: bool = False,
        save_analysis: bool = False,
        folder_path: str = None,
        prefix: str = None,
    )->bool:

        # ----------------------------------------------------------
        # numerical features analysis
        num_feat_analysis = pd.DataFrame(
            index=self.param.numerical_variables, columns=["Mean", "Std", "Min", "Max"]
        )
        for feat in self.param.numerical_variables:
            # average
            mean = data[feat].mean()
            logging.info("Var: " + str(feat) + " Mean: " + str(mean))
            num_feat_analysis["Mean"].loc[feat] = mean

            # std
            std = data[feat].std()
            logging.info("Var: " + str(feat) + " Std: " + str(std))
            num_feat_analysis["Std"].loc[feat] = std

            # min
            min = data[feat].min()
            logging.info("Var: " + str(feat) + " Min: " + str(min))
            num_feat_analysis["Min"].loc[feat] = min

            # max
            max = data[feat].max()
            logging.info("Var: " + str(feat) + " Max: " + str(max))
            num_feat_analysis["Max"].loc[feat] = max

        if save_analysis:
            filename = prefix + "num_variables.tsv"
            full_path = folder_path + prefix + "num_variables.tsv"
            num_feat_analysis.to_csv(full_path, index=True, sep="\t", encoding="utf-8")
            logging.info(
                "Numerical descriptive analysis saved in: "
                + folder_path
                + prefix
                + "num_features.tsv"
            )
            self.include_files_history({filename: full_path})

        if save_plots:
            dv = DataPlotting(
                dataframe=data,
                view_plots=view_plots,
                save_plots=save_plots,
                folder_path=folder_path,
                prefix=prefix,
            )
            for feat in self.param.numerical_variables:
                src = dv.plot_line_steps(y_column=feat)
                self.include_files_history({src: src})

                src = dv.plot_numerical_histogram(y_column=feat)
                self.include_files_history({src: src})

        # ----------------------------------------------------------
        # categorical features
        cat_feat_analysis = pd.DataFrame(
            index=self.param.categorical_variables, columns=["Count", "Unique", "Top"]
        )

        for feat in self.param.categorical_variables:
            # count
            count = data[feat].count()
            logging.info("Var: " + str(feat) + " Count: " + str(count))
            cat_feat_analysis["Count"].loc[feat] = count

            # unique
            unique = len(data[feat].unique())
            logging.info("Var: " + str(feat) + " Unique: " + str(unique))
            cat_feat_analysis["Unique"].loc[feat] = unique

            # top
            top = Util.get_top_categorical_feature(data=data, column=feat)
            logging.info("Var: " + str(feat) + " Top: " + str(top))
            cat_feat_analysis["Top"].loc[feat] = top

        if save_analysis:
            filename = prefix + "cat_variables.tsv"
            full_path = folder_path + prefix + "cat_variables.tsv"
            cat_feat_analysis.to_csv(full_path, index=True, sep="\t", encoding="utf-8")
            logging.info(
                "Categorical descriptive analysis saved in: "
                + folder_path
                + prefix
                + "cat_features.tsv"
            )
            self.include_files_history({filename: full_path})

        if save_plots:
            dv = DataPlotting(
                dataframe=data,
                view_plots=view_plots,
                save_plots=save_plots,
                folder_path=folder_path,
                prefix=prefix,
            )
            for feat in self.param.categorical_variables:
                src = dv.plot_count_cat_histogram(y_column=feat)
                if src is not False:
                    self.include_files_history({src: src})

        return True

    def load_csv_database(
        self,
        filepath: str,
        separator: str,
        selected_columns: list,
        perc_sample: float,
        select_sample: str,
        order: str,
    ) -> pd:

        # number of rows from file without header
        num_lines = sum(1 for l in open(filepath)) - 1

        # sample lines size
        num_lines_selected = int(perc_sample * num_lines)
        skip_lines = num_lines - num_lines_selected

        logging.info("Total samples: {a:.1f}".format(a=num_lines))
        logging.info("Total samples target: {a:.1f}".format(a=num_lines_selected))

        # Partial loading
        if perc_sample < 1:
            lines2skip = []
            if select_sample == "random":
                lines2skip = np.random.choice(
                    np.arange(1, num_lines + 1), (num_lines - num_lines_selected), replace=False
                )

            if select_sample == "sequential":
                if order == "asc":
                    lines2skip = [x for x in range(1, num_lines + 1) if x > num_lines_selected]

                if order == "desc":
                    lines2skip = [
                        x for x in range(1, num_lines + 1) if x <= (num_lines - num_lines_selected)
                    ]

            if len(selected_columns) > 0:
                df = pd.read_csv(
                    filepath,
                    header=0,
                    sep=separator,
                    usecols=selected_columns,
                    skiprows=lines2skip,
                    encoding="utf-8",
                    low_memory=True,
                    quoting=csv.QUOTE_NONE,
                    warn_bad_lines=True,
                    skipinitialspace=True,
                )
            else:
                df = pd.read_csv(
                    filepath,
                    header=0,
                    sep=separator,
                    skiprows=lines2skip,
                    encoding="utf-8",
                    low_memory=True,
                    quoting=csv.QUOTE_NONE,
                    warn_bad_lines=True,
                    skipinitialspace=True,
                )
        # Integral loading
        else:
            if len(selected_columns) > 0:
                df = pd.read_csv(
                    filepath,
                    header=0,
                    sep=separator,
                    usecols=selected_columns,
                    encoding="utf-8",
                    low_memory=True,
                    quoting=csv.QUOTE_NONE,
                    warn_bad_lines=True,
                    skipinitialspace=True,
                )
            else:
                df = pd.read_csv(
                    filepath,
                    header=0,
                    sep=separator,
                    encoding="utf-8",
                    low_memory=True,
                    quoting=csv.QUOTE_NONE,
                    warn_bad_lines=True,
                    skipinitialspace=True,
                )

        logging.info("Selected dataset samples: {a:.1f}".format(a=df.shape[0]))
        logging.info("Number of variables: {a:.1f}".format(a=df.shape[1]))
        logging.info("Variables list: {a:s}".format(a=str(df.columns.values.tolist())))

        return df

    def build_dataset(self, data: pd) -> pd:

        # spliting subsets for model building
        data_train, data_test = self.split_data_subsets(data=data)

        return data_train, data_test


    def split_data_subsets(self, data: pd) -> pd:

        logging.info('Processing subset selection...')

        # Splitting the dataset into the Training set and Test set
        data_train, data_test = train_test_split(data,
                                                 test_size=self.param.test_subset,
                                                 shuffle=self.param.test_shuffle,
                                                 random_state=None)

        logging.info('Train and test subsets with shuffle = ' + str(self.param.test_shuffle))
        logging.info('Train samples: ' + str(data_train.shape))
        logging.info('Test samples: ' + str(data_test.shape))

        # Registering tracking
        self.include_metric_history(dict={'samples_lifecycle': self.samples_lifecycle})
        self.include_param_history(
            dict={'train_samples_dim': data_train.shape, 'test_samples_dim': data_test.shape})

        return data_train, data_test
