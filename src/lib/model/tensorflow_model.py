"""
===========================================================================================
Tensor Flow Model Building Package (version:
===========================================================================================
Script Reviewed by COGNAS
===========================================================================================
"""

import os
import numpy as np
#from tensorflow import keras as k
import tensorflow as tf
import logging


class XTensorFlowModel:
    def __init__(self, param=None):
        """Constructor for this class"""
        self._model_parameters = param
        self._callbacks = []
        self._model = None
        self._history = None
        self._statefull = False
        self._topology_id = None
        self._topology_details = None
        self._run_folder_path = None
        self._prefix_name = None
        self._tracking = False
        self._application = None
        self._application_type = None

    def set_application(self, value: str) -> bool:
        self._application = value
        return True

    def set_application_type(self, value: str) -> bool:
        self._application_type = value
        return True

    def set_topology_id(self, topology: str) -> bool:
        self._topology_id = topology
        return True

    def set_topology_details(self, topology: dict) -> bool:
        self._topology_details = topology
        return True

    def set_run_folder_path(self, path: str) -> bool:
        self._run_folder_path = path
        return True

    def set_prefix_name(self, prefix: str) -> bool:
        self._prefix_name = prefix
        return True

    def set_tracking(self, value: bool) -> bool:
        self._tracking = value
        return True

    def fit(self, X, Y, input_var_dict):

        # Logging infos
        logging.info("TensorFlow version: ", tf.__version__)
        logging.info("GPU Available: ", tf.test.is_gpu_available())

        # Build architeture
        self.set_architeture(inputs=X, outputs=Y, input_var_dict=input_var_dict)

        self._history = self._model.fit(
            X,
            Y[0],
            batch_size=self._model_parameters.batch_size,
            epochs=self._model_parameters.epochs,
            validation_split=self._model_parameters.validation_split,
            callbacks=self._callbacks,
            verbose=self._model_parameters.verbose,
            shuffle=self._model_parameters.shuffle,
        )

        return True

    def predict(self, X):
        if self._statefull:
            Y = self._model.predict(X, batch_size=self._model_parameters.batch_size)
            self._model.reset_states()
        else:
            Y = self._model.predict(X)

        return Y

    def set_architeture(self, inputs=None, outputs=None, input_var_dict=None) -> bool:

        # Check pre-models
        if self._topology_id == "DNN-FCCx":

            self._model = k.Sequential()
            teste = len(self._model_parameters.hidden_nodes)
            for i in range(len(self._model_parameters.hidden_nodes)):
                if i == 0:
                    self._model.add(
                        k.layers.Dense(
                            units=self._model_parameters.hidden_nodes[0],
                            use_bias=True,
                            bias_initializer="zeros",
                            kernel_initializer="glorot_uniform",
                            input_dim=input_shape[1],
                            activation=self._model_parameters.hidden_func_nodes[0].value,
                        )
                    )
                    self._model.add(k.layers.Dropout(self._model_parameters.hidden_dropout[0]))
                else:
                    self._model.add(
                        k.layers.Dense(
                            units=self._model_parameters.hidden_nodes[i],
                            activation=self._model_parameters.hidden_func_nodes[i].value,
                        )
                    )
                    self._model.add(k.layers.Dropout(self._model_parameters.hidden_dropout[i]))

            # last layer
            self._model.add(
                k.layers.Dense(
                    units=output_shape[1],
                    activation=self._model_parameters.output_func_nodes.value,
                )
            )

        if self._topology_id == "RNN-LTSMx-FCCx":

            # List of inputs with [samples, time steps, features] matrix
            # sequence of X inputs is number, cat , txt
            seq = 0
            input_list = []
            seq_in_list = []

            # sequences of number
            input_feature_list = input_var_dict.get("number_inputs")
            if len(input_feature_list) > 0:
                for i in range(len(input_feature_list)):
                    in_layer = tf.keras.Input(shape=(inputs[seq].shape[1], inputs[seq].shape[2]))

                    # input sequence LSTM stack
                    for stack in range(len(self._topology_details.get("seq_hidden_nodes")[seq])):
                        # first layer sequence encoder
                        if stack == 0:
                            # last layer too
                            if stack == len(self._topology_details.get("seq_hidden_nodes")[seq]) - 1:
                                return_sequences = False
                            else:
                                return_sequences = True

                            # adjust return sequences
                            seq_in_layer = tf.keras.layers.LSTM(
                                units=self._topology_details.get("seq_hidden_nodes")[seq][stack],
                                return_sequences=return_sequences,
                                dropout=self._topology_details.get("seq_hidden_dropout")[seq][stack],
                            )(in_layer)

                        # last layer sequence encoder
                        elif stack == len(self._topology_details.get("seq_hidden_nodes")[seq]) - 1:
                            seq_in_layer = tf.keras.layers.LSTM(
                                units=self._topology_details.get("seq_hidden_nodes")[seq][stack],
                                return_sequences=False,
                                dropout=self._topology_details.get("seq_hidden_dropout")[seq][stack],
                            )(in_layer)

                        # intermediate sequence encoder
                        else:
                            seq_in_layer = tf.keras.layers.LSTM(
                                units=self._topology_details.get("seq_hidden_nodes")[seq][stack],
                                return_sequences=True,
                                dropout=self._topology_details.get("seq_hidden_dropout")[seq][stack],
                            )(in_layer)

                    input_list.append(in_layer)
                    seq_in_list.append(seq_in_layer)
                    seq = seq + 1

            # sequences of categorical
            input_feature_list = input_var_dict.get("categorical_inputs")
            if len(input_feature_list) > 0:
                for i in range(len(input_feature_list)):
                    in_layer = tf.keras.Input(shape=(inputs[seq].shape[1], inputs[seq].shape[2]))

                    # input sequence LSTM stack
                    for stack in range(len(self._topology_details.get("seq_hidden_nodes")[seq])):
                        # first layer sequence encoder
                        if stack == 0:
                            # last layer too
                            if stack == len(self._topology_details.get("seq_hidden_nodes")[seq]) - 1:
                                return_sequences = False
                            else:
                                return_sequences = True

                            # adjust return sequences
                            seq_in_layer = tf.keras.layers.LSTM(
                                units=self._topology_details.get("seq_hidden_nodes")[seq][stack],
                                return_sequences=return_sequences,
                                dropout=self._topology_details.get("seq_hidden_dropout")[seq][stack],
                            )(in_layer)

                        # last layer sequence encoder
                        elif stack == len(self._topology_details.get("seq_hidden_nodes")[seq]) - 1:
                            seq_in_layer = tf.keras.layers.LSTM(
                                units=self._topology_details.get("seq_hidden_nodes")[seq][stack],
                                return_sequences=False,
                                dropout=self._topology_details.get("seq_hidden_dropout")[seq][stack],
                            )(in_layer)

                        # intermediate sequence encoder
                        else:
                            seq_in_layer = tf.keras.layers.LSTM(
                                units=self._topology_details.get("seq_hidden_nodes")[seq][stack],
                                return_sequences=True,
                                dropout=self._topology_details.get("seq_hidden_dropout")[seq][stack],
                            )(in_layer)

                    input_list.append(in_layer)
                    seq_in_list.append(seq_in_layer)
                    seq = seq + 1

            # sequences of txt
            input_feature_list = input_var_dict.get("txt_inputs")
            if len(input_feature_list) > 0:
                for i in range(len(input_feature_list)):
                    in_layer = tf.keras.Input(shape=(inputs[seq].shape[1],))
                    # input_dim = vocabulary_size
                    vocab_size = np.max(inputs[seq]) + 1
                    # output_dim = word embedded vector dim
                    # input_length = time steps of input
                    in_embedding_layer = tf.keras.layers.Embedding(
                        input_dim=vocab_size,
                        output_dim=int(vocab_size * 0.1),
                        input_length=inputs[seq].shape[1],
                        trainable=True,
                    )(in_layer)

                    # input sequence LSTM stack
                    for stack in range(len(self._topology_details.get("seq_hidden_nodes")[seq])):
                        # first layer sequence encoder
                        if stack == 0:
                            # last layer too
                            if stack == len(self._topology_details.get("seq_hidden_nodes")[seq]) - 1:
                                return_sequences = False
                            else:
                                return_sequences = True

                            # adjust return sequences
                            seq_in_layer = tf.keras.layers.LSTM(
                                units=self._topology_details.get("seq_hidden_nodes")[seq][stack],
                                return_sequences=return_sequences,
                                dropout=self._topology_details.get("seq_hidden_dropout")[seq][stack],
                            )(in_embedding_layer)

                        # last layer sequence encoder
                        elif stack == len(self._topology_details.get("seq_hidden_nodes")[seq])-1:
                            seq_in_layer = tf.keras.layers.LSTM(
                                units=self._topology_details.get("seq_hidden_nodes")[seq][stack],
                                return_sequences=False,
                                dropout=self._topology_details.get("seq_hidden_dropout")[seq][stack],
                            )(seq_in_layer)

                        # intermediate sequence encoder
                        else:
                            seq_in_layer = tf.keras.layers.LSTM(
                                units=self._topology_details.get("seq_hidden_nodes")[seq][stack],
                                return_sequences=True,
                                dropout=self._topology_details.get("seq_hidden_dropout")[seq][stack],
                            )(seq_in_layer)


                    input_list.append(in_layer)
                    seq_in_list.append(seq_in_layer)
                    seq = seq + 1

            # concatenate all sequences input - use concatenate for > 1 sequences
            if len(seq_in_list) > 1:
                concat_layer = tf.keras.layers.concatenate(seq_in_list, axis=-1)

                # hidden layer
                for stack in range(len(self._topology_details.get("join_hidden_nodes"))):

                    # first hidden layer
                    if stack == 0:
                        hidden_layer = tf.keras.layers.Dense(
                            units=self._topology_details.get("join_hidden_nodes")[stack],
                            activation=self._topology_details.get("join_hidden_func_nodes")[stack],
                        )(concat_layer)
                    else:
                        hidden_layer = tf.keras.layers.Dense(
                            units=self._topology_details.get("join_hidden_nodes")[stack],
                            activation=self._topology_details.get("join_hidden_func_nodes")[stack],
                        )(hidden_layer)

                # output layer
                if len(self._topology_details.get("join_hidden_nodes")) == 0:
                    # using sigmoid for binary classifier and softmax for multi category
                    if (
                        self._application == "classification"
                        and self._application_type == "binary_category"
                    ):
                        output_layer = tf.keras.layers.Dense(
                            units=outputs[0].shape[1], activation="sigmoid"
                        )(concat_layer)
                    elif (
                        self._application == "classification"
                        and self._application_type == "multi_category_unilabel"
                    ):
                        output_layer = tf.keras.layers.Dense(
                            units=outputs[0].shape[1], activation="softmax"
                        )(concat_layer)
                    else:
                        logging.error(
                            "Application and/or application type is not valid for topolgy "
                            + self._topology_id
                        )
                else:
                    # using sigmoid for binary classifier and softmax for multi category
                    if (
                            self._application == "classification"
                            and self._application_type == "binary_category"
                    ):
                        output_layer = tf.keras.layers.Dense(
                            units=outputs[0].shape[1], activation="sigmoid"
                        )(hidden_layer)
                    elif (
                            self._application == "classification"
                            and self._application_type == "multi_category_unilabel"
                    ):
                        output_layer = tf.keras.layers.Dense(
                            units=outputs[0].shape[1], activation="softmax"
                        )(hidden_layer)
                    else:
                        logging.error(
                            "Application and/or application type is not valid for topolgy "
                            + self._topology_id)

            # just 1 sequence input
            else:
                # hidden layer
                for stack in range(len(self._topology_details.get("join_hidden_nodes"))):

                    # first hidden layer
                    if stack == 0:
                        hidden_layer = tf.keras.layers.Dense(
                            units=self._topology_details.get("join_hidden_nodes")[stack],
                            activation=self._topology_details.get("join_hidden_func_nodes")[stack],
                        )(seq_in_list[0])
                    else:
                        hidden_layer = tf.keras.layers.Dense(
                            units=self._topology_details.get("join_hidden_nodes")[stack],
                            activation=self._topology_details.get("join_hidden_func_nodes")[stack],
                        )(hidden_layer)

                # output layer model
                if len(self._topology_details.get("join_hidden_nodes")) == 0:
                    # using sigmoid for binary classifier and softmax for multi category
                    if (
                            self._application == "classification"
                            and self._application_type == "binary_category"
                    ):
                        output_layer = tf.keras.layers.Dense(
                            units=outputs[0].shape[1], activation="sigmoid"
                        )(seq_in_list[0])
                    elif (
                            self._application == "classification"
                            and self._application_type == "multi_category_unilabel"
                    ):
                        output_layer = tf.keras.layers.Dense(
                            units=outputs[0].shape[1], activation="softmax"
                        )(seq_in_list[0])
                    else:
                        logging.error(
                            "Application and/or application type is not valid for topolgy "
                            + self._topology_id
                        )
                else:
                    # using sigmoid for binary classifier and softmax for multi category
                    if (
                            self._application == "classification"
                            and self._application_type == "binary_category"
                    ):
                        output_layer = tf.keras.layers.Dense(
                            units=outputs[0].shape[1], activation="sigmoid"
                        )(hidden_layer)
                    elif (
                            self._application == "classification"
                            and self._application_type == "multi_category_unilabel"
                    ):
                        output_layer = tf.keras.layers.Dense(
                            units=outputs[0].shape[1], activation="softmax"
                        )(hidden_layer)
                    else:
                        logging.error(
                            "Application and/or application type is not valid for topolgy "
                            + self._topology_id)

            self._model = tf.keras.models.Model(inputs=input_list, outputs=output_layer)

        else:
            raise ValueError("This topology_id is not valid")

        # Set callbacks for training

        # Enable tensorboard tracking
        if self._tracking:
            logdir = "tbruns/logs/" + self._prefix_name + "log"
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir,
                                                                  histogram_freq=1,
                                                                  write_graph=True)
            self._callbacks.append(tensorboard_callback)

        if self._model_parameters.reduce_lr:
            # Learning rate adjustment after patient epochs on constant val_loss (factor * original) return to original
            # value after cooldown epochs
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.2,
                patience=3,
                min_lr=0.001,
                verbose=1,
                mode="auto",
                cooldown=10,
            )
            self._callbacks.append(reduce_lr)

        if self._model_parameters.early_stopping:
            # Early stopping with validation perfomance (save best model)
            earlystopping = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=2, verbose=1, mode="min", restore_best_weights=True
            )
            self._callbacks.append(earlystopping)

        # Checkpoint models during training
        if self._model_parameters.save_checkpoints:
            ckp_model_file = (
                self._run_folder_path
                + self._prefix_name
                + "ep_{epoch:02d}-val-loss_{val_loss:.2f}.h5"
            )
            checkpointer = tf.keras.callbacks.ModelCheckpoint(
                filepath=ckp_model_file,
                verbose=1,
                save_best_only=True,
                monitor="val_loss",
                mode="min",
                save_weights_only=False,
            )

            self._callbacks.append(checkpointer)

        # Model compile
        self._model.compile(
            loss=self._model_parameters.loss_optim.value,
            optimizer=self._model_parameters.optimizer.value,
            metrics=self._model_parameters.metrics,
        )

        # Print model resume
        self._model.summary()

        return True

    def save_model(self, working_folder=None, prefix=None, format="h5"):

        # Filename
        model_filename_path = working_folder
        model_filename = prefix + "best_"

        # Topology without weights in JSON
        if format == "json":
            model_json = self._model.to_json()
            with open(
                os.path.join(model_filename_path, model_filename + "model.json"), "w"
            ) as json_file:
                json_file.write(model_json)
            return os.path.join(model_filename_path, model_filename + "model.json")

        # Topology without weights in YAML
        elif format == "yaml":
            model_yaml = self._model.to_yaml()
            with open(
                os.path.join(model_filename_path, model_filename + "model.yaml"), "w"
            ) as yaml_file:
                yaml_file.write(model_yaml)
            return os.path.join(model_filename_path, model_filename + "model.yaml")

        # Topology with weights in HDF5
        elif format == "h5":
            self._model.save(os.path.join(model_filename_path, model_filename + "model.h5"))
            return os.path.join(model_filename_path, model_filename + "model.h5")
