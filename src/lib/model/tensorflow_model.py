"""
===========================================================================================
Tensor Flow Model Building Package (version:
===========================================================================================
Script Reviewed by COGNAS
===========================================================================================
"""

import os
from tensorflow import keras as k


class XTensorFlowModel:
    def __init__(self, param=None):
        """Constructor for this class"""
        self._model_parameters = param
        self._callbacks = []
        self._model = None
        self._history = None
        self._statefull = False
        self._topology_id = None
        self._run_folder_path = None
        self._prefix_name = None

    def set_topology_id(self, topology: str) -> bool:
        self._topology_id = topology
        return True

    def set_run_folder_path(self, path: str) -> bool:
        self._run_folder_path = path
        return True

    def set_prefix_name(self, prefix: str) -> bool:
        self._prefix_name = prefix
        return True

    def fit(self, X, Y):

        # Build architeture
        self.set_architeture(input_shape=X.shape, output_shape=Y.shape)

        # Go to training
        # if self._statefull:
        #     for i in range(self._model_parameters.get("epochs")):
        #         self._history = self._model.fit(
        #             X,
        #             Y,
        #             batch_size=self._model_parameters.get("batch_size"),
        #             epochs=1,
        #             validation_split=self._model_parameters.get("validation_split"),
        #             callbacks=self._callbacks,
        #             verbose=self._model_parameters.get("verbose"),
        #             shuffle=False,
        #         )
        #         self._model.reset_states()
        # else:
        self._history = self._model.fit(
                X,
                Y,
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

    def set_architeture(self, input_shape=None, output_shape=None) -> bool:

        # Check pre-models
        if self._topology_id == "DNN-DENSE":

            self._model = k.Sequential()
            teste = len(self._model_parameters.hidden_nodes)
            for i in range(len(self._model_parameters.hidden_nodes)):
                if i == 0:
                    self._model.add(
                        k.layers.Dense(units = self._model_parameters.hidden_nodes[0],
                                       use_bias=True,
                                       bias_initializer='zeros',
                                       kernel_initializer='glorot_uniform',
                                       input_dim=input_shape[1],
                                       activation=self._model_parameters.hidden_func_nodes[0].value)
                    )
                    self._model.add(k.layers.Dropout(self._model_parameters.hidden_dropout[0]))
                else:
                    self._model.add(k.layers.Dense(units = self._model_parameters.hidden_nodes[i],
                                                   activation=self._model_parameters.hidden_func_nodes[i].value))
                    self._model.add(k.layers.Dropout(self._model_parameters.hidden_dropout[i]))

            # last layer
            self._model.add(k.layers.Dense(units = output_shape[1], activation=self._model_parameters.output_func_nodes.value))

        # elif self._model_parameters.topology_id == "DNN-DENSE-DENSE-DENSE-DENSE":
        #
        #     nodes = self._model_parameters.nodes
        #     func = self._model_parameters.func
        #     drop = self._model_parameters.dropout
        #
        #     self._model = Sequential()
        #     self._model.add(Dense(nodes[0], input_shape=(input_shape[1],), activation=func[0]))
        #     self._model.add(Dropout(drop))
        #     self._model.add(Dense(nodes[1], activation=func[1]))
        #     self._model.add(Dropout(drop))
        #     self._model.add(Dense(nodes[2], activation=func[2]))
        #     self._model.add(Dropout(drop))
        #     self._model.add(Dense(output_shape[1], activation=func[3]))
        #
        # # elif self.model_parameters.topology_id == "RNN-LSTM-DENSE":
        # #     # Vanilla LSTM with stateless between batches
        # #     # input shape: samples x time_steps x features
        # #
        # #     nodes = self.model_parameters.nodes
        # #     func = self.model_parameters.func
        # #     drop = self.model_parameters.dropout
        # #
        # #     self.model = Sequential()
        # #     self.model.add(LSTM(nodes[0],
        # #                         input_shape=(input_shape[1], input_shape[2]),
        # #                         dropout=drop,
        # #                         recurrent_dropout=drop,
        # #                         activation=func[0],
        # #                         stateful=False))
        # #     self.model.add(Dense(output_shape[1], activation=func[1]))
        #
        # elif self._model_parameters.topology_id == "RNN-EMB-FIX-LSTM-DENSE":
        #     # ref: https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
        #
        #     nodes = self._model_parameters.nodes
        #     func = self._model_parameters.func
        #     drop = self._model_parameters.dropout
        #     output_classes = len(set(output))
        #
        #     # loading model
        #     embedding_model = Word2Vec()
        #     embedding_model.load_model(self._model_parameters.embedding_word_model_file)
        #
        #     # matrix from w2v model
        #     embedding_matrix = embedding_model.get_embedding_matrix()
        #
        #     self._model = tf.keras.models.Sequential()
        #     self._model.add(
        #         tf.keras.layers.Embedding(
        #             embedding_matrix.shape[0],
        #             embedding_matrix.shape[1],
        #             weights=[embedding_matrix],
        #             input_length=dataset_param.input_txt_max_seq,
        #             trainable=False,
        #         )
        #     )
        #     self._model.add(
        #         tf.keras.layers.LSTM(nodes[0], dropout=drop[0], activation=func[0], stateful=False)
        #     )
        #     self._model.add(tf.keras.layers.Dense(output_classes, activation=func[1]))
        #
        # elif self._model_parameters.topology_id == "RNN-EMB-FIX-LSTM-DENSE-ATTENTION":
        #     # ref: https://www.analyticsvidhya.com/blog/2019/11/comprehensive-guide-attention-mechanism-deep-learning/
        #
        #     nodes = self._model_parameters.nodes
        #     func = self._model_parameters.func
        #     drop = self._model_parameters.dropout
        #
        #     # loading model
        #     embedding_model = Word2Vec()
        #     embedding_model.load_model(self._model_parameters.embedding_word_model_file)
        #
        #     # dictionary and matrix from w2v model
        #     word_index = {
        #         word: vocab.index for word, vocab in embedding_model._model.wv.vocab.items()
        #     }
        #     embedding_model.word_index = word_index
        #     embedding_matrix = embedding_model.get_embedding_matrix()
        #
        #     inputs = Input(shape=(self._model_parameters.input_txt_max_seq,), dtype="int32")
        #
        #     embedded_sequences = Embedding(
        #         input_dim=embedding_matrix.shape[0],
        #         output_dim=embedding_matrix.shape[1],
        #         input_length=self._model_parameters.input_txt_max_seq,
        #         weights=[embedding_matrix],
        #         trainable=False,
        #     )(inputs)
        #
        #     att_in = LSTM(
        #         nodes[0], return_sequences=True, dropout=drop[0], recurrent_dropout=drop[0]
        #     )(embedded_sequences)
        #     att_out = Attention()(att_in)
        #     outputs = Dense(output_shape[1], activation=func[1], trainable=True)(att_out)
        #     self._model = Model(inputs, outputs)
        #
        #     # summarize layers
        #     # print(model.summary())
        #
        # elif self._model_parameters.topology_id == "RNN-EMB-FIX-LSTM-DENSE-ATTENTION2":
        #     # ref: https://github.com/philipperemy/keras-attention-mechanism
        #
        #     nodes = self._model_parameters.nodes
        #     func = self._model_parameters.func
        #     drop = self._model_parameters.dropout
        #
        #     # loading model
        #     embedding_model = Word2Vec()
        #     embedding_model.load_model(self._model_parameters.embedding_word_model_file)
        #
        #     # dictionary and matrix from w2v model
        #     word_index = {
        #         word: vocab.index for word, vocab in embedding_model._model.wv.vocab.items()
        #     }
        #     embedding_model.word_index = word_index
        #     embedding_matrix = embedding_model.get_embedding_matrix()
        #
        #     inputs = Input(shape=(self._model_parameters.input_txt_max_seq,), dtype="int32")
        #
        #     embedded_sequences = Embedding(
        #         input_dim=embedding_matrix.shape[0],
        #         output_dim=embedding_matrix.shape[1],
        #         input_length=self._model_parameters.input_txt_max_seq,
        #         weights=[embedding_matrix],
        #         trainable=False,
        #     )(inputs)
        #
        #     att_in = LSTM(
        #         nodes[0], return_sequences=True, dropout=drop[0], recurrent_dropout=drop[0]
        #     )(embedded_sequences)
        #     att_out = self.attention_3d_block(att_in)
        #     outputs = Dense(output_shape[1], activation=func[1], trainable=True, name="output")(
        #         att_out
        #     )
        #     self._model = Model(inputs, outputs)
        #
        # elif self._model_parameters.topology_id == "RNN-EMB-FIX-LSTM-DENSE-ATTENTION3":
        #     # ref: https://github.com/thushv89/attention_keras
        #
        #     nodes = self._model_parameters.nodes
        #     func = self._model_parameters.func
        #     drop = self._model_parameters.dropout
        #
        #     # loading model
        #     embedding_model = Word2Vec()
        #     embedding_model.load_model(self._model_parameters.embedding_word_model_file)
        #
        #     # dictionary and matrix from w2v model
        #     word_index = {
        #         word: vocab.index for word, vocab in embedding_model._model.wv.vocab.items()
        #     }
        #     embedding_model.word_index = word_index
        #     embedding_matrix = embedding_model.get_embedding_matrix()
        #
        #     inputs = Input(shape=(self._model_parameters.input_txt_max_seq,), dtype="int32")
        #
        #     embedded_sequences = Embedding(
        #         input_dim=embedding_matrix.shape[0],
        #         output_dim=embedding_matrix.shape[1],
        #         input_length=self._model_parameters.input_txt_max_seq,
        #         weights=[embedding_matrix],
        #         trainable=False,
        #     )(inputs)
        #
        #     encoder = Bidirectional(
        #         LSTM(
        #             nodes[0],
        #             return_state=True,
        #             return_sequences=True,
        #             dropout=drop[0],
        #             recurrent_dropout=drop[0],
        #         )
        #     )
        #     encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(
        #         embedded_sequences
        #     )
        #
        #     state_h = Concatenate(axis=-1)([forward_h, backward_h])
        #     state_c = Concatenate(axis=-1)([forward_c, backward_c])
        #     encoder_states = Concatenate(axis=-1)([state_h, state_c])
        #
        #     decoder_outputs = TimeDistributed(
        #         Dense(nodes[0] * 2, activation=func[1], trainable=True, name="decoder")
        #     )(encoder_outputs)
        #
        #     # Attention layer
        #     attn_layer = AttentionLayer(name="attention_layer")
        #     attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])
        #
        #     # Concat attention output and decoder output
        #     decoder_concat_input = Concatenate(axis=-1, name="concat_layer")(
        #         [decoder_outputs, attn_out]
        #     )
        #
        #     decoder_dense = Dense(output_shape[1], activation="softmax")
        #     outputs = decoder_dense(decoder_concat_input)
        #
        #     # Define the model that will turn
        #     self._model = Model([inputs, decoder_outputs], outputs)
        #
        # elif self._model_parameters.topology_id == "RNN-EMB-TRAIN-LSTM-DENSE":
        #     # ref: https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
        #
        #     nodes = self._model_parameters.nodes
        #     func = self._model_parameters.func
        #     drop = self._model_parameters.dropout
        #
        #     # loading model
        #     embedding_model = Word2Vec()
        #     embedding_model.load_model(self._model_parameters.embedding_word_model_file)
        #
        #     # dictionary and matrix from w2v model
        #     word_index = {
        #         word: vocab.index for word, vocab in embedding_model._model.wv.vocab.items()
        #     }
        #     embedding_model.word_index = word_index
        #     embedding_matrix = embedding_model.get_embedding_matrix()
        #
        #     self._model = Sequential()
        #     self._model.add(
        #         Embedding(
        #             embedding_matrix.shape[0],
        #             embedding_matrix.shape[1],
        #             weights=[embedding_matrix],
        #             input_length=self._model_parameters.input_txt_max_seq,
        #             trainable=True,
        #         )
        #     )
        #     self._model.add(LSTM(nodes[0], dropout=drop[0], activation=func[0], stateful=False))
        #     self._model.add(Dense(output_shape[1], activation=func[1]))
        #
        # elif self._model_parameters.topology_id == "RNN-CONV1D-MAXP1D-LSTM-DENSE":
        #     # todo: criar uma sequencia especifica pra CNN em multisequencias de LSTM
        #     # input shape: samples x time_steps x features
        #
        #     nodes = self._model_parameters.nodes
        #     func = self._model_parameters.func
        #     drop = self._model_parameters.dropout
        #
        #     self._model = Sequential()
        #     self._model.add(
        #         Convolution1D(
        #             input_shape=(input_shape[1], input_shape[2]),
        #             filters=32,
        #             kernel_size=3,
        #             padding="same",
        #             activation="relu",
        #         )
        #     )
        #     self._model.add(MaxPooling1D(pool_size=2))
        #     self._model.add(
        #         LSTM(
        #             nodes[0],
        #             dropout=drop,
        #             recurrent_dropout=drop,
        #             activation=func[0],
        #             stateful=False,
        #         )
        #     )
        #     self._model.add(Dense(output_shape[1], activation=func[1]))
        #
        # elif self._model_parameters.topology_id == "RNN-LSTM-LSTM-DENSE":
        #     # Vanilla LSTM with stateless between batches
        #     # input shape: samples x time_steps x features
        #
        #     nodes = self._model_parameters.nodes
        #     func = self._model_parameters.func
        #     drop = self._model_parameters.get.dropout
        #
        #     self._model = Sequential()
        #     self._model.add(
        #         LSTM(
        #             nodes[0],
        #             input_shape=(input_shape[1], input_shape[2]),
        #             activation=func[0],
        #             dropout=drop,
        #             recurrent_dropout=drop,
        #             stateful=False,
        #             return_sequences=True,
        #         )
        #     )
        #     # self.model.add(Dropout(drop))
        #     self._model.add(
        #         LSTM(
        #             nodes[1],
        #             activation=func[1],
        #             dropout=drop,
        #             recurrent_dropout=drop,
        #             stateful=False,
        #         )
        #     )
        #     self._model.add(Dense(output_shape[1], activation=func[2]))
        #
        # elif self._model_parameters.topology_id == "RNN-LSTM-DENSE-Statefull":
        #     # Vanilla LSTM with statefull between batches (check that shuffle = False)
        #     # input shape: samples x time_steps x features
        #
        #     self._statefull = True
        #     nodes = self._model_parameters.nodes
        #     func = self._model_parameters.func
        #     drop = self._model_parameters.dropout
        #
        #     self._model = Sequential()
        #     self._model.add(
        #         LSTM(
        #             nodes[0],
        #             dropout=drop,
        #             batch_input_shape=(
        #                 self._model_parameters.batch_size,
        #                 input_shape[1],
        #                 input_shape[2],
        #             ),
        #             activation=func[0],
        #             stateful=True,
        #         )
        #     )
        #     self._model.add(Dense(output_shape[1], activation=func[1]))

        else:
            raise ValueError("This topology_id is not valid")

        if self._model_parameters.reduce_lr:
            # Learning rate adjustment after patient epochs on constant val_loss (factor * original) return to original
            # value after cooldown epochs
            reduce_lr = k.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.2,
                patience=3,
                min_lr=0.001,
                verbose=1,
                mode="auto",
                cooldown=10,
            )
            self._callbacks = [reduce_lr]

        if self._model_parameters.early_stopping:
            # Early stopping with validation perfomance (save best model)
            earlystopping = k.callbacks.EarlyStopping(
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
            checkpointer = k.callbacks.ModelCheckpoint(
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
            metrics= self._model_parameters.metrics
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
