'''
===========================================================================================
Tensor Flow Model Building Package (version:
===========================================================================================
Script Reviewed by COGNAS
===========================================================================================
'''
from lib_xplore.nlp.word2vec import Word2Vec

import os
import tensorflow as tf


class TensorFlowModel:

    def __init__(self, param=None):
        '''Constructor for this class'''
        self.model_parameters = param
        self.callbacks = None
        self.model = None
        self.history = None
        self.statefull = False

    def fit(self, X, Y, dataset_param):

        # Build architeture
        self.set_architeture(input=X, output=Y, dataset_param=dataset_param)

        # Go to training
        if self.statefull:
            for i in range(self.model_parameters.get("epochs")):
                self.history = self.model.fit(X, Y,
                                              batch_size=self.model_parameters.get("batch_size"),
                                              epochs=1,
                                              validation_split=self.model_parameters.get("validation_split"),
                                              callbacks=self.callbacks,
                                              verbose=self.model_parameters.get("verbose"),
                                              shuffle=False)
                self.model.reset_states()
        else:
            self.history = self.model.fit(X, Y,
                                          batch_size=self.model_parameters.batch_size,
                                          epochs=self.model_parameters.epochs,
                                          validation_split=self.model_parameters.validation_split,
                                          callbacks=self.callbacks,
                                          verbose=self.model_parameters.verbose,
                                          shuffle=self.model_parameters.shuffle)

        return True

    def evaluate(self, X):
        if self.statefull:
            Y = self.model.predict(X, batch_size=self.model_parameters.batch_size)
            self.model.reset_states()
        else:
            Y = self.model.predict(X)

        return Y

    def set_architeture(self, input=None, output=None, dataset_param=None):

        # Check pre-models
        if self.model_parameters.topology_id == "DNN-DENSE":
            nodes = self.model_parameters.nodes
            func = self.model_parameters.func
            drop = self.model_parameters.dropout

            self.model = Sequential()
            self.model.add(Dense(nodes[0], input_shape=(input_shape[1],), activation=func[0]))
            self.model.add(Dropout(drop))
            self.model.add(Dense(nodes[1], activation=func[1]))
            self.model.add(Dropout(drop))
            self.model.add(Dense(nodes[2], activation=func[2]))
            self.model.add(Dropout(drop))
            self.model.add(Dense(output_shape[1], activation=func[3]))

        elif self.model_parameters.topology_id == "DNN-DENSE-DENSE-DENSE-DENSE":

            nodes = self.model_parameters.nodes
            func = self.model_parameters.func
            drop = self.model_parameters.dropout

            self.model = Sequential()
            self.model.add(Dense(nodes[0], input_shape=(input_shape[1],), activation=func[0]))
            self.model.add(Dropout(drop))
            self.model.add(Dense(nodes[1], activation=func[1]))
            self.model.add(Dropout(drop))
            self.model.add(Dense(nodes[2], activation=func[2]))
            self.model.add(Dropout(drop))
            self.model.add(Dense(output_shape[1], activation=func[3]))

        # elif self.model_parameters.topology_id == "RNN-LSTM-DENSE":
        #     # Vanilla LSTM with stateless between batches
        #     # input shape: samples x time_steps x features
        #
        #     nodes = self.model_parameters.nodes
        #     func = self.model_parameters.func
        #     drop = self.model_parameters.dropout
        #
        #     self.model = Sequential()
        #     self.model.add(LSTM(nodes[0],
        #                         input_shape=(input_shape[1], input_shape[2]),
        #                         dropout=drop,
        #                         recurrent_dropout=drop,
        #                         activation=func[0],
        #                         stateful=False))
        #     self.model.add(Dense(output_shape[1], activation=func[1]))

        elif self.model_parameters.topology_id == 'RNN-EMB-FIX-LSTM-DENSE':
            # ref: https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

            nodes = self.model_parameters.nodes
            func = self.model_parameters.func
            drop = self.model_parameters.dropout
            output_classes = len(set(output))

            # loading model
            embedding_model = Word2Vec()
            embedding_model.load_model(self.model_parameters.embedding_word_model_file)

            # matrix from w2v model
            embedding_matrix = embedding_model.get_embedding_matrix()

            self.model = tf.keras.models.Sequential()
            self.model.add(tf.keras.layers.Embedding(embedding_matrix.shape[0],
                                                     embedding_matrix.shape[1],
                                                     weights=[embedding_matrix],
                                                     input_length=dataset_param.input_txt_max_seq,
                                                     trainable=False))
            self.model.add(tf.keras.layers.LSTM(nodes[0],
                                                dropout=drop[0],
                                                activation=func[0],
                                                stateful=False))
            self.model.add(tf.keras.layers.Dense(output_classes, activation=func[1]))

        elif self.model_parameters.topology_id == 'RNN-EMB-FIX-LSTM-DENSE-ATTENTION':
            # ref: https://www.analyticsvidhya.com/blog/2019/11/comprehensive-guide-attention-mechanism-deep-learning/

            nodes = self.model_parameters.nodes
            func = self.model_parameters.func
            drop = self.model_parameters.dropout

            # loading model
            embedding_model = Word2Vec()
            embedding_model.load_model(self.model_parameters.embedding_word_model_file)

            # dictionary and matrix from w2v model
            word_index = {word: vocab.index for word, vocab in embedding_model.model.wv.vocab.items()}
            embedding_model.word_index = word_index
            embedding_matrix = embedding_model.get_embedding_matrix()

            inputs = Input(shape=(self.model_parameters.input_txt_max_seq,), dtype='int32')

            embedded_sequences = Embedding(input_dim=embedding_matrix.shape[0],
                                           output_dim=embedding_matrix.shape[1],
                                           input_length=self.model_parameters.input_txt_max_seq,
                                           weights=[embedding_matrix],
                                           trainable=False)(inputs)

            att_in = LSTM(nodes[0], return_sequences=True, dropout=drop[0], recurrent_dropout=drop[0])(
                embedded_sequences)
            att_out = Attention()(att_in)
            outputs = Dense(output_shape[1], activation=func[1], trainable=True)(att_out)
            self.model = Model(inputs, outputs)

            # summarize layers
            # print(model.summary())

        elif self.model_parameters.topology_id == 'RNN-EMB-FIX-LSTM-DENSE-ATTENTION2':
            # ref: https://github.com/philipperemy/keras-attention-mechanism

            nodes = self.model_parameters.nodes
            func = self.model_parameters.func
            drop = self.model_parameters.dropout

            # loading model
            embedding_model = Word2Vec()
            embedding_model.load_model(self.model_parameters.embedding_word_model_file)

            # dictionary and matrix from w2v model
            word_index = {word: vocab.index for word, vocab in embedding_model.model.wv.vocab.items()}
            embedding_model.word_index = word_index
            embedding_matrix = embedding_model.get_embedding_matrix()

            inputs = Input(shape=(self.model_parameters.input_txt_max_seq,), dtype='int32')

            embedded_sequences = Embedding(input_dim=embedding_matrix.shape[0],
                                           output_dim=embedding_matrix.shape[1],
                                           input_length=self.model_parameters.input_txt_max_seq,
                                           weights=[embedding_matrix],
                                           trainable=False)(inputs)

            att_in = LSTM(nodes[0], return_sequences=True, dropout=drop[0], recurrent_dropout=drop[0])(
                embedded_sequences)
            att_out = self.attention_3d_block(att_in)
            outputs = Dense(output_shape[1], activation=func[1], trainable=True, name='output')(att_out)
            self.model = Model(inputs, outputs)

        elif self.model_parameters.topology_id == 'RNN-EMB-FIX-LSTM-DENSE-ATTENTION3':
            # ref: https://github.com/thushv89/attention_keras

            nodes = self.model_parameters.nodes
            func = self.model_parameters.func
            drop = self.model_parameters.dropout

            # loading model
            embedding_model = Word2Vec()
            embedding_model.load_model(self.model_parameters.embedding_word_model_file)

            # dictionary and matrix from w2v model
            word_index = {word: vocab.index for word, vocab in embedding_model.model.wv.vocab.items()}
            embedding_model.word_index = word_index
            embedding_matrix = embedding_model.get_embedding_matrix()

            inputs = Input(shape=(self.model_parameters.input_txt_max_seq,), dtype='int32')

            embedded_sequences = Embedding(input_dim=embedding_matrix.shape[0],
                                           output_dim=embedding_matrix.shape[1],
                                           input_length=self.model_parameters.input_txt_max_seq,
                                           weights=[embedding_matrix],
                                           trainable=False)(inputs)

            encoder = Bidirectional(
                LSTM(nodes[0], return_state=True, return_sequences=True, dropout=drop[0], recurrent_dropout=drop[0]))
            encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(embedded_sequences)

            state_h = Concatenate(axis=-1)([forward_h, backward_h])
            state_c = Concatenate(axis=-1)([forward_c, backward_c])
            encoder_states = Concatenate(axis=-1)([state_h, state_c])

            decoder_outputs = TimeDistributed(Dense(nodes[0] * 2, activation=func[1], trainable=True, name='decoder'))(
                encoder_outputs)

            # Attention layer
            attn_layer = AttentionLayer(name='attention_layer')
            attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

            # Concat attention output and decoder output
            decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

            decoder_dense = Dense(output_shape[1], activation='softmax')
            outputs = decoder_dense(decoder_concat_input)

            # Define the model that will turn
            self.model = Model([inputs, decoder_outputs], outputs)

        elif self.model_parameters.topology_id == 'RNN-EMB-TRAIN-LSTM-DENSE':
            # ref: https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

            nodes = self.model_parameters.nodes
            func = self.model_parameters.func
            drop = self.model_parameters.dropout

            # loading model
            embedding_model = Word2Vec()
            embedding_model.load_model(self.model_parameters.embedding_word_model_file)

            # dictionary and matrix from w2v model
            word_index = {word: vocab.index for word, vocab in embedding_model.model.wv.vocab.items()}
            embedding_model.word_index = word_index
            embedding_matrix = embedding_model.get_embedding_matrix()

            self.model = Sequential()
            self.model.add(Embedding(embedding_matrix.shape[0],
                                     embedding_matrix.shape[1],
                                     weights=[embedding_matrix],
                                     input_length=self.model_parameters.input_txt_max_seq,
                                     trainable=True))
            self.model.add(LSTM(nodes[0],
                                dropout=drop[0],
                                activation=func[0],
                                stateful=False))
            self.model.add(Dense(output_shape[1], activation=func[1]))

        elif self.model_parameters.topology_id == "RNN-CONV1D-MAXP1D-LSTM-DENSE":
            # todo: criar uma sequencia especifica pra CNN em multisequencias de LSTM
            # input shape: samples x time_steps x features

            nodes = self.model_parameters.nodes
            func = self.model_parameters.func
            drop = self.model_parameters.dropout

            self.model = Sequential()
            self.model.add(Convolution1D(input_shape=(input_shape[1], input_shape[2]),
                                         filters=32,
                                         kernel_size=3,
                                         padding='same',
                                         activation='relu'))
            self.model.add(MaxPooling1D(pool_size=2))
            self.model.add(LSTM(nodes[0],
                                dropout=drop,
                                recurrent_dropout=drop,
                                activation=func[0],
                                stateful=False))
            self.model.add(Dense(output_shape[1], activation=func[1]))

        elif self.model_parameters.topology_id == "RNN-LSTM-LSTM-DENSE":
            # Vanilla LSTM with stateless between batches
            # input shape: samples x time_steps x features

            nodes = self.model_parameters.nodes
            func = self.model_parameters.func
            drop = self.model_parameters.get.dropout

            self.model = Sequential()
            self.model.add(LSTM(nodes[0],
                                input_shape=(input_shape[1], input_shape[2]),
                                activation=func[0],
                                dropout=drop,
                                recurrent_dropout=drop,
                                stateful=False,
                                return_sequences=True))
            # self.model.add(Dropout(drop))
            self.model.add(LSTM(nodes[1],
                                activation=func[1],
                                dropout=drop,
                                recurrent_dropout=drop,
                                stateful=False))
            self.model.add(Dense(output_shape[1], activation=func[2]))

        elif self.model_parameters.topology_id == "RNN-LSTM-DENSE-Statefull":
            # Vanilla LSTM with statefull between batches (check that shuffle = False)
            # input shape: samples x time_steps x features

            self.statefull = True
            nodes = self.model_parameters.nodes
            func = self.model_parameters.func
            drop = self.model_parameters.dropout

            self.model = Sequential()
            self.model.add(LSTM(nodes[0], dropout=drop,
                                batch_input_shape=(
                                    self.model_parameters.batch_size, input_shape[1], input_shape[2]),
                                activation=func[0], stateful=True))
            self.model.add(Dense(output_shape[1], activation=func[1]))

        else:
            raise ValueError('This topology_id is not valid')

        # Learning rate adjustment after patient epochs on constant val_loss (factor * original) return to original
        # value after cooldown epochs
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                         factor=0.2,
                                                         patience=3,
                                                         min_lr=0.001,
                                                         verbose=1,
                                                         mode='auto',
                                                         cooldown=10)
        self.callbacks = [reduce_lr]

        # Early stopping with validation perfomance (save best model)
        if self.model_parameters.early_stopping:
            earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                             patience=2,
                                                             verbose=1,
                                                             mode='min',
                                                             restore_best_weights=True)
            self.callbacks.append(earlystopping)

        # Checkpoint models during training
        if self.model_parameters.save_checkpoints:
            ckp_model_file = self.model_parameters.working_folder + \
                             self.model_parameters.prefix_name + "ep_{epoch:02d}-val-loss_{val_loss:.2f}.h5"
            checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=ckp_model_file,
                                                              verbose=1,
                                                              save_best_only=True,
                                                              monitor='val_loss',
                                                              mode='min',
                                                              save_weights_only=False)

            self.callbacks.append(checkpointer)

        # Model compile
        self.model.compile(loss=self.model_parameters.loss,
                           optimizer=self.model_parameters.optimizer,
                           metrics=self.model_parameters.metrics)

        return True

    def save_model(self, working_folder=None, prefix=None, format='h5'):

        # Filename
        model_filename_path = working_folder
        model_filename = prefix + 'best_'

        # Topology without weights in JSON
        if format == 'json':
            model_json = self.model.to_json()
            with open(os.path.join(model_filename_path, model_filename + "model.json"), "w") as json_file:
                json_file.write(model_json)
            return os.path.join(model_filename_path, model_filename + "model.json")

        # Topology without weights in YAML
        elif format == 'yaml':
            model_yaml = self.model.to_yaml()
            with open(os.path.join(model_filename_path, model_filename + "model.yaml"), "w") as yaml_file:
                yaml_file.write(model_yaml)
            return os.path.join(model_filename_path, model_filename + "model.yaml")

        # Topology with weights in HDF5
        elif format == 'h5':
            self.model.save(os.path.join(model_filename_path, model_filename + "model.h5"))
            return os.path.join(model_filename_path, model_filename + "model.h5")
