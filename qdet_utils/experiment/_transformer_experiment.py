from typing import Optional, Dict, List
import os
import pandas as pd
import pickle

from qdet_utils.experiment import BaseExperiment
from qdet_utils.constants import (
    OUTPUT_DIR,
    DATA_DIR,
    AM,
    TF_Q_ONLY,
    TF_Q_CORRECT,
    TF_CORRECT,
    TF_ANS_ID,
    TF_DESCRIPTION,
    TF_QUESTION_ID,
    TF_ANSWERS,
    DISTILBERT,
    BERT,
    DEV,
    TEST,
    TRAIN,
    TF_DIFFICULTY,
    TF_PREDICTED_DIFFICULTY,
)


import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, TFAutoModel, TFAutoModelWithLMHead
import shutil


class TransformerExperiment(BaseExperiment):
    def __init__(
            self,
            dataset_name: str,
            data_dir: str = DATA_DIR,
            output_root_dir: str = OUTPUT_DIR,
            random_seed: Optional[int] = None,
            tpu: bool = False,
    ):
        super().__init__(dataset_name, data_dir, output_root_dir, random_seed)

        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)
        os.environ['TF_DETERMINISTIC_OPS'] = str(self.random_seed)

        self.df_train_original = None
        self.df_test_original = None
        self.df_dev_original = None
        self.input_mode = None
        self.tpu = tpu
        self.batch_size = None
        self.max_length = None
        self.patience = None
        self.dropout_final = None
        self.dropout_internal = None
        self.model_to_finetune = None
        self.hf_model_name = None
        self.autotune = None
        self.learning_rate = None
        self.train_data_original = None
        self.train_data = None
        self.test_data = None
        self.dev_data = None
        self.callback = None
        self.train_steps = None

    def get_dataset(self, input_mode, *args, **kwargs):
        self.df_train_original = pd.read_csv(os.path.join(self.data_dir, f'tf_{self.dataset_name}_text_difficulty_train.csv'))
        self.df_test_original = pd.read_csv(os.path.join(self.data_dir, f'tf_{self.dataset_name}_text_difficulty_test.csv'))
        self.df_dev_original = pd.read_csv(os.path.join(self.data_dir, f'ttf_{self.dataset_name}_text_difficulty_dev.csv'))
        self.input_mode = input_mode

        # Update the texts depending on the INPUT_MODE
        if input_mode != TF_Q_ONLY:
            if self.dataset_name == AM:
                raise ValueError()  # AM cannot get in here
            # load answers to integrate the stem
            df_answers = pd.read_csv(os.path.join(self.data_dir, f'tf_{self.dataset_name}_answers_texts.csv'))
            if input_mode == TF_Q_CORRECT:
                df_answers = df_answers[df_answers[TF_CORRECT] == True]
            answers_dict = dict()
            for q_id, text in df_answers[[TF_QUESTION_ID, TF_DESCRIPTION]].values:
                if q_id not in answers_dict.keys():
                    answers_dict[q_id] = ''
                answers_dict[q_id] = f'{answers_dict[q_id]} [SEP] {text}'
            df_answers = pd.DataFrame(answers_dict.items(), columns=[TF_QUESTION_ID, TF_ANSWERS])

            self.df_train_original = pd.merge(df_answers, self.df_train_original, right_on=TF_QUESTION_ID, left_on=TF_QUESTION_ID)
            self.df_train_original[TF_DESCRIPTION] = self.df_train_original[TF_DESCRIPTION] + self.df_train_original[TF_ANSWERS]
            self.df_test_original = pd.merge(df_answers, self.df_test_original, right_on=TF_QUESTION_ID, left_on=TF_QUESTION_ID)
            self.df_test_original[TF_DESCRIPTION] = self.df_test_original[TF_DESCRIPTION] + self.df_test_original[TF_ANSWERS]
            self.df_dev_original = pd.merge(df_answers, self.df_dev_original, right_on=TF_QUESTION_ID, left_on=TF_QUESTION_ID)
            self.df_dev_original[TF_DESCRIPTION] = self.df_dev_original[TF_DESCRIPTION] + self.df_dev_original[TF_ANSWERS]

    def init_model(
            self,
            pretrained_model: Optional = None,
            model_name: str = 'model',
            input_mode: int = None,
            batch_size: int = 16,
            max_length: int = 256,
            patience: int = 5,
            learning_rate: float = 2e-5,
            dropout_final: float = 0.3,
            dropout_internal: float = 0.3,
            model_to_finetune: str = DISTILBERT,
            *args, **kwargs,
    ):
        self.batch_size = batch_size
        self.max_length = max_length
        self.patience = patience
        self.learning_rate = learning_rate
        self.dropout_final = dropout_final
        self.dropout_internal = dropout_internal
        self.model_to_finetune = model_to_finetune
        self.hf_model_name = self.map_model_name_to_hf_name(self.model_to_finetune)

        self.autotune = tf.data.experimental.AUTOTUNE
        if self.tpu:
            # Create strategy from tpu
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.TPUStrategy(tpu)
            self.batch_size = self.batch_size * strategy.num_replicas_in_sync

        # define the tokenizer specific for the model
        tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)
        # TODO: probably for this I could remove the is_train=True (as it is the thing that causes the inference not to work on all the samples).
        self.train_data_original = self.perform_encoding_and_get_tf_dataset(self.df_train_original, tokenizer, self.max_length, is_train=True)
        self.df_train = self.df_train_original.sample(frac=1.0)
        self.train_data = self.perform_encoding_and_get_tf_dataset(self.df_train, tokenizer, self.max_length, is_train=True)
        self.test_data = self.perform_encoding_and_get_tf_dataset(self.df_test_original, tokenizer, self.max_length)
        self.dev_data = self.perform_encoding_and_get_tf_dataset(self.df_dev_original, tokenizer, self.max_length)
        if pretrained_model:
            # This is to use if the model is already defined and I don't have to train it
            raise NotImplementedError()
        else:
            # callback for early stopping
            self.callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=self.patience, restore_best_weights=True)
            # set train_steps in order to go through all the train ds
            self.train_steps = len(self.df_train) // self.batch_size

            if self.tpu:
                with strategy.scope():
                    self.model = self.create_model(model_name=self.hf_model_name, learning_rate=self.learning_rate, dropout_intern=self.dropout_internal, dropout_final=self.dropout_final, max_length=self.max_length)
            else:
                self.model = self.create_model(model_name=self.hf_model_name, learning_rate=self.learning_rate, dropout_intern=self.dropout_internal, dropout_final=self.dropout_final, max_length=self.max_length)
            # self.model.summary()

    def train(
            self,
            epochs: int = 10,
            dict_params: Dict[str, List[Dict[str, List[float]]]] = None,
            n_iter: int = 10,
            n_jobs: int = None,
            cv: int = 5,
            *args, **kwargs,
    ):
        history = self.model.fit(self.train_data, epochs=epochs, validation_data=self.dev_data, steps_per_epoch=self.train_steps, callbacks=[self.callback])
        # TODO save model (below the example for the R2DE model)
        # pickle.dump(self.model, open(os.path.join(self.output_dir, f'model_r2de_encoding_{self.encoding_idx}.p'), 'wb'))

    def predict(self, save_predictions: bool = True):
        self.store_model_predictions(self.train_data_original, self.df_train_original, TRAIN, self.train_steps, f'{self.model_name}_{self.input_mode}_{self.random_seed}')
        self.store_model_predictions(self.test_data, self.df_test_original, TEST, f'{self.model_name}_{self.input_mode}_{self.random_seed}')
        self.store_model_predictions(self.dev_data, self.df_dev_original, DEV, f'{self.model_name}_{self.input_mode}_{self.random_seed}')
        # TODO , this is from R2DE
        # self.y_pred_train = self.model.predict(self.x_train)
        # self.y_pred_test = self.model.predict(self.x_test)
        # if save_predictions:
        #     pickle.dump(self.y_pred_test, open(os.path.join(self.output_dir, f'predictions_test_r2de_encoding_{self.encoding_idx}.p'), 'wb'))
        #     pickle.dump(self.y_pred_train, open(os.path.join(self.output_dir, f'predictions_train_r2de_encoding_{self.encoding_idx}.p'), 'wb'))

# # # # # # # # # # Below the methods to clean

    def create_model(self, model_name, max_length=128, dropout_intern=0.5, dropout_final=0.5, learning_rate=2e-5):
        """
        It creates the model composed by transformer, specified by model name, + top layers to do regression.
        @param dropout_final: the dropout of the fully connected layer on the top of the transformer
        @param dropout_intern the dropout for fully connected layers in the transformer
        @param max_length: max sequence length
        @param model_name: the name of the huggingface model
        @return: tf.keras model
        """
        transformer_model = TFAutoModel.from_pretrained(model_name)
        # specify internal dropout
        transformer_model.config.dropout = dropout_intern
        input_ids = tf.keras.layers.Input(shape=(max_length,), name='input_ids', dtype='int32')
        embedding_layer = transformer_model([input_ids])[0]
        # take only the embeddings of [CLS]
        cls_token = embedding_layer[:, 0, :]
        # set final dropout
        dropout = tf.keras.layers.Dropout(dropout_final)(cls_token)
        output = tf.keras.layers.Dense(1)(dropout)
        model = tf.keras.Model(inputs=[input_ids], outputs=output)
        # set optimizer
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        # set loss
        loss = tf.keras.losses.MeanSquaredError()
        # set metrics (used for evaluation, not looked at while training)
        metric1 = tf.keras.metrics.MeanAbsoluteError()
        metric2 = tf.keras.metrics.RootMeanSquaredError()
        # compile the model and return it
        model.compile(optimizer=optimizer, loss=loss, metrics=[metric1, metric2])
        return model

    def encode(self, samples, tokenizer, max_length=256):
        """
        It encodes the textual information of the questions using the tokenizer
        @param samples: list of tuple tuple (textual information, target difficulty)
        @param tokenizer: the tokenizer
        @param max_length: max sequence length
        @return:
                encoded_text: list of encoded text x
                target_difficulty: list of target difficulty y
        """
        encoded_text = []
        target_difficulty = []
        for t in samples:
            text, target = t
            input_dict = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=max_length,
                return_token_type_ids=True,
                return_attention_mask=True,
                pad_to_max_length=True,
                truncation='longest_first'
            )
            encoded_text.append(input_dict['input_ids'])
            target_difficulty.append(target)
        return encoded_text, target_difficulty

    def store_model_predictions(self, data, original_input_df, split, train_steps=None, output_root_filename='mymodel'):
        if split == TRAIN:
            predictions = self.model.predict(data, steps=train_steps)
        else:
            predictions = self.model.predict(data)
        out_df = original_input_df[[TF_QUESTION_ID, TF_DIFFICULTY]].copy()
        out_df[TF_PREDICTED_DIFFICULTY] = pd.DataFrame(predictions.flatten())
        # TODO for saving the models, I have to check that the format is the same as for the other models (NON TF).
        out_df.to_csv(os.path.join(self.data_dir, self.dataset_name, f'predictions_{split}_{output_root_filename}.csv'), index=False)

    def perform_encoding_and_get_tf_dataset(self, input_df, tokenizer, max_length, is_train=False):
        samples = list(zip(input_df[TF_DESCRIPTION], input_df[TF_DIFFICULTY]))
        # encode the data
        xs, ys = self.encode(samples, tokenizer, max_length=max_length)
        # transform the dataset into a tf.dataset
        if is_train:
            dataset = tf.data.Dataset.from_tensor_slices((xs, ys)).repeat(-1).batch(self.batch_size).prefetch(self.autotune)
        else:
            dataset = tf.data.Dataset.from_tensor_slices((xs, ys)).batch(self.batch_size).prefetch(self.autotune)
        return dataset

    def map_model_name_to_hf_name(self, model_name):
        if model_name == BERT:
            return 'bert-base-uncased'
        if model_name == DISTILBERT:
            return 'distilbert-base-uncased'
