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
    TF_TEXT,
    TF_LABEL,
    TF_DESCRIPTION,
    TF_QUESTION_ID,
    TF_ANSWERS,
    DISTILBERT,
    BERT,
    DEV,
    TEST,
    TRAIN,
    VALIDATION,
    TF_DIFFICULTY,
    TF_PREDICTED_DIFFICULTY,
)

import numpy as np

from datasets import Dataset, DatasetDict
import evaluate
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from transformers import EarlyStoppingCallback
import torch  # TODO add torch to requirements.txt and setup.py


class TransformerExperiment(BaseExperiment):
    def __init__(
            self,
            dataset_name: str,
            data_dir: str = DATA_DIR,
            output_root_dir: str = OUTPUT_DIR,
            random_seed: Optional[int] = None,
    ):
        super().__init__(dataset_name, data_dir, output_root_dir, random_seed)
        self.dataset = None
        self.tokenizer = None
        self.tokenized_dataset = None
        self.data_collator = None
        self.input_mode = None

    def get_dataset(self, input_mode, *args, **kwargs):
        df_train_original = pd.read_csv(os.path.join(self.data_dir, f'tf_{self.dataset_name}_text_difficulty_train.csv'))
        df_test_original = pd.read_csv(os.path.join(self.data_dir, f'tf_{self.dataset_name}_text_difficulty_test.csv'))
        df_dev_original = pd.read_csv(os.path.join(self.data_dir, f'tf_{self.dataset_name}_text_difficulty_dev.csv'))
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
            df_train_original = pd.merge(df_answers, df_train_original, right_on=TF_QUESTION_ID, left_on=TF_QUESTION_ID)
            df_train_original[TF_DESCRIPTION] = df_train_original[TF_DESCRIPTION] + df_train_original[TF_ANSWERS]
            df_test_original = pd.merge(df_answers, df_test_original, right_on=TF_QUESTION_ID, left_on=TF_QUESTION_ID)
            df_test_original[TF_DESCRIPTION] = df_test_original[TF_DESCRIPTION] + df_test_original[TF_ANSWERS]
            df_dev_original = pd.merge(df_answers, df_dev_original, right_on=TF_QUESTION_ID, left_on=TF_QUESTION_ID)
            df_dev_original[TF_DESCRIPTION] = df_dev_original[TF_DESCRIPTION] + df_dev_original[TF_ANSWERS]

        df_train_original = df_train_original.rename(columns={TF_DESCRIPTION: TF_TEXT, TF_DIFFICULTY: TF_LABEL}).astype({TF_LABEL: float})
        df_test_original = df_test_original.rename(columns={TF_DESCRIPTION: TF_TEXT, TF_DIFFICULTY: TF_LABEL}).astype({TF_LABEL: float})
        df_dev_original = df_dev_original.rename(columns={TF_DESCRIPTION: TF_TEXT, TF_DIFFICULTY: TF_LABEL}).astype({TF_LABEL: float})

        dataset = DatasetDict({
            TRAIN: Dataset.from_pandas(df_train_original[[TF_QUESTION_ID, TF_TEXT, TF_LABEL]]),
            TEST: Dataset.from_pandas(df_test_original[[TF_QUESTION_ID, TF_TEXT, TF_LABEL]]),
            VALIDATION: Dataset.from_pandas(df_dev_original[[TF_QUESTION_ID, TF_TEXT, TF_LABEL]]),
        })
        self.dataset = dataset.remove_columns(["__index_level_0__"])

    def init_model(
            self,
            pretrained_model: Optional[str] = DISTILBERT,
            model_name: str = 'model',
            max_length: int = 256,
            pretrained_tokenizer: Optional = None,
            *args, **kwargs,
    ):
        self.model_name = model_name
        if pretrained_tokenizer is None:
            pretrained_tokenizer = pretrained_model
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer)
        
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_model, num_labels=1)  # default loss for regression (num_labels=1) is MSELoss()
        # TODO possibly move the two lines below somewhere else
        self.tokenized_dataset = self.dataset.map(self._preprocess_function, batched=True, max_length=max_length, padding=True)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def train(
            self,
            epochs: int = 10,
            batch_size: int = 16,
            early_stopping_patience: int = 5,
            learning_rate: float = 2e-5,
            weight_decay: float = 0.01,
            *args, **kwargs,
    ):
        training_args = TrainingArguments(
            output_dir=os.path.join(self.output_dir, f'{self.model_name}_{self.input_mode}'),  # TODO check this
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=weight_decay,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            # push_to_hub=False,
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset[TRAIN],
            eval_dataset=self.tokenized_dataset[VALIDATION],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
        )
        trainer.train()
        # TODO possibly change the names below so that they use the training args as well (e.g. learning rate, etc.).
        self.tokenizer.save_pretrained(os.path.join(self.output_dir, f'{self.model_name}_{self.input_mode}_tokenizer'))
        trainer.save_model(os.path.join(self.output_dir, f'{self.model_name}_{self.input_mode}_model'))

    def predict(self, save_predictions: bool = True):
        # TODO
        inputs = tokenizer(text, return_tensors="pt")  # todo this is tmp, text is a single text/question
        with torch.no_grad():  # todo check if I can remove this
            logits = model(**inputs).logits

        # TODO , this is from R2DE
        # self.y_pred_train = self.model.predict(self.x_train)
        # self.y_pred_test = self.model.predict(self.x_test)
        # if save_predictions:
        #     pickle.dump(self.y_pred_test, open(os.path.join(self.output_dir, f'predictions_test_r2de_encoding_{self.encoding_idx}.p'), 'wb'))
        #     pickle.dump(self.y_pred_train, open(os.path.join(self.output_dir, f'predictions_train_r2de_encoding_{self.encoding_idx}.p'), 'wb'))

    def _preprocess_function(self, examples):
        return self.tokenizer(examples[TF_TEXT], truncation=True)


# TODO clean this method below
def compute_metrics(eval_pred):
    r_squared = evaluate.load("r_squared")
    mse = evaluate.load("mse")
    mae = evaluate.load("mae")
    pearsonr = evaluate.load("pearsonr")

    predictions, labels = eval_pred
    return {"r_squared": r_squared.compute(predictions=predictions, references=labels)}
