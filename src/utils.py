from src.params import *

import pandas as pd
import torch
import re
from lightning.pytorch.callbacks import Callback
from colorama import Fore, Style

def log(statement):
    print(Fore.GREEN + "\n" + "LOG: " + statement + "\n" + Style.RESET_ALL)

def load_df(df_path):
    df = pd.read_json(df_path)
    df = df[['image', 'question', 'answers']]

    return df


def train_collate_fn(examples_, processor):
    examples = examples_.copy()
    images = [example[0] for example in examples]
    questions = [PROMPT + example[1] for example in examples]
    answers = [example[2] for example in examples]

    inputs = processor(text=questions, images=images, suffix=answers, return_tensors="pt", padding=True,
                        truncation="only_second", max_length=MAX_LENGTH,
                        tokenize_newline_separately=False)

    input_ids = inputs["input_ids"]
    token_type_ids = inputs["token_type_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"]
    labels = inputs["labels"]

    return input_ids, token_type_ids, attention_mask, pixel_values, labels


def eval_collate_fn(examples, processor):
    images = [example[0] for example in examples]
    questions = [PROMPT + example[1] for example in examples]
    answers = [example[2] for example in examples]

    inputs = processor(text=questions, images=images, return_tensors="pt", padding=True, tokenize_newline_separately=False)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"]

    return input_ids, attention_mask, pixel_values, answers


def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
    total_acc = 0.

    for pred, answers in zip(batch_pred, batch_answers):
        total_acc += VQA_eval(pred, answers)

    return total_acc / len(batch_pred)


def VQA_eval(pred, answers):
    acc = 0.
    pred = process_pred(pred)
    for i in range(len(answers)):
        num_match = 0
        for j in range(len(answers)):
            if i == j:
                continue
            if pred == answers[j]:
                num_match += 1
        acc += min(num_match / 3, 1)

    return acc / 10


def process_pred(pred):
    pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
    if "unanswer" in pred:
        pred = "unanswerable"

    return pred


class PushToHubCallback(Callback):
    def __init__(self):
        self.model_id = FINETUNED_MODEL_ID

    def on_train_epoch_end(self, trainer, pl_module):
        print(f"Pushing model to the hub, epoch {trainer.current_epoch}")
        pl_module.model.push_to_hub(
            self.model_id,
            commit_message=f"Training in progress, epoch {trainer.current_epoch}",
            revision=f"epoch_{trainer.current_epoch}"
        )
        pl_module.processor.push_to_hub(
            self.model_id,
            commit_message=f"Training in progress, epoch {trainer.current_epoch}",
            revision=f"epoch_{trainer.current_epoch}"
        )

    def on_train_end(self, trainer, pl_module):
        print("Pushing model to the hub after training")
        # Save and push model
        pl_module.model.push_to_hub(
            self.model_id,
            commit_message="Training done"
        )
        # Save and push processor if it exists
        pl_module.processor.push_to_hub(
            self.model_id,
            commit_message="Training done"
        )
