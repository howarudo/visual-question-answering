import pandas as pd
import torch
import re
from src.params import PROMPT, MAX_LENGTH

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
