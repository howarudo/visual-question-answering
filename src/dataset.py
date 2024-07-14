from torch.utils.data import Dataset
from PIL import Image
import cv2
from collections import Counter
import numpy as np
from tqdm import tqdm
import Levenshtein as lev
from src.params import *


class VQATrainDataset(Dataset):
    def __init__(self, df, image_dir, max_length=512):
        super().__init__()
        self.max_length = max_length
        self.image_dir = image_dir
        self.df = df
        self.copied_df = None

        self.questions = []
        self.image_paths = []
        self.answers = []

        self.build_answer_vocab()
        self.load_data()

    def load_data(self):
        for index, row in tqdm(self.df.iterrows(), total=len(self.df), leave=True, position=0):
            image_path = self.image_dir + '/' + row['image']
            self.image_paths.append(image_path)
            question = row['question']
            answer = self.copied_df.at[index, 'answer']

            self.questions.append(question)
            self.answers.append(answer)

    def build_answer_vocab(self):
        self.copied_df = self.df.copy()
        self.copied_df.drop(columns=['answers'], inplace=True)
        self.copied_df['answer'] = None

        for index, row in self.df.iterrows():
            intermediate_counter = Counter()
            for answer_map in row['answers']:
                answer = answer_map['answer']
                intermediate_counter.update([answer])

            top_answers = intermediate_counter.most_common(1)
            if len(top_answers) == 1:
                self.copied_df.at[index, 'answer'] = top_answers[0][0]
            else:
                current_min = np.inf
                current_ans = None
                for answer in top_answers:
                    total_distance = 0
                    for answer2 in top_answers:
                        if answer != answer2:
                            lev_distance = lev.distance(answer[0], answer2[0])
                            total_distance += lev_distance
                    if total_distance < current_min:
                        current_min = total_distance
                        current_ans = answer[0]
                self.copied_df.at[index, 'answer'] = current_ans
        return

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert("RGB")
        except OSError:
            cvimg = cv2.imread(image_path)
            image = Image.fromarray(cvimg)
        question = self.questions[idx]
        answers = self.answers[idx]

        return image, question, answers


class VQAValDataset(Dataset):
    def __init__(self, df, image_dir, max_length=512):
        super().__init__()
        self.max_length = max_length
        self.image_dir = image_dir
        self.df = df
        self.questions = []
        self.image_paths = []
        self.answers = []

        self.load_data()

    def load_data(self):
        for index, row in tqdm(self.df.iterrows(), total=len(self.df), leave=True, position=0):
            image_path = self.image_dir + '/' + row['image']
            question = row['question']
            answers = [answer_map['answer'] for answer_map in row['answers']]
            self.image_paths.append(image_path)
            self.questions.append(question)
            self.answers.append(answers)

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert("RGB")
        except OSError:
            cvimg = cv2.imread(image_path)
            image = Image.fromarray(cvimg)
        question = self.questions[idx]
        answers = self.answers[idx]

        return image, question, answers



class TestDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = VALIDATION_PATH + '/' + row['image']
        try:
            image = Image.open(image_path).convert("RGB")
        except OSError:
            cvimg = cv2.imread(image_path)
            image = Image.fromarray(cvimg)
        question = PROMPT + row['question']

        return image, question
