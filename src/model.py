from src.utils import *
from src.params import *

import lightning as L
import torch
from functools import partial
from torch.utils.data import DataLoader



class PaliGemmaModelPLModule(L.LightningModule):
    def __init__(self, config, processor, model, train_dataset, val_dataset):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.batch_size = config.get("batch_size")

    def training_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask, pixel_values, labels = batch

        outputs = self.model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                pixel_values=pixel_values,
                                labels=labels)
        loss = outputs.loss

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):

        input_ids, attention_mask, pixel_values, answers = batch
        with torch.no_grad():
            # autoregressively generate token IDs
            generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                        pixel_values=pixel_values, max_new_tokens=MAX_LENGTH)
            # turn them back into text, chopping of the prompt
            # important: we don't skip special tokens here, because we want to see them in the output
            predictions = self.processor.batch_decode(generated_ids[:, input_ids.size(1):], skip_special_tokens=True)

            vizwiz_acc = VQA_criterion(predictions, answers)
            for pred, answer in zip(predictions, answers):
                pred = process_pred(pred)

                if self.config.get("verbose", False):
                    print(f"Prediction: {pred}")
                    print(f"    Answer: {answer}")


            self.log("vizwiz_accuracy", vizwiz_acc)
            print(f"Vizwiz accuracy: {vizwiz_acc}")

        return vizwiz_acc

    def configure_optimizers(self):
        # you could also add a learning rate scheduler if you want
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.get("lr"))

        return optimizer

    def train_dataloader(self):
        return DataLoader(self.train_dataset, collate_fn=partial(train_collate_fn, processor=self.processor), batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, collate_fn=partial(eval_collate_fn, processor=self.processor), batch_size=self.batch_size, shuffle=False, num_workers=4)
