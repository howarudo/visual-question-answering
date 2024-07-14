from src.utils import *
from src.params import *
from src.dataset import VQATrainDataset, VQAValDataset, TestDataset
from src.model import PaliGemmaModelPLModule

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from torch.utils.data import DataLoader
from transformers import AutoProcessor
from transformers import PaliGemmaForConditionalGeneration
from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig

import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


def train():

    ### LOAD DATA ###
    log("Loading data")
    train_df = load_df(ANNOTATIONS_TRAIN_PATH)
    val_df = train_df[:100]

    train_dataset = VQATrainDataset(train_df, TRAIN_PATH)
    val_dataset = VQAValDataset(val_df, TRAIN_PATH)
    log(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples")

    ### LOAD MODEL ###
    log("Loading model")
    processor = AutoProcessor.from_pretrained(MODEL_REPO_ID, token=HUGGINGFACE_TOKEN)
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
    )
    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = PaliGemmaForConditionalGeneration.from_pretrained(MODEL_REPO_ID, quantization_config=bnb_config, device_map={"":0}, token=HUGGINGFACE_TOKEN)
    model = get_peft_model(model, lora_config)
    log("Model loaded")
    print(model.print_trainable_parameters())

    ### TRAIN ###
    log("Training model")
    config = {"max_epochs": MAX_TRAIN_EPOCHS,
        # "val_check_interval": 0.2, # how many times we want to validate during an epoch
        "check_val_every_n_epoch": 1,
        "gradient_clip_val": 1.0,
        "accumulate_grad_batches": 16,
        "lr": 1e-4,
        "batch_size": TRAIN_BATCH_SIZE,
        # "seed":2022,
        "num_nodes": 1,
        "warmup_steps": 50,
        "result_path": "./result",
        "verbose": True,
    }
    model_module = PaliGemmaModelPLModule(config, processor, model, train_dataset, val_dataset)
    early_stop_callback = EarlyStopping(monitor="vizwiz_accuracy", patience=10, verbose=False, mode="min")
    torch.set_float32_matmul_precision('high')
    trainer = L.Trainer(
            accelerator="gpu",
            devices=[0],
            max_epochs=config.get("max_epochs"),
            accumulate_grad_batches=config.get("accumulate_grad_batches"),
            check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
            gradient_clip_val=config.get("gradient_clip_val"),
            precision="16-mixed",
            limit_val_batches=5,
            num_sanity_val_steps=0,
            callbacks=[PushToHubCallback(), early_stop_callback],
    )

    trainer.fit(model_module)


def eval():
    """
    Evaluates the model on the test dataset and saves the submission file.
    The model used is specified in EVAL_REPO_ID. (see src/params.py)
    """

    log(f"Evaluating model with ID {EVAL_REPO_ID}")

    ### LOAD DATA ###
    df = pd.read_json(ANNOTATIONS_VAL_PATH)
    df = df[['image', 'question']]
    log(f"Loaded {len(df)} samples")

    ### LOAD MODEL ###
    log("Loading model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PaliGemmaForConditionalGeneration.from_pretrained(EVAL_REPO_ID, token=HUGGINGFACE_TOKEN).to(device)
    model.eval()

    ### PROCESS SAMPLES ###
    log("Processing samples")
    processor = AutoProcessor.from_pretrained(EVAL_REPO_ID, token=HUGGINGFACE_TOKEN)
    def test_collate_fn(batch):
        images, questions = zip(*batch)
        inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True, tokenize_newline_separately=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        return inputs

    data_loader = DataLoader(TestDataset(df), batch_size=TEST_BATCH_SIZE, collate_fn=test_collate_fn)
    torch.cuda.empty_cache()

    ### EVALUATE ###
    log("Processing batches")
    model_answers = []
    with torch.no_grad():
        for inputs in tqdm(data_loader, desc="Processing batches"):
            generated_ids = model.generate(**inputs, max_new_tokens=MAX_LENGTH)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            model_answers.extend([process_pred(ans.split("\n")[1]) for ans in generated_text])
            torch.cuda.empty_cache()

    ### SAVE SUBMISSION ###
    log("Saving submission")
    submission = np.array(model_answers)
    np.save(OUTPUT_PATH + '/submission.npy', submission)
