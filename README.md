# Visual Question Answering

VizWiz VQA final challenge for Matsuo-Iwasawa lab's Deep Learning Fundamentals course.

## Report
Report can be found [here](report/report.pdf).

The best result is stored in [submission](submission/submitted.npy), which achieved an accuracy of 0.72783 in the test dataset.

## Notebooks
Notebooks can be found [here](notebooks).
These notebooks are meant to run in Google Colab environment and is mainly used in this implementation.

Although only PaliGemma fine-tuning was implemented in `src`, you can find implementation of CLIP models in `notebooks`.

## Implementation
1. Download necessary data

Download `valid.zip`, `train.zip`, `train.json`, and `valid.json` from `dl-common` GCS bucket.
```bash
make get_data
```

2. Unzip data, create directories, and add .env file

Unzipping data download from `1.`, setting `.env` file, and create
```bash
make setup
```

3. Add Hugging face token to .env file

Make sure Hugging Face account has access to [Google's PaliGemma model](https://huggingface.co/google/paligemma-3b-pt-224).

```bash
vim .env
```

4. Install dependencies

Installing packages not pre-installed in machine.
```bash
make install_packages
```

5. Eval best model

This command will evaluate model that was implemented previously and saved in [howarudo's hugging face repo](https://huggingface.co/howarudo/paligemma-3b-pt-224-vqa-continue-ft-0).
```bash
make evaluate
```

6. WIP: Train from `src`
```bash
make train
```
