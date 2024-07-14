# Visual Question Answering

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
