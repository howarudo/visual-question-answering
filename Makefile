get_data:
	@echo "Creating data directory..."
	@if [ ! -d data ]; then mkdir -p data; fi
	if [ ! -f ./data/train.zip ]; then gsutil cp gs://dl-common/2024/VQA/train.zip ./data/; fi
	if [ ! -f ./data/valid.zip ]; then gsutil cp gs://dl-common/2024/VQA/valid.zip ./data/; fi
	if [ ! -f ./data/train.json ]; then gsutil cp gs://dl-common/2024/VQA/train.json ./data/; fi
	if [ ! -f ./data/valid.json ]; then gsutil cp gs://dl-common/2024/VQA/valid.json ./data/; fi

setup:
	@echo "Creating submission directory..."
	@if [ ! -d submission ]; then mkdir -p submission; fi
	@echo "Creating .env file..."
	@if [ ! -f .env ]; then cp .env.example .env; fi
	@echo "Unzipping train.zip..."
	@if [ -f data/train.zip ]; then unzip -qn data/train.zip -d data/; fi
	@echo "Unzipping valid.zip..."
	@if [ -f data/valid.zip ]; then unzip -qn data/valid.zip -d data/; fi
	@echo "Setup complete."

install_packages:
	@echo "Installing packages..."
	@pip3 install -r requirements.txt

evaluate:
	@echo "Evaluating model..."
	@python -c 'from src.main import eval; eval()'

train:
	@echo "Training model..."
	@python -c 'from src.main import train; train()'
