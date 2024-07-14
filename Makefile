get_data:
	if [ ! -f ./data/train.zip ]; then gsutil cp gs://dl-common/2024/VQA/train.zip ./data/; fi
	if [ ! -f ./data/valid.zip ]; then gsutil cp gs://dl-common/2024/VQA/valid.zip ./data/; fi
	if [ ! -f ./data/train.json ]; then gsutil cp gs://dl-common/2024/VQA/train.json ./data/; fi
	if [ ! -f ./data/valid.json ]; then gsutil cp gs://dl-common/2024/VQA/valid.json ./data/; fi

setup:
	@echo "creating data directory..."
	@if [ ! -d data ]; then mkdir -p data; fi
	@echo "creating submission directory..."
	@if [ ! -d submission ]; then mkdir -p submission; fi
	@echo "creating .env file..."
	@if [ ! -f .env ]; then cp .env.example .env; fi
	@if [ -f data/train.zip ]; then unzip -n data/train.zip -d data/; fi
	@if [ -f data/valid.zip ]; then unzip -n data/valid.zip -d data/; fi

install_packages:
	@echo "Installing packages..."
	@pip3 install -r requirements.txt

evaluate:
	@echo "Evaluating model..."
	@python -c 'from src.main import eval; eval()'
