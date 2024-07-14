setup:
	@echo "creating data directory..."
	@mkdir -p data
	@echo "creating submission directory..."
	@mkdir -p submission
	@echo "creating .env file..."
	@cp .env.example .env
	@unzip data/train-001.zip -d data
	@unzip data/valid-002.zip -d data

install_packages:
	@echo "Installing packages..."
	@pip3 install -r requirements.txt

evaluate:
	@echo "Evaluating model..."
	@python -c 'from src.main import eval; eval()'
