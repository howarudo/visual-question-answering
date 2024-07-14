install_packages:
	@echo "Installing packages..."
	@pip3 install -r requirements.txt

evaluate:
	@echo "Evaluating model..."
	@python -c 'from src.main import eval; eval()'
