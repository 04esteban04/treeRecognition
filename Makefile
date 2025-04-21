PYTHON=python
PYTEST=pytest
SRC_DIR=src
TESTS_DIR=tests
ASSETS_DIR=assets

.PHONY: test clean

test:
	@echo "Corriendo tests..."
	PYTHONPATH=$(SRC_DIR) $(PYTEST) $(TESTS_DIR)

clean:
	@echo "Limpiando archivos temporales..."
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -name "*.pyc" -exec rm -f {} +
