PYTHON=python
PYTEST=pytest
SRC_DIR=src
TESTS_DIR=app/src/tests
ASSETS_DIR=assets

.PHONY: test clean

test:
	@echo "Corriendo tests..."
	PYTHONPATH=$(SRC_DIR) $(PYTEST) $(TESTS_DIR)

clean:
	@echo "Limpiando archivos temporales..."
	@echo " "
	# Eliminar __pycache__ y archivos .pyc
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -name "*.pyc" -exec rm -f {} +

	# Eliminar carpeta 'output'
	rm -rf app/src/output

	# Eliminar archivos de cache de pytest
	find . -type d -name ".pytest_cache" -exec rm -rf {} + || true

	# Eliminar .pytest_cache en la ruta app/src directamente
	rm -rf app/src/.pytest_cache || true
