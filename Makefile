.PHONY: help install install-dev test lint format clean run

help: ## Показать эту справку
	@echo "Доступные команды:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Установить зависимости
	pip install -r requirements.txt

install-dev: ## Установить зависимости для разработки
	pip install -r requirements.txt
	pip install -e ".[dev]"

test: ## Запустить тесты
	pytest tests/ -v

lint: ## Проверить код линтером
	flake8 src/ tests/
	mypy src/

format: ## Форматировать код
	black src/ tests/

clean: ## Очистить временные файлы
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	rm -rf build/ dist/ .pytest_cache/ .mypy_cache/ .coverage htmlcov/

run: ## Запустить основной скрипт
	cd src && python main.py

