# Tests

Директория для модульных и интеграционных тестов проекта.

## Структура

- `test_models.py` - тесты для моделей нейронных сетей
- `test_utils.py` - тесты для утилит
- `test_loader.py` - тесты для загрузки данных

## Запуск тестов

```bash
pytest tests/
```

С покрытием кода:
```bash
pytest tests/ --cov=src --cov-report=html
```

