# API Service для ML модели

API сервис для интеграции ML модели с Rust приложением.

## Структура

- `models.py` - Pydantic модели для валидации запросов/ответов
- `data_adapter.py` - Преобразование JSON запроса в графовую структуру для модели
- `response_adapter.py` - Преобразование выхода модели в JSON ответ
- `service.py` - FastAPI приложение с endpoints
- `run_server.py` - Скрипт для запуска сервера

## Запуск

```bash
cd src
python -m api.run_server
```

Или через uvicorn напрямую:

```bash
cd src
uvicorn api.service:app --host 0.0.0.0 --port 8000
```

## Endpoints

### POST /api/v1/analyze

Анализирует одну транзакцию.

**Запрос:**
```json
{
  "transaction_id": "...",
  "transaction_features": {
    "n_inputs": 2,
    "n_outputs": 3,
    ...
  },
  ...
}
```

**Ответ:**
```json
{
  "success": true,
  "transaction_id": "...",
  "prediction": {
    "class": "illicit",
    "confidence": 0.95,
    "risk_score": 0.87
  },
  ...
}
```

### POST /api/v1/batch_analyze

Пакетная обработка транзакций.

### GET /health

Health check endpoint.

## Особенности

1. **Адаптер данных**: Преобразует JSON запрос от Rust в графовую структуру, которую ожидает модель
2. **Адаптер ответа**: Преобразует выход модели (логиты) в JSON формат для Rust
3. **Модель не изменяется**: Вся логика преобразования находится в адаптерах

## Требования

- Модель должна быть обучена и сохранена в `models/aml_bitcoin.pth`
- Конфигурация в `config.yaml` должна быть доступна

