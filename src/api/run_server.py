"""
Скрипт для запуска API сервиса
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api.service:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Автоперезагрузка при изменении кода
        log_level="info"
    )

