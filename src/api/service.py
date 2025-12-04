"""
FastAPI сервис для ML модели
"""
import os
import sys
import time
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Добавляем пути для импорта
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
training_path = os.path.join(base_dir, 'training')
src_path = os.path.dirname(__file__)

if training_path not in sys.path:
    sys.path.insert(0, training_path)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from api.models import MLRequest, MLResponse, ErrorResponse
from api.data_adapter import create_graph_from_request
from api.response_adapter import create_response_from_model_output
import utils as u

# Загружаем модель при старте сервиса
MODEL = None
MODEL_ARGS = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_VERSION = "1.0.0"

app = FastAPI(
    title="Bitcoin AML Detection ML Service",
    description="ML сервис для обнаружения отмывания денег в Bitcoin транзакциях",
    version=MODEL_VERSION
)

# CORS middleware для работы с Rust приложением
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене указать конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_model():
    """Загружает обученную модель"""
    global MODEL, MODEL_ARGS
    
    try:
        # Определяем базовую директорию проекта
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        base_dir = os.path.abspath(base_dir)
        
        # Загружаем конфигурацию
        # Устанавливаем правильный путь к config.yaml
        config_path = os.path.join(base_dir, 'config.yaml')
        original_cwd = os.getcwd()
        try:
            os.chdir(base_dir)  # Временно меняем рабочую директорию для get_config()
            MODEL_ARGS = u.get_config()
        finally:
            os.chdir(original_cwd)  # Возвращаем обратно
        
        MODEL_ARGS.use_cuda = (torch.cuda.is_available() and MODEL_ARGS.use_cuda)
        MODEL_ARGS.device = 'cuda' if MODEL_ARGS.use_cuda else 'cpu'
        
        # Путь к модели (абсолютный путь)
        model_path = os.path.join(base_dir, 'models', 'aml_bitcoin.pth')
        model_path = os.path.abspath(model_path)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Определяем количество признаков (из конфигурации или по умолчанию)
        num_features = 94  # Размер признаков из оригинального датасета
        hidden_units = MODEL_ARGS.hidden_units_noAgg if hasattr(MODEL_ARGS, 'hidden_units_noAgg') else 128
        
        # Импортируем модель из training директории
        training_path = os.path.join(base_dir, 'training')
        training_path = os.path.abspath(training_path)
        if training_path not in sys.path:
            sys.path.insert(0, training_path)
        from models import ChebyshevConvolution
        
        # Создаем модель
        MODEL = ChebyshevConvolution(
            MODEL_ARGS,
            [1, 2, 4],
            num_features,
            hidden_units
        ).to(DEVICE)
        
        # Загружаем веса
        checkpoint = torch.load(model_path, map_location=DEVICE)
        MODEL.load_state_dict(checkpoint)
        MODEL.eval()
        
        print(f"Model loaded successfully on {DEVICE}")
        print(f"Model path: {model_path}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Инициализация при старте сервиса"""
    load_model()


@app.get("/")
async def root():
    """Корневой endpoint"""
    return {
        "service": "Bitcoin AML Detection ML Service",
        "version": MODEL_VERSION,
        "status": "running",
        "model_loaded": MODEL is not None
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "device": str(DEVICE)
    }


@app.post("/api/v1/analyze", response_model=MLResponse)
async def analyze_transaction(request: MLRequest):
    """
    Анализирует транзакцию и возвращает предсказание модели.
    
    Args:
        request: Запрос с данными транзакции от Rust приложения
        
    Returns:
        MLResponse: Предсказание модели в формате JSON
    """
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    try:
        start_time = time.time()
        
        # Преобразуем JSON запрос в граф
        graph_data = create_graph_from_request(request)
        graph_data = graph_data.to(DEVICE)
        
        # Запускаем модель
        with torch.no_grad():
            model_output, _ = MODEL((graph_data.x, graph_data.edge_index))
        
        # Вычисляем время инференса
        inference_time_ms = (time.time() - start_time) * 1000
        
        # Преобразуем выход модели в JSON ответ
        response = create_response_from_model_output(
            model_output=model_output,
            transaction_id=request.transaction_id,
            inference_time_ms=inference_time_ms,
            model_version=MODEL_VERSION
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )


@app.post("/api/v1/batch_analyze")
async def batch_analyze(requests: list[MLRequest]):
    """
    Пакетная обработка транзакций.
    
    Args:
        requests: Список запросов с данными транзакций
        
    Returns:
        Список ответов в формате JSON
    """
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    responses = []
    
    for request in requests:
        try:
            response = await analyze_transaction(request)
            responses.append(response)
        except Exception as e:
            # В случае ошибки добавляем ответ с ошибкой
            error_response = ErrorResponse(
                error={
                    "code": "PROCESSING_ERROR",
                    "message": str(e),
                    "transaction_id": request.transaction_id
                }
            )
            responses.append(error_response)
    
    return responses

