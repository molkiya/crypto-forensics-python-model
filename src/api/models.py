"""
Pydantic models for API request/response validation
"""
from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class TransactionFeatures(BaseModel):
    """Базовые характеристики транзакции"""
    n_inputs: int = Field(..., description="Число входов транзакции")
    n_outputs: int = Field(..., description="Число выходов транзакции")
    input_value_sum: float = Field(..., description="Сумма BTC на входах")
    output_value_sum: float = Field(..., description="Сумма BTC на выходах")
    transaction_fee: float = Field(..., description="Комиссия транзакции")
    avg_input_value: float = Field(..., description="Средний размер входящего UTXO")
    avg_output_value: float = Field(..., description="Средний размер создаваемого UTXO")


class ExtendedFeatures(BaseModel):
    """Расширенные характеристики транзакции"""
    time_step: Optional[int] = Field(None, description="Порядковый номер временного шага")
    avg_input_incoming_txs: Optional[float] = Field(None, description="Среднее число входящих транзакций на адресах входов")
    avg_output_outgoing_txs: Optional[float] = Field(None, description="Среднее число исходящих транзакций с адресов выходов")
    unique_input_addresses: Optional[int] = Field(None, description="Число уникальных адресов среди входов")
    unique_output_addresses: Optional[int] = Field(None, description="Число уникальных адресов среди выходов")
    num_coinbase_inputs: Optional[int] = Field(None, description="Флаг coinbase")
    old_input_fraction: Optional[float] = Field(None, description="Доля старых входов")
    change_output_ratio: Optional[float] = Field(None, description="Соотношение суммы сдачи")
    inputs_address_entropy: Optional[float] = Field(None, description="Энтропия адресов входов")
    outputs_address_entropy: Optional[float] = Field(None, description="Энтропия адресов выходов")
    spent_outputs_count: Optional[int] = Field(None, description="Число потраченных выходов")
    unspent_outputs_count: Optional[int] = Field(None, description="Число не потраченных выходов")
    time_diff_prev_output: Optional[float] = Field(None, description="Среднее время жизни использованных входов")
    avg_outgoing_txs_inputs: Optional[float] = Field(None, description="Среднее число транзакций расходующих адреса входов")
    avg_incoming_txs_outputs: Optional[float] = Field(None, description="Среднее число транзакций получающих адреса выходов")


class Addresses(BaseModel):
    """Адреса входов и выходов"""
    inputs: List[str] = Field(default_factory=list, description="Список адресов входов")
    outputs: List[str] = Field(default_factory=list, description="Список адресов выходов")


class AddressFeatures(BaseModel):
    """Признаки адреса"""
    address: str = Field(..., description="Bitcoin адрес")
    features: List[float] = Field(..., description="Массив признаков адреса (55 признаков)")


class MLRequest(BaseModel):
    """Запрос от Rust приложения"""
    transaction_id: str = Field(..., description="ID транзакции")
    transaction_features: TransactionFeatures = Field(..., description="Базовые характеристики транзакции")
    extended_features: Optional[ExtendedFeatures] = Field(None, description="Расширенные характеристики")
    addresses: Optional[Addresses] = Field(None, description="Адреса входов и выходов")
    input_features: Optional[List[AddressFeatures]] = Field(default_factory=list, description="Признаки входных адресов")
    output_features: Optional[List[AddressFeatures]] = Field(default_factory=list, description="Признаки выходных адресов")


class Prediction(BaseModel):
    """Предсказание модели"""
    class_: str = Field(..., alias="class", description="Класс транзакции: illicit, licit, unknown")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Уверенность модели (0.0 - 1.0)")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Оценка риска (0.0 - 1.0)")

    class Config:
        allow_population_by_field_name = True


class ResponseDetails(BaseModel):
    """Детали ответа"""
    model_version: str = Field(default="1.0.0", description="Версия модели ML")
    inference_time_ms: float = Field(..., description="Время инференса в миллисекундах")
    feature_importance: Optional[Dict[str, float]] = Field(None, description="Важность признаков")


class MLResponse(BaseModel):
    """Успешный ответ Python сервиса"""
    success: bool = Field(True, description="Успешность обработки запроса")
    transaction_id: str = Field(..., description="ID транзакции")
    prediction: Prediction = Field(..., description="Предсказание модели")
    explanation: str = Field(..., description="Текстовое объяснение предсказания")
    details: ResponseDetails = Field(..., description="Детали ответа")


class ErrorResponse(BaseModel):
    """Ответ с ошибкой"""
    success: bool = Field(False, description="Успешность обработки запроса")
    error: Dict = Field(..., description="Информация об ошибке")

