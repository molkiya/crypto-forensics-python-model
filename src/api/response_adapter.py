"""
Адаптер для преобразования выхода модели в JSON ответ для Rust
"""
import torch
from typing import Dict, Optional
from .models import MLResponse, Prediction, ResponseDetails


def create_response_from_model_output(
    model_output: torch.Tensor,
    transaction_id: str,
    inference_time_ms: float,
    model_version: str = "1.0.0"
) -> MLResponse:
    """
    Преобразует выход модели в JSON ответ для Rust приложения.
    
    Args:
        model_output: Выход модели - логиты [num_nodes, num_classes]
        transaction_id: ID транзакции
        inference_time_ms: Время инференса в миллисекундах
        model_version: Версия модели
        
    Returns:
        MLResponse: Ответ в формате JSON
    """
    # Берем предсказание для центрального узла (транзакции, индекс 0)
    if model_output.shape[0] == 0:
        raise ValueError("Model output is empty")
    
    tx_logits = model_output[0]  # [num_classes]
    
    # Проверяем размерность
    if tx_logits.shape[0] < 2:
        raise ValueError(f"Model output has insufficient classes: {tx_logits.shape[0]}, expected at least 2")
    
    # Применяем softmax для получения вероятностей
    probabilities = torch.softmax(tx_logits, dim=0)
    
    # Определяем класс
    predicted_class_idx = torch.argmax(probabilities).item()
    
    # Маппинг индексов классов на названия
    # 0 = licit (легальная), 1 = illicit (нелегальная)
    class_mapping = {0: "licit", 1: "illicit"}
    predicted_class = class_mapping.get(predicted_class_idx, "unknown")
    
    # Уверенность = максимальная вероятность
    confidence = probabilities[predicted_class_idx].item()
    
    # Risk score = вероятность класса "illicit"
    if predicted_class_idx == 1:
        risk_score = probabilities[1].item()
    else:
        # Если предсказали licit, risk_score = 1 - confidence класса licit
        risk_score = 1.0 - probabilities[0].item()
    
    # Создаем объяснение
    explanation = _generate_explanation(predicted_class, confidence, risk_score)
    
    # Важность признаков (опционально, можно расширить)
    feature_importance = _calculate_feature_importance(model_output, probabilities)
    
    # Создаем ответ
    prediction = Prediction(
        class_=predicted_class,
        confidence=round(confidence, 4),
        risk_score=round(risk_score, 4)
    )
    
    details = ResponseDetails(
        model_version=model_version,
        inference_time_ms=round(inference_time_ms, 2),
        feature_importance=feature_importance
    )
    
    response = MLResponse(
        success=True,
        transaction_id=transaction_id,
        prediction=prediction,
        explanation=explanation,
        details=details
    )
    
    return response


def _generate_explanation(class_: str, confidence: float, risk_score: float) -> str:
    """
    Генерирует текстовое объяснение предсказания.
    """
    if class_ == "illicit":
        if confidence > 0.8:
            return f"Transaction shows strong patterns consistent with money laundering (confidence: {confidence:.2%}, risk: {risk_score:.2%})"
        elif confidence > 0.6:
            return f"Transaction exhibits suspicious characteristics that may indicate illicit activity (confidence: {confidence:.2%}, risk: {risk_score:.2%})"
        else:
            return f"Transaction shows some patterns that could be associated with illicit activity, but confidence is moderate (confidence: {confidence:.2%}, risk: {risk_score:.2%})"
    elif class_ == "licit":
        if confidence > 0.8:
            return f"Transaction appears to be legitimate with high confidence (confidence: {confidence:.2%}, risk: {risk_score:.2%})"
        else:
            return f"Transaction appears to be legitimate, though with moderate confidence (confidence: {confidence:.2%}, risk: {risk_score:.2%})"
    else:
        return f"Transaction classification is uncertain (confidence: {confidence:.2%}, risk: {risk_score:.2%})"


def _calculate_feature_importance(
    model_output: torch.Tensor,
    probabilities: torch.Tensor
) -> Optional[Dict[str, float]]:
    """
    Вычисляет важность признаков (упрощенная версия).
    В реальной реализации можно использовать gradient-based методы.
    """
    # Упрощенная версия: используем разницу в вероятностях классов
    if model_output.shape[0] > 0 and model_output.shape[1] >= 2:
        # Разница между вероятностями классов как мера важности
        prob_diff = abs(probabilities[0] - probabilities[1]).item()
        
        return {
            "class_separation": round(prob_diff, 4),
            "illicit_probability": round(probabilities[1].item(), 4),
            "licit_probability": round(probabilities[0].item(), 4)
        }
    
    return None

