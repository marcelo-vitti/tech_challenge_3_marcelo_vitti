import numpy as np

from xgboost import XGBClassifier


def _treat_input(prompt: str) -> np.ndarray:
    user_input = np.array([[float(x.strip()) for x in prompt.split(",")]])
    return user_input


def _get_prediction(model: XGBClassifier, user_input: np.ndarray) -> int:
    prediction = model.predict(user_input)
    return int(prediction[0])


def _get_probability(model: XGBClassifier, user_input: np.ndarray) -> float:
    probability = model.predict_proba(user_input)
    return float(probability[0][1])


def _generate_numerical_response(
    numpy_array_input: np.ndarray, model: XGBClassifier
) -> tuple[int, float]:
    prediction = _get_prediction(model, numpy_array_input)
    probability = _get_probability(model, numpy_array_input)
    return prediction, probability


def generate_user_answer(prompt: str, model: XGBClassifier) -> str:
    numpy_array_input = _treat_input(prompt)
    prediction, probability = _generate_numerical_response(numpy_array_input, model)
    response = (
        "The prediction is: {} with probability of having diabetes: {:.2f}%.".format(
            prediction, probability * 100
        )
    )
    return response
