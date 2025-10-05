import numpy as np

from xgboost import XGBClassifier


def _treat_input(prompt: str) -> np.ndarray:
    """
    Converts a comma-separated string of numbers into a NumPy array.

    Args:
        prompt (str): A string containing numbers separated by commas (e.g., "1.0, 2.5, 3").

    Returns:
        np.ndarray: A 2D NumPy array with a single row containing the parsed float values.

    Raises:
        ValueError: If any of the values in the input string cannot be converted to float.
    """
    user_input = np.array([[float(x.strip()) for x in prompt.split(",")]])
    return user_input


def _get_prediction(model: XGBClassifier, user_input: np.ndarray) -> int:
    """
    Generates a prediction using the provided XGBClassifier model and user input.

    Args:
        model (XGBClassifier): The trained XGBoost classifier used for prediction.
        user_input (np.ndarray): The input features as a NumPy array, shaped appropriately for the model.

    Returns:
        int: The predicted class label as an integer.
    """
    prediction = model.predict(user_input)
    return int(prediction[0])


def _get_probability(model: XGBClassifier, user_input: np.ndarray) -> float:
    """
    Calculates the probability of the positive class for a given user input using the provided XGBClassifier model.

    Args:
        model (XGBClassifier): The trained XGBoost classifier used to predict probabilities.
        user_input (np.ndarray): The input features for which the probability is to be predicted. Should be shaped appropriately for the model.

    Returns:
        float: The predicted probability of the positive class (class 1) for the given input.
    """
    probability = model.predict_proba(user_input)
    return float(probability[0][1])


def _generate_numerical_response(
    numpy_array_input: np.ndarray, model: XGBClassifier
) -> tuple[int, float]:
    """
    Generates a numerical response by predicting the class and its probability using the provided model and input data.

    Args:
        numpy_array_input (np.ndarray): Input features as a NumPy array.
        model (XGBClassifier): Trained XGBoost classifier model.

    Returns:
        tuple[int, float]: A tuple containing the predicted class (int) and the associated probability (float).
    """
    prediction = _get_prediction(model, numpy_array_input)
    probability = _get_probability(model, numpy_array_input)
    return prediction, probability


def generate_user_answer(prompt: str, model: XGBClassifier) -> str:
    """
    Generates a user-friendly answer based on a prompt and an XGBClassifier model prediction.

    Args:
        prompt (str): The input string containing user data to be processed for prediction.
        model (XGBClassifier): The trained XGBoost classifier used to make the prediction.

    Returns:
        str: A formatted string containing the prediction and the probability of having diabetes.
    """
    numpy_array_input = _treat_input(prompt)
    prediction, probability = _generate_numerical_response(numpy_array_input, model)
    response = (
        "The prediction is: {} with probability of having diabetes: {:.2f}%.".format(
            prediction, probability * 100
        )
    )
    return response
