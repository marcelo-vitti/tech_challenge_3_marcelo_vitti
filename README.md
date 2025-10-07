# Diabetes Health Indicators ML Repository

This repository provides an end-to-end solution for predicting diabetes using health indicators. It includes data extraction, ETL, exploratory analysis, feature engineering, model training, and a Streamlit chatbot for user interaction.

## Directory Structure

```
.
├── main.py
├── requirements.txt
├── data/
│   └── raw/
│       └── diabetes_dataset.csv
├── database/
│   └── health_indicators.db
├── etl/
│   ├── extract.py
│   └── load.py
├── exploratory_analysis/
│   └── exploratory_analysis.ipynb
├── model/
│   ├── training.ipynb
│   └── models/
│       └── xgboost_model_v01.json
├── utils/
│   └── handle_user_answer.py
```

## Components

### Data Extraction & Loading
- **etl/extract.py**: Downloads the diabetes dataset from Kaggle.
- **etl/load.py**: Loads the CSV data into a SQLite database (`database/health_indicators.db`).

### Exploratory Analysis
- **exploratory_analysis/exploratory_analysis.ipynb**: Jupyter notebook for EDA, visualizations, and understanding feature distributions.

### Model Training
- **model/training.ipynb**: Trains an XGBoost classifier on selected features and saves the model to `xgboost_model_v01.json`.

### Chatbot Application
- **main.py**: Streamlit app that loads the trained model and interacts with users. Users input health indicators, and the chatbot predicts diabetes risk.

### Utilities
- **utils/handle_user_answer.py**: Contains functions for parsing user input, generating predictions, and formatting responses.

## Usage

### 1. Create virtual environment and Install Dependencies

```sh
python -m venv .venv
```

```sh
pip install -r requirements.txt
```

### 2. Download and Load Data

Run the ETL scripts to fetch and store the dataset:

```sh
python etl/extract.py
python etl/load.py
```

### 3. Train the Model

Open and run the cells in `model/training.ipynb` to train and save the XGBoost model.

### 4. Run the Chatbot

Start the Streamlit app:

```sh
streamlit run main.py
```

### 5. Interact

Provide comma-separated values for: `hba1c, glucose_postprandial, glucose_fasting, family_history_diabetes, age`

**Example:**
```
8.18, 236, 136, 0, 58
```

## File References

- **main.py**: Streamlit chatbot interface.
- **etl/extract.py**: Kaggle data download.
- **etl/load.py**: Load CSV to SQLite.
- **utils/handle_user_answer.py**: User input handling and prediction.
- **model/training.ipynb**: Model training notebook.
- **exploratory_analysis/exploratory_analysis.ipynb**: EDA notebook.

## Requirements

See `requirements.txt` for all dependencies.

## License

See `LICENSE` for details.

---

**Note:** Ensure you have a Kaggle API key for data extraction and the correct Python environment for Streamlit and XGBoost.