import kaggle

kaggle.api.authenticate()

kaggle.api.dataset_download_files(
    "mohankrishnathalla/diabetes-health-indicators-dataset",
    path="../data/raw",
    unzip=True,
)
