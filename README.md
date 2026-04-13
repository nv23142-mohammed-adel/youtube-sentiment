# YouTube Sentiment Analysis

Sentiment analysis pipeline for YouTube/Reddit comments using LightGBM + TF-IDF, tracked with DVC and MLflow.

# Environment Setup


## Create and activate conda environment

```bash
conda create -n youtube python=3.11 -c main -y
conda init bash
conda activate youtube
```

## Install dependencies

```bash
pip install -r requirements.txt
```


# DVC Pipeline

## Initialize DVC

```bash
dvc init -f
```

## Run the full pipeline

```bash
dvc repro
```

## View pipeline graph

```bash
dvc dag
```

### Pipeline stages

| Stage | Script | Output |
|---|---|---|
| `data_ingestion` | `src/data/data_ingestion.py` | `data/raw/` |
| `data_preprocessing` | `src/data/data_preprocessing.py` | `data/interim/` |
| `model_building` | `src/model/model_building.py` | `lgbm_model.pkl`, `tfidf_vectorizer.pkl` |
| `model_evaluation` | `src/model/model_evaluation.py` | `experiment_info.json` |
| `model_registration` | `src/model/register_model.py` | — |

### Pipeline parameters (`params.yaml`)

| Parameter | Default | Description |
|---|---|---|
| `data_ingestion.test_size` | `0.20` | Fraction of data held out for testing |
| `model_building.ngram_range` | `[1, 3]` | N-gram range for TF-IDF |
| `model_building.max_features` | `1000` | Maximum TF-IDF vocabulary size |
| `model_building.learning_rate` | `0.09` | LightGBM learning rate |
| `model_building.max_depth` | `20` | LightGBM max tree depth |
| `model_building.n_estimators` | `367` | Number of LightGBM boosting rounds |


# Environment Variables

## MLflow / Pipeline

| Variable | Required | Default | Description |
|---|---|---|---|
| `MLFLOW_TRACKING_URI` | No | `http://localhost:5000` | MLflow tracking server URL. Set to your AWS-hosted MLflow instance for remote tracking. |

## Flask App (`flask_app/app.py`)

| Variable | Required | Default | Description |
|---|---|---|---|
| `MODEL_SOURCE` | No | `local` | Set to `mlflow` to load the model from the MLflow registry; omit or set to `local` to load from local pickle files. |
| `MODEL_PATH` | No | `./lgbm_model.pkl` | Path to the local LightGBM model pickle. Used when `MODEL_SOURCE=local`. |
| `VECTORIZER_PATH` | No | `./tfidf_vectorizer.pkl` | Path to the local TF-IDF vectorizer pickle. Used for both `local` and `mlflow` sources. |
| `MODEL_NAME` | No | `yt_chrome_plugin_model` | MLflow registered model name. Used when `MODEL_SOURCE=mlflow`. |
| `MODEL_VERSION` | No | `1` | MLflow model version to load. Used when `MODEL_SOURCE=mlflow`. |
| `MLFLOW_TRACKING_URI` | No | `http://localhost:5000` | MLflow tracking server URL. Used when `MODEL_SOURCE=mlflow`. |


# Running the Flask App

## Local (pickle files)

```bash
python flask_app/app.py
```

The app starts on `http://0.0.0.0:5000`.

## With MLflow registry (e.g. AWS)

```bash
export MODEL_SOURCE=mlflow
export MLFLOW_TRACKING_URI=http://<your-mlflow-server>:5000
export MODEL_NAME=yt_chrome_plugin_model
export MODEL_VERSION=1
python flask_app/app.py
```


# Cloud Configuration (AWS)

```bash
aws configure
```
