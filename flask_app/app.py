import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import base64
import io
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates
import pickle

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define the preprocessing function
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        comment = comment.lower()
        comment = comment.strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])
        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment


def load_model_from_mlflow(model_name, model_version, vectorizer_path):
    """Load the model from MLflow registry and vectorizer from local storage."""
    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    with open(vectorizer_path, 'rb') as file:
        vectorizer = pickle.load(file)
    return model, vectorizer


def load_model_local(model_path, vectorizer_path):
    """Load the model and vectorizer from local pickle files."""
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    with open(vectorizer_path, 'rb') as file:
        vectorizer = pickle.load(file)
    return model, vectorizer


# Select loading strategy via MODEL_SOURCE env var.
# Set MODEL_SOURCE=mlflow (and MLFLOW_TRACKING_URI, MODEL_NAME, MODEL_VERSION) for AWS.
# Defaults to local pickle files for local testing.
_model_source = os.environ.get('MODEL_SOURCE', 'local')

if _model_source == 'mlflow':
    _mlflow_model_name = os.environ.get('MODEL_NAME', 'yt_chrome_plugin_model')
    _mlflow_model_version = os.environ.get('MODEL_VERSION', '1')
    _vectorizer_path = os.environ.get('VECTORIZER_PATH', './tfidf_vectorizer.pkl')
    model, vectorizer = load_model_from_mlflow(_mlflow_model_name, _mlflow_model_version, _vectorizer_path)
else:
    _model_path = os.environ.get('MODEL_PATH', './lgbm_model.pkl')
    _vectorizer_path = os.environ.get('VECTORIZER_PATH', './tfidf_vectorizer.pkl')
    model, vectorizer = load_model_local(_model_path, _vectorizer_path)


# ---------------------------------------------------------------------------
# In-memory server-side cache: video_id -> analysis result dict
# ---------------------------------------------------------------------------
_analysis_cache = {}


# ---------------------------------------------------------------------------
# Image-generation helpers — return base64-encoded PNG strings
# ---------------------------------------------------------------------------

def _build_chart_b64(sentiment_counts):
    """Build a sentiment pie chart and return it as a base64 PNG string."""
    labels = ['Positive', 'Neutral', 'Negative']
    sizes = [
        int(sentiment_counts.get('1', 0)),
        int(sentiment_counts.get('0', 0)),
        int(sentiment_counts.get('-1', 0))
    ]
    if sum(sizes) == 0:
        raise ValueError("Sentiment counts sum to zero")
    colors = ['#36A2EB', '#C9CBCF', '#FF6384']
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
           startangle=140, textprops={'color': 'w'})
    ax.axis('equal')
    buf = io.BytesIO()
    fig.savefig(buf, format='PNG', transparent=True)
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode('utf-8')


def _build_wordcloud_b64(comments):
    """Build a word cloud from raw comment texts and return it as a base64 PNG string."""
    preprocessed = [preprocess_comment(c) for c in comments]
    text = ' '.join(preprocessed)
    wc = WordCloud(
        width=800, height=400, background_color='black',
        colormap='Blues', stopwords=set(stopwords.words('english')), collocations=False
    ).generate(text)
    buf = io.BytesIO()
    wc.to_image().save(buf, format='PNG')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def _build_trend_b64(sentiment_data):
    """Build a monthly sentiment trend graph and return it as a base64 PNG string."""
    df = pd.DataFrame(sentiment_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df['sentiment'] = df['sentiment'].astype(int)

    sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
    monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)
    monthly_totals = monthly_counts.sum(axis=1)
    monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

    for sv in [-1, 0, 1]:
        if sv not in monthly_percentages.columns:
            monthly_percentages[sv] = 0
    monthly_percentages = monthly_percentages[[-1, 0, 1]]

    colors = {-1: 'red', 0: 'gray', 1: 'green'}
    fig, ax = plt.subplots(figsize=(12, 6))
    for sv in [-1, 0, 1]:
        ax.plot(monthly_percentages.index, monthly_percentages[sv],
                marker='o', linestyle='-', label=sentiment_labels[sv], color=colors[sv])
    ax.set_title('Monthly Sentiment Percentage Over Time')
    ax.set_xlabel('Month')
    ax.set_ylabel('Percentage of Comments (%)')
    ax.grid(True)
    plt.xticks(rotation=45)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
    ax.legend()
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='PNG')
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode('utf-8')


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def home():
    return "Welcome to our flask api"


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Combined endpoint: run predictions + generate all images in one call.
    Caches results by video_id so repeated calls for the same video are instant.

    Request body:
        {
            "video_id": "<youtube-video-id>",     # optional but required for caching
            "comments": [{"text": "...", "timestamp": "...", "authorId": "..."}, ...]
        }

    Response:
        {
            "predictions":      [{"comment": "...", "sentiment": "1", "timestamp": "..."}, ...],
            "sentiment_counts": {"1": N, "0": N, "-1": N},
            "chart_image":      "<base64 PNG>",
            "trend_image":      "<base64 PNG>",
            "wordcloud_image":  "<base64 PNG>",
            "from_cache":       true | false
        }
    """
    data = request.json
    video_id = data.get('video_id')
    comments_data = data.get('comments')

    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    # Return from server cache if available
    if video_id and video_id in _analysis_cache:
        return jsonify({**_analysis_cache[video_id], "from_cache": True})

    try:
        comments = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]

        preprocessed = [preprocess_comment(c) for c in comments]
        transformed = vectorizer.transform(preprocessed)
        dense = transformed.toarray()
        preds = [str(p) for p in model.predict(dense).tolist()]

        predictions = [
            {"comment": c, "sentiment": s, "timestamp": t}
            for c, s, t in zip(comments, preds, timestamps)
        ]

        sentiment_counts = {"1": 0, "0": 0, "-1": 0}
        sentiment_data = []
        for item in predictions:
            key = item['sentiment']
            sentiment_counts[key] = sentiment_counts.get(key, 0) + 1
            sentiment_data.append({"timestamp": item['timestamp'], "sentiment": int(item['sentiment'])})

        result = {
            "predictions": predictions,
            "sentiment_counts": sentiment_counts,
            "chart_image": _build_chart_b64(sentiment_counts),
            "trend_image": _build_trend_b64(sentiment_data),
            "wordcloud_image": _build_wordcloud_b64(comments),
        }

        if video_id:
            _analysis_cache[video_id] = result

        return jsonify({**result, "from_cache": False})

    except Exception as e:
        app.logger.error(f"Error in /analyze: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    data = request.json
    comments_data = data.get('comments')

    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    try:
        comments = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        transformed_comments = vectorizer.transform(preprocessed_comments)
        dense_comments = transformed_comments.toarray()
        predictions = model.predict(dense_comments).tolist()
        predictions = [str(pred) for pred in predictions]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    response = [
        {"comment": comment, "sentiment": sentiment, "timestamp": timestamp}
        for comment, sentiment, timestamp in zip(comments, predictions, timestamps)
    ]
    return jsonify(response)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comments = data.get('comments')
    print("i am the comment: ", comments)
    print("i am the comment type: ", type(comments))

    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        transformed_comments = vectorizer.transform(preprocessed_comments)
        dense_comments = transformed_comments.toarray()
        predictions = model.predict(dense_comments).tolist()
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    response = [{"comment": comment, "sentiment": sentiment} for comment, sentiment in zip(comments, predictions)]
    return jsonify(response)


@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get('sentiment_counts')
        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400
        img_bytes = base64.b64decode(_build_chart_b64(sentiment_counts))
        return send_file(io.BytesIO(img_bytes), mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500


@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get('comments')
        if not comments:
            return jsonify({"error": "No comments provided"}), 400
        img_bytes = base64.b64decode(_build_wordcloud_b64(comments))
        return send_file(io.BytesIO(img_bytes), mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud: {e}")
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500


@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')
        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400
        img_bytes = base64.b64decode(_build_trend_b64(sentiment_data))
        return send_file(io.BytesIO(img_bytes), mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_trend_graph: {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
