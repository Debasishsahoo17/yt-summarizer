
import os
import re
import logging
from urllib.parse import urlparse, parse_qs

import numpy as np
import nltk
import yt_dlp

from flask import Flask, request, jsonify, render_template
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from youtube_transcript_api import YouTubeTranscriptApi
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize


# ------------------------------
# Flask Configuration
# ------------------------------

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024  # 16 KB request limit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["100 per hour"],
    storage_uri="memory://",
)

# Download tokenizer data if unavailable.
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


# ------------------------------
# Homepage
# ------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# ------------------------------
# YouTube URL Validation
# ------------------------------

def extract_video_id(video_url):
    """
    Extract a valid YouTube video ID.
    Reject non-YouTube URLs.
    """

    if not isinstance(video_url, str):
        return None

    video_url = video_url.strip()

    if len(video_url) > 2048:
        return None

    try:
        parsed = urlparse(video_url)

        if parsed.scheme not in ("http", "https"):
            return None

        hostname = (parsed.hostname or "").lower()

        if hostname in ("youtube.com", "www.youtube.com", "m.youtube.com"):
            if parsed.path == "/watch":
                video_id = parse_qs(parsed.query).get("v", [None])[0]

            elif parsed.path.startswith(("/shorts/", "/embed/", "/live/")):
                parts = parsed.path.strip("/").split("/")
                video_id = parts[1] if len(parts) == 2 else None

            else:
                return None

        elif hostname == "youtu.be":
            video_id = parsed.path.strip("/")

        else:
            return None

        if video_id and re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
            return video_id

    except (ValueError, TypeError):
        pass

    return None


# ------------------------------
# Song / Lyrics Detection
# ------------------------------

def is_song_lyrics(text):
    """
    Basic heuristic to detect music transcripts.
    """

    if not text or not text.strip():
        return False

    if "♪" in text or "♫" in text:
        return True

    lines = [
        line.strip().lower()
        for line in text.splitlines()
        if line.strip()
    ]

    if len(lines) >= 5:
        unique_ratio = len(set(lines)) / len(lines)

        if unique_ratio < 0.6:
            return True

    return False


# ------------------------------
# YouTube Metadata Fallback
# ------------------------------

def get_video_metadata(video_id):
    """
    Retrieve title and description when a transcript
    is unavailable.
    """

    video_url = f"https://www.youtube.com/watch?v={video_id}"

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "socket_timeout": 10,
        "retries": 1,
        "extractor_retries": 1,
        "ignoreerrors": False,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)

        if not info:
            return None

        title = info.get("title") or ""
        description = info.get("description") or ""
        tags = info.get("tags") or []

        metadata = " ".join([
            title,
            description,
            " ".join(str(tag) for tag in tags[:20])
        ])

        return metadata.strip()[:20000] or None

    except Exception:
        logger.exception("Failed to retrieve video metadata")
        return None


# ------------------------------
# Transcript Extraction
# ------------------------------

def get_transcript(video_id):
    """
    Retrieve English transcript.
    Fall back to metadata if unavailable.
    """

    try:
        api = YouTubeTranscriptApi()

        transcript = api.fetch(
            video_id,
            languages=["en"]
        )

        text = " ".join(
            snippet.text
            for snippet in transcript
            if snippet.text.strip()
        )

        if text.strip():
            if is_song_lyrics(text):
                return None, None, (
                    "This appears to be a music video. "
                    "Summarization is not supported."
                )

            return text, "transcript", None

    except Exception:
        logger.info(
            "Transcript unavailable for video %s",
            video_id
        )

    metadata = get_video_metadata(video_id)

    if metadata:
        return metadata, "metadata", None

    return None, None, (
        "Unable to retrieve a transcript or video metadata."
    )


# ------------------------------
# TF-IDF Text Summarization
# ------------------------------

def summarize_text(text, num_sentences=5):
    """
    Extract the most relevant sentences using TF-IDF.
    This is extractive summarization, not generative AI.
    """

    if not text or not text.strip():
        return []

    sentences = [
        sentence.strip()
        for sentence in sent_tokenize(text)
        if len(sentence.strip()) >= 15
    ]

    if not sentences:
        return [text.strip()[:500]]

    if len(sentences) <= num_sentences:
        return sentences

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000
        )

        matrix = vectorizer.fit_transform(sentences)

        # Score sentences using their TF-IDF weights.
        scores = np.asarray(
            matrix.sum(axis=1)
        ).ravel()

        top_indices = np.argsort(scores)[-num_sentences:]

        # Preserve the original video narration order.
        top_indices = sorted(top_indices)

        return [
            sentences[index]
            for index in top_indices
        ]

    except ValueError:
        logger.warning("TF-IDF failed; using first sentences")
        return sentences[:num_sentences]


# ------------------------------
# Summarization API
# ------------------------------

@app.route("/transcript", methods=["POST"])
@limiter.limit("10 per minute")
def transcript_api():

    if not request.is_json:
        return jsonify({
            "error": "Please send JSON data."
        }), 400

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Invalid JSON request."
        }), 400

    video_url = data.get("url")

    video_id = extract_video_id(video_url)

    if not video_id:
        return jsonify({
            "error": "Please provide a valid YouTube URL."
        }), 400

    text, source, error = get_transcript(video_id)

    if error:
        return jsonify({
            "error": error
        }), 422

    summary_sentences = summarize_text(
        text,
        num_sentences=5
    )

    if not summary_sentences:
        return jsonify({
            "error": "Unable to generate a summary."
        }), 422

    summary = "\n".join(
        f"- {sentence}"
        for sentence in summary_sentences
    )

    return jsonify({
        "video_id": video_id,
        "summary": summary,
        "source": source,
        "sentences": len(summary_sentences),
        "method": "TF-IDF Extractive Summarization"
    }), 200


# ------------------------------
# Health Check
# ------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok"
    }), 200


# ------------------------------
# Run Application
# ------------------------------

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
