


from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import requests
import os
import json
from collections import defaultdict
import re
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "YOUR_TMDB_API_KEY_HERE")
TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG = "https://image.tmdb.org/t/p/w500"

# ─── MOOD → GENRE MAPPING ─────────────────────────────────────────────────────
MOOD_GENRE_MAP = {
    "happy":      {"genres": [35, 10751, 16],        "keywords": ["feel-good", "uplifting", "fun"]},
    "sad":        {"genres": [18, 10749],             "keywords": ["emotional", "touching", "heartfelt"]},
    "excited":    {"genres": [28, 12, 878],           "keywords": ["thrilling", "epic", "adventure"]},
    "scared":     {"genres": [27, 9648, 53],          "keywords": ["scary", "suspense", "dark"]},
    "romantic":   {"genres": [10749, 35],             "keywords": ["romance", "love", "passionate"]},
    "thoughtful": {"genres": [18, 878, 9648],         "keywords": ["philosophical", "cerebral", "deep"]},
    "bored":      {"genres": [28, 12, 35, 878],       "keywords": ["fast-paced", "entertaining", "exciting"]},
    "relaxed":    {"genres": [99, 10751, 36],         "keywords": ["calm", "easy", "gentle"]},
    "angry":      {"genres": [28, 53, 80],            "keywords": ["intense", "powerful", "raw"]},
    "nostalgic":  {"genres": [36, 10751, 10402],      "keywords": ["classic", "retro", "timeless"]},
    "curious":    {"genres": [878, 9648, 99],         "keywords": ["mysterious", "educational", "mind-bending"]},
    "lonely":     {"genres": [10749, 18, 10751],      "keywords": ["companionship", "friendship", "heartwarming"]},
}

GENRE_MAP = {
    28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy",
    80: "Crime", 99: "Documentary", 18: "Drama", 10751: "Family",
    14: "Fantasy", 36: "History", 27: "Horror", 10402: "Music",
    9648: "Mystery", 10749: "Romance", 878: "Science Fiction",
    10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western"
}

# ─── IN-MEMORY STORES ─────────────────────────────────────────────────────────
user_ratings = defaultdict(dict)   # {user_id: {movie_id: rating}}
movie_cache  = {}                   # {movie_id: movie_data}

# ─── SENTENCE TRANSFORMER (lazy load) ─────────────────────────────────────────
_st_model = None
def get_st_model():
    global _st_model
    if _st_model is None:
        _st_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _st_model

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def tmdb_get(path, params=None):
    params = params or {}
    params["api_key"] = TMDB_API_KEY
    try:
        r = requests.get(f"{TMDB_BASE}{path}", params=params, timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e), "results": []}

def enrich_movie(m):
    return {
        "id":          m.get("id"),
        "title":       m.get("title", m.get("name", "")),
        "overview":    m.get("overview", ""),
        "poster":      f"{TMDB_IMG}{m['poster_path']}" if m.get("poster_path") else None,
        "backdrop":    f"https://image.tmdb.org/t/p/w1280{m['backdrop_path']}" if m.get("backdrop_path") else None,
        "genres":      [GENRE_MAP.get(g, g) for g in (m.get("genre_ids") or [g["id"] for g in m.get("genres", [])])],
        "rating":      round(m.get("vote_average", 0), 1),
        "votes":       m.get("vote_count", 0),
        "year":        (m.get("release_date") or m.get("first_air_date") or "")[:4],
        "popularity":  m.get("popularity", 0),
        "adult":       m.get("adult", False),
    }

def detect_mood_from_text(text):
    """NLP mood classifier using keyword matching + sentence embeddings."""
    text_lower = text.lower()
    # Rule-based fast pass
    keyword_scores = defaultdict(int)
    mood_keywords = {
        "happy":      ["happy", "joy", "fun", "laugh", "cheerful", "good mood", "great"],
        "sad":        ["sad", "cry", "depressed", "unhappy", "lonely", "grief", "miss"],
        "excited":    ["excited", "thrilled", "pumped", "hyped", "energetic", "wired"],
        "scared":     ["scared", "afraid", "anxious", "nervous", "fear", "terrified"],
        "romantic":   ["romantic", "love", "date", "crush", "heart", "valentine"],
        "thoughtful": ["thinking", "philosophical", "ponder", "reflect", "deep", "intellectual"],
        "bored":      ["bored", "nothing to do", "dull", "flat", "uninterested"],
        "relaxed":    ["relaxed", "chill", "calm", "peaceful", "cozy", "quiet"],
        "angry":      ["angry", "mad", "frustrated", "irritated", "rage", "furious"],
        "nostalgic":  ["nostalgic", "memories", "old days", "classic", "childhood", "retro"],
        "curious":    ["curious", "wonder", "explore", "discover", "learn", "fascinated"],
        "lonely":     ["lonely", "alone", "isolated", "want company", "miss people"],
    }
    for mood, kws in mood_keywords.items():
        for kw in kws:
            if kw in text_lower:
                keyword_scores[mood] += 1
    if keyword_scores:
        return max(keyword_scores, key=keyword_scores.get)
    return "happy"  # default

def content_based_score(movie, preferences):
    """Score a movie based on user's preferred genres and keywords."""
    score = 0.0
    movie_genres = set(movie.get("genres", []))
    pref_genres  = set(preferences.get("genres", []))
    if pref_genres:
        score += 0.5 * len(movie_genres & pref_genres) / max(len(pref_genres), 1)
    score += 0.3 * min(movie.get("rating", 0) / 10.0, 1.0)
    score += 0.2 * min(np.log1p(movie.get("popularity", 0)) / 10.0, 1.0)
    return round(score, 4)

def hybrid_score(user_score, genre_score, trend_score, w=(0.5, 0.3, 0.2)):
    return round(w[0]*user_score + w[1]*genre_score + w[2]*trend_score, 4)

def explain_recommendation(movie, mood, matched_genres):
    parts = []
    if matched_genres:
        parts.append(f"Matches your preferred genres: {', '.join(matched_genres[:2])}")
    if movie.get("rating", 0) >= 7.5:
        parts.append(f"Highly rated ({movie['rating']}/10)")
    if mood:
        parts.append(f"Great for a {mood} mood")
    if movie.get("votes", 0) > 5000:
        parts.append("Widely loved by audiences")
    return " · ".join(parts) if parts else "Recommended based on your taste"

# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "message": "CineMind API running"})


# ── MOVIES ────────────────────────────────────────────────────────────────────
@app.route("/api/movies/trending")
def trending():
    data = tmdb_get("/trending/movie/week")
    movies = [enrich_movie(m) for m in data.get("results", [])[:20]]
    return jsonify({"movies": movies})

@app.route("/api/movies/popular")
def popular():
    page = request.args.get("page", 1)
    data = tmdb_get("/movie/popular", {"page": page})
    movies = [enrich_movie(m) for m in data.get("results", [])]
    return jsonify({"movies": movies, "total_pages": data.get("total_pages", 1)})

@app.route("/api/movies/top_rated")
def top_rated():
    data = tmdb_get("/movie/top_rated")
    movies = [enrich_movie(m) for m in data.get("results", [])[:20]]
    return jsonify({"movies": movies})

@app.route("/api/movies/<int:movie_id>")
def movie_detail(movie_id):
    data   = tmdb_get(f"/movie/{movie_id}")
    credits= tmdb_get(f"/movie/{movie_id}/credits")
    similar= tmdb_get(f"/movie/{movie_id}/similar")
    movie  = enrich_movie(data)
    movie["cast"] = [
        {"name": c["name"], "character": c["character"],
         "photo": f"{TMDB_IMG}{c['profile_path']}" if c.get("profile_path") else None}
        for c in credits.get("cast", [])[:8]
    ]
    movie["similar"] = [enrich_movie(m) for m in similar.get("results", [])[:6]]
    movie["tagline"] = data.get("tagline", "")
    movie["runtime"] = data.get("runtime", 0)
    movie["revenue"] = data.get("revenue", 0)
    return jsonify(movie)

@app.route("/api/movies/search")
def search():
    q    = request.args.get("q", "")
    page = request.args.get("page", 1)
    if not q:
        return jsonify({"movies": [], "total_pages": 0})
    data   = tmdb_get("/search/movie", {"query": q, "page": page})
    movies = [enrich_movie(m) for m in data.get("results", [])]
    return jsonify({"movies": movies, "total_pages": data.get("total_pages", 1)})

@app.route("/api/movies/genre/<int:genre_id>")
def by_genre(genre_id):
    data   = tmdb_get("/discover/movie", {"with_genres": genre_id, "sort_by": "popularity.desc"})
    movies = [enrich_movie(m) for m in data.get("results", [])[:20]]
    return jsonify({"movies": movies, "genre": GENRE_MAP.get(genre_id, str(genre_id))})


# ── RECOMMENDATIONS ───────────────────────────────────────────────────────────
@app.route("/api/recommend/mood", methods=["POST"])
def recommend_by_mood():
    body  = request.get_json() or {}
    mood  = body.get("mood", "happy").lower()
    text  = body.get("text", "")
    user_id = body.get("user_id", "anonymous")

    # Detect mood from free text if provided
    if text and mood == "auto":
        mood = detect_mood_from_text(text)

    config = MOOD_GENRE_MAP.get(mood, MOOD_GENRE_MAP["happy"])
    genre_ids = config["genres"]

    # Fetch movies for each genre and merge
    all_movies = []
    seen = set()
    for gid in genre_ids[:2]:
        data = tmdb_get("/discover/movie", {
            "with_genres": gid,
            "sort_by": "vote_average.desc",
            "vote_count.gte": 500,
            "page": 1
        })
        for m in data.get("results", []):
            if m["id"] not in seen:
                seen.add(m["id"])
                enriched = enrich_movie(m)
                enriched["mood_score"]   = content_based_score(enriched, {"genres": [GENRE_MAP[g] for g in genre_ids if g in GENRE_MAP]})
                enriched["mood"]         = mood
                enriched["explanation"]  = explain_recommendation(
                    enriched, mood,
                    list(set(enriched["genres"]) & {GENRE_MAP[g] for g in genre_ids if g in GENRE_MAP})
                )
                all_movies.append(enriched)

    all_movies.sort(key=lambda x: (x["mood_score"], x["rating"]), reverse=True)
    return jsonify({"mood": mood, "movies": all_movies[:16], "genres_used": [GENRE_MAP.get(g, g) for g in genre_ids]})


@app.route("/api/recommend/content/<int:movie_id>")
def recommend_content_based(movie_id):
    """Content-based filtering using TF-IDF on overviews."""
    # Fetch the source movie
    source_data = tmdb_get(f"/movie/{movie_id}")
    source = enrich_movie(source_data)
    source["overview_text"] = source_data.get("overview", "")

    # Fetch candidates from similar genres
    candidates = []
    seen = {movie_id}
    similar = tmdb_get(f"/movie/{movie_id}/similar")
    for m in similar.get("results", [])[:10]:
        if m["id"] not in seen:
            seen.add(m["id"])
            e = enrich_movie(m)
            e["overview_text"] = m.get("overview", "")
            candidates.append(e)

    # Also fetch by genre
    for genre_id in source_data.get("genres", [])[:1]:
        data = tmdb_get("/discover/movie", {"with_genres": genre_id["id"], "sort_by": "vote_average.desc", "vote_count.gte": 200})
        for m in data.get("results", [])[:10]:
            if m["id"] not in seen:
                seen.add(m["id"])
                e = enrich_movie(m)
                e["overview_text"] = m.get("overview", "")
                candidates.append(e)

    if not candidates:
        return jsonify({"source": source, "recommendations": []})

    # TF-IDF cosine similarity
    texts = [source["overview_text"]] + [c["overview_text"] for c in candidates]
    try:
        tfidf = TfidfVectorizer(stop_words="english", max_features=500)
        mat   = tfidf.fit_transform(texts)
        sims  = cosine_similarity(mat[0:1], mat[1:])[0]
        for i, c in enumerate(candidates):
            c["similarity"] = round(float(sims[i]), 4)
            genre_overlap = set(c["genres"]) & set(source["genres"])
            c["explanation"] = f"Similar story · {', '.join(genre_overlap)}" if genre_overlap else "Similar story and themes"
        candidates.sort(key=lambda x: x["similarity"], reverse=True)
    except Exception:
        for c in candidates:
            c["similarity"] = 0.5

    return jsonify({"source": source, "recommendations": candidates[:12]})


@app.route("/api/recommend/collaborative", methods=["POST"])
def recommend_collaborative():
    """Simple user-based collaborative filtering."""
    body    = request.get_json() or {}
    user_id = body.get("user_id", "user_1")
    n       = body.get("n", 10)

    if not user_ratings or user_id not in user_ratings:
        # Cold start → fall back to popular
        data = tmdb_get("/movie/popular")
        movies = [enrich_movie(m) for m in data.get("results", [])[:n]]
        return jsonify({"movies": movies, "method": "popular_fallback", "reason": "Not enough rating data yet"})

    # Build user-item matrix
    all_users  = list(user_ratings.keys())
    all_movies = list({mid for ratings in user_ratings.values() for mid in ratings})
    if len(all_users) < 2:
        data = tmdb_get("/movie/popular")
        return jsonify({"movies": [enrich_movie(m) for m in data.get("results", [])[:n]], "method": "popular_fallback"})

    mat = np.zeros((len(all_users), len(all_movies)))
    uidx = {u: i for i, u in enumerate(all_users)}
    midx = {m: i for i, m in enumerate(all_movies)}
    for u, ratings in user_ratings.items():
        for mid, r in ratings.items():
            mat[uidx[u]][midx[mid]] = r

    # Cosine similarity between users
    user_vec = mat[uidx[user_id]]
    sims = cosine_similarity([user_vec], mat)[0]
    sims[uidx[user_id]] = -1  # exclude self
    best_neighbor_idx = np.argmax(sims)
    neighbor_id = all_users[best_neighbor_idx]

    # Movies neighbor liked that user hasn't seen
    user_seen = set(user_ratings[user_id].keys())
    neighbor_movies = {mid: r for mid, r in user_ratings[neighbor_id].items() if mid not in user_seen and r >= 3.5}
    top_movie_ids = sorted(neighbor_movies, key=neighbor_movies.get, reverse=True)[:n]

    results = []
    for mid in top_movie_ids:
        if mid in movie_cache:
            m = movie_cache[mid]
        else:
            m = enrich_movie(tmdb_get(f"/movie/{mid}"))
            movie_cache[mid] = m
        m["explanation"] = "Users like you also loved this"
        results.append(m)

    return jsonify({"movies": results, "method": "collaborative", "neighbor_similarity": round(float(sims[best_neighbor_idx]), 3)})


@app.route("/api/recommend/hybrid", methods=["POST"])
def recommend_hybrid():
    """Hybrid: content + collaborative + trending with explainability."""
    body    = request.get_json() or {}
    user_id = body.get("user_id", "anonymous")
    mood    = body.get("mood", "happy")
    genres  = body.get("genres", [])
    n       = body.get("n", 12)

    config     = MOOD_GENRE_MAP.get(mood, MOOD_GENRE_MAP["happy"])
    genre_ids  = config["genres"]

    data = tmdb_get("/discover/movie", {
        "with_genres": "|".join(str(g) for g in genre_ids[:3]),
        "sort_by": "popularity.desc",
        "vote_count.gte": 300,
    })
    movies = [enrich_movie(m) for m in data.get("results", [])[:30]]

    trend_data = tmdb_get("/trending/movie/week")
    trend_ids  = {m["id"]: i for i, m in enumerate(trend_data.get("results", []))}

    for m in movies:
        u_score  = user_ratings.get(user_id, {}).get(m["id"], 3.0) / 5.0
        g_score  = m["rating"] / 10.0
        t_rank   = trend_ids.get(m["id"], 50)
        t_score  = 1.0 - (t_rank / 50.0) if t_rank < 50 else 0.0
        m["hybrid_score"] = hybrid_score(u_score, g_score, t_score)
        matched = list(set(m["genres"]) & {GENRE_MAP.get(g, "") for g in genre_ids})
        m["explanation"] = explain_recommendation(m, mood, matched)
        m["score_breakdown"] = {"user": round(u_score, 2), "genre": round(g_score, 2), "trend": round(t_score, 2)}

    movies.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return jsonify({"movies": movies[:n], "mood": mood, "method": "hybrid"})


# ── CHATBOT ───────────────────────────────────────────────────────────────────
@app.route("/api/chat", methods=["POST"])
def chat():
    body    = request.get_json() or {}
    message = body.get("message", "").strip()
    history = body.get("history", [])

    if not message:
        return jsonify({"reply": "What kind of movie are you in the mood for?", "movies": []})

    msg_lower = message.lower()

    # Intent detection
    intent = "recommend"
    if any(w in msg_lower for w in ["search", "find", "look for", "show me"]):
        intent = "search"
    elif any(w in msg_lower for w in ["similar", "like", "more of"]):
        intent = "similar"
    elif any(w in msg_lower for w in ["genre", "type", "kind"]):
        intent = "genre"
    elif any(w in msg_lower for w in ["mood", "feel", "feeling"]):
        intent = "mood"

    mood    = detect_mood_from_text(message)
    movies  = []
    reply   = ""

    if intent in ("mood", "recommend"):
        config   = MOOD_GENRE_MAP.get(mood, MOOD_GENRE_MAP["happy"])
        data     = tmdb_get("/discover/movie", {
            "with_genres": str(config["genres"][0]),
            "sort_by": "vote_average.desc",
            "vote_count.gte": 500,
        })
        movies = [enrich_movie(m) for m in data.get("results", [])[:6]]
        reply  = f"I sense a **{mood}** mood! Here are some perfect picks for you:"

    elif intent == "search":
        # Extract query (remove filler words)
        query = re.sub(r'\b(search|find|look for|show me|movies?|films?|about)\b', '', msg_lower).strip()
        if query:
            data   = tmdb_get("/search/movie", {"query": query})
            movies = [enrich_movie(m) for m in data.get("results", [])[:6]]
            reply  = f"Here's what I found for **{query}**:"
        else:
            reply = "What title or topic would you like me to search for?"

    elif intent == "genre":
        # Find genre name in message
        found_genre = None
        for gid, gname in GENRE_MAP.items():
            if gname.lower() in msg_lower:
                found_genre = (gid, gname)
                break
        if found_genre:
            data   = tmdb_get("/discover/movie", {"with_genres": found_genre[0], "sort_by": "popularity.desc"})
            movies = [enrich_movie(m) for m in data.get("results", [])[:6]]
            reply  = f"Top **{found_genre[1]}** movies coming right up:"
        else:
            reply = f"Which genre interests you? Try: {', '.join(list(GENRE_MAP.values())[:8])}"

    if not reply:
        reply = "Tell me how you're feeling or what kind of movie you want, and I'll find the perfect match!"

    return jsonify({"reply": reply, "movies": movies, "detected_mood": mood, "intent": intent})


# ── RATINGS ───────────────────────────────────────────────────────────────────
@app.route("/api/ratings", methods=["POST"])
def rate_movie():
    body     = request.get_json() or {}
    user_id  = body.get("user_id", "user_1")
    movie_id = body.get("movie_id")
    rating   = float(body.get("rating", 3.0))
    if movie_id:
        user_ratings[user_id][movie_id] = rating
        return jsonify({"success": True, "message": f"Rated movie {movie_id}: {rating}/5"})
    return jsonify({"success": False, "message": "movie_id required"}), 400

@app.route("/api/ratings/<user_id>")
def get_ratings(user_id):
    return jsonify({"user_id": user_id, "ratings": user_ratings.get(user_id, {})})


# ── GENRES ────────────────────────────────────────────────────────────────────
@app.route("/api/genres")
def genres():
    return jsonify({"genres": [{"id": k, "name": v} for k, v in GENRE_MAP.items()]})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
