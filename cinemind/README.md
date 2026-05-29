# 🎬 CineMind AI — Smart Movie Recommendation System

A full-stack AI-powered movie recommendation platform with mood detection, collaborative filtering, content-based filtering, explainable AI, and a semantic chatbot.

---

## ✨ Features

| Module | Description | Status |
|--------|-------------|--------|
| **Mood Detection** | NLP text classifier maps feelings → genres | ✅ |
| **Content-Based Filtering** | TF-IDF cosine similarity on movie overviews | ✅ |
| **Collaborative Filtering** | User-based CF with cosine similarity | ✅ |
| **Hybrid Scoring** | `0.5·User + 0.3·Genre + 0.2·Trend` formula | ✅ |
| **Explainable AI** | Every recommendation has a human-readable reason | ✅ |
| **AI Chatbot** | Semantic intent detection + movie search | ✅ |
| **TMDb Integration** | Live movie data, posters, cast, similar films | ✅ |
| **Star Ratings** | Users rate movies, feeds into CF | ✅ |
| **Direct TMDB Fallback** | Frontend works without backend via direct API | ✅ |

---

## 🏗️ Architecture

```
CineMind/
├── backend/
│   ├── app.py              # Flask API (all routes)
│   ├── requirements.txt    # Python dependencies
│   └── .env.example        # Environment variables
└── frontend/
    └── public/
        └── index.html      # Complete single-file React-like SPA
```

### Hybrid Scoring Formula

```
HybridScore = 0.5 × UserScore + 0.3 × GenreScore + 0.2 × TrendScore

Where:
  UserScore  = user's historical rating / 5.0  (cold start: 0.6)
  GenreScore = TMDb vote_average / 10.0
  TrendScore = 1 - (trending_rank / 50)
```

### Mood → Genre Mapping

| Mood | Primary Genres | Secondary |
|------|---------------|-----------|
| Happy | Comedy, Family | Animation |
| Sad | Drama, Romance | — |
| Excited | Action, Adventure | Sci-Fi |
| Scared | Horror, Thriller | Mystery |
| Romantic | Romance, Comedy | — |
| Thoughtful | Drama, Sci-Fi | Mystery |
| Bored | Action, Comedy | Adventure |
| Relaxed | Documentary, Family | History |
| Angry | Action, Crime | Thriller |
| Nostalgic | History, Music | Family |
| Curious | Sci-Fi, Mystery | Documentary |
| Lonely | Romance, Drama | Family |

---

## 🚀 Quick Start

### Option A — Frontend Only (no backend needed)

1. Get a free API key from [TMDb](https://www.themoviedb.org/settings/api)
2. Open `frontend/public/index.html`
3. Replace `YOUR_TMDB_API_KEY` with your key
4. Open in browser — **done!**

```html
<!-- Line ~390 in index.html -->
const TMDB_KEY = 'your_actual_key_here';
```

### Option B — Full Stack

#### Backend Setup

```bash
cd backend
cp .env.example .env
# Edit .env with your TMDB_API_KEY

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
# → Running on http://localhost:5000
```

#### Frontend

```bash
# Just open the file, or serve it:
cd frontend/public
python -m http.server 3000
# → http://localhost:3000
```

---

## 🛣️ API Reference

### Movies

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/movies/trending` | Trending this week |
| GET | `/api/movies/popular?page=1` | Popular movies |
| GET | `/api/movies/top_rated` | Top rated |
| GET | `/api/movies/{id}` | Movie detail + cast + similar |
| GET | `/api/movies/search?q=inception` | Search |
| GET | `/api/movies/genre/{genre_id}` | By genre |

### Recommendations

| Method | Endpoint | Body | Description |
|--------|----------|------|-------------|
| POST | `/api/recommend/mood` | `{mood, text, user_id}` | Mood-based recs |
| GET | `/api/recommend/content/{id}` | — | Content-based similar |
| POST | `/api/recommend/collaborative` | `{user_id, n}` | User-based CF |
| POST | `/api/recommend/hybrid` | `{user_id, mood, genres, n}` | Full hybrid |

### Chat & Ratings

| Method | Endpoint | Body |
|--------|----------|------|
| POST | `/api/chat` | `{message, history}` |
| POST | `/api/ratings` | `{user_id, movie_id, rating}` |
| GET | `/api/ratings/{user_id}` | — |

---

## 📊 Explainability

Every recommendation includes a human-readable explanation:

```json
{
  "title": "Interstellar",
  "hybrid_score": 0.847,
  "explanation": "Matches your preferred genres: Sci-Fi · Drama · Highly rated (8.6/10)",
  "score_breakdown": {
    "user": 0.72,
    "genre": 0.86,
    "trend": 0.80
  }
}
```

---

## 🗺️ Build Order (Solo Dev Roadmap)

1. ✅ **Week 1-2**: Static movie browser + TMDb integration
2. ✅ **Week 3-4**: Content-based filtering (TF-IDF cosine similarity)
3. ✅ **Week 5-6**: Mood → genre mapping (NLP keyword matching)
4. ✅ **Week 7-8**: Chatbot with intent detection
5. 🔄 **Week 9-10**: Collaborative filtering (needs user ratings data)
6. 🔮 **Week 11+**: Face/voice emotion input (DeepFace + SpeechRecognition)

---

## 🔮 Future Modules

### Face Emotion Detection
```python
# pip install deepface opencv-python
from deepface import DeepFace
result = DeepFace.analyze(img_path='face.jpg', actions=['emotion'])
dominant_emotion = result[0]['dominant_emotion']
# → maps to mood → genre
```

### Voice Input
```python
# pip install SpeechRecognition
import speech_recognition as sr
r = sr.Recognizer()
with sr.Microphone() as source:
    audio = r.listen(source)
text = r.recognize_google(audio)
# → detect_mood_from_text(text)
```

### Sentence Embeddings Chatbot
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
query_emb = model.encode("something romantic and sad")
movie_embs = model.encode([m['overview'] for m in movies])
sims = cosine_similarity([query_emb], movie_embs)[0]
```

---

## 🎨 Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Vanilla JS SPA (no build step required) |
| Styling | Custom CSS with CSS variables |
| Backend | Python Flask + Flask-CORS |
| ML/NLP | scikit-learn, sentence-transformers |
| Data | TMDb API |
| Fonts | Playfair Display + DM Sans |

---

## 📝 License

MIT — build freely, credit appreciated.
