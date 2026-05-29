#!/bin/bash
# CineMind AI - Quick Start Script

echo "🎬 CineMind AI Setup"
echo "===================="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

echo "✅ Python found: $(python3 --version)"

# Backend setup
echo ""
echo "📦 Setting up backend..."
cd backend

if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "⚠️  Created .env file — please add your TMDB_API_KEY!"
    echo "   Get a free key at: https://www.themoviedb.org/settings/api"
fi

python3 -m venv venv 2>/dev/null && echo "✅ Virtual environment created"
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null

pip install -r requirements.txt -q && echo "✅ Dependencies installed"

echo ""
echo "🚀 Starting CineMind API on http://localhost:5000"
echo "   Open frontend/public/index.html in your browser"
echo "   (Remember to set TMDB_API_KEY in frontend/public/index.html too)"
echo ""

python app.py
