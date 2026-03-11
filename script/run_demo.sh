#!/usr/bin/env bash

echo "=============================================="
echo " Intent Decomposition Project - Demo Runner"
echo "=============================================="
echo ""


cd "$(dirname "$0")" || exit 1

# Activate environment if exists
if [ -d "myenv" ]; then
  echo "Activating virtual environment: myenv"

  source myenv/bin/activate
else
  echo " No virtual environment found (myenv). Running with system Python."
fi

echo ""
echo " Checking required files..."
if [ ! -f "data/cluster_centroids.npy" ]; then
  echo " ERROR: data/cluster_centroids.npy not found!"
  echo "Please ensure centroids file exists inside data/ folder."
  exit 1
fi

echo " All required files found."
echo ""

echo "Choose demo mode:"
echo "1) Run CLI Chatbot"
echo "2) Run Streamlit Frontend"
echo ""

read -r -p "Enter choice (1/2): " choice

if [ "$choice" = "1" ]; then
  echo ""
  echo " Starting CLI Chatbot..."
  python3 phase6_chatbot.py

elif [ "$choice" = "2" ]; then
  echo ""
  echo " Starting Streamlit Frontend..."
  echo "Open in browser: http://localhost:8501"
  streamlit run frontend_app.py

else
  echo " Invalid choice. Please enter 1 or 2."
  exit 1
fi
