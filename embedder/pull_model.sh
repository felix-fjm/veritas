#!/bin/sh
# Start the Ollama server in the background, then pull the embedding model.
# The server must be running before `ollama pull` can be issued.
ollama serve &
SERVER_PID=$!

# Wait until the server is accepting requests
echo "Waiting for Ollama server to start..."
until curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; do
  sleep 1
done

echo "Pulling nomic-embed-text:v1.5..."
ollama pull nomic-embed-text:v1.5

echo "Model ready. Keeping server in foreground."
wait $SERVER_PID
