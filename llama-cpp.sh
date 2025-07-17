mkdir -p ~/LLM/models
wget -P ~/LLM/models \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf

docker run --name llama-server \
  -p 8080:8080 \
  -v "$HOME/LLM/models:/models" \
  ghcr.io/ggml-org/llama.cpp:server \
  -m /models/qwen2.5-0.5b-instruct-q4_k_m.gguf

