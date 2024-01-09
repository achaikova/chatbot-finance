# Financial Q/A Chatbot

## Overview

This project is a chatbot designed for question and answer interactions over financial data, including articles, slides from lectures, and more.

## Installation

Create a virtualenv and activate it:
```bash
python3 -m venv .venv && source .venv/bin/activate

```
To get started, ensure you have the required packages installed. You can do this by running the following command:

```bash
pip install -r requirements.txt
```

### Model Download

To use the chatbot, you need to download the llama model. Run the following command to download the model:
```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python
wget "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q5_K_S.gguf?download=true" -P <MODEL_PATH>
```

### Configuration

Before running the chatbot, specify the necessary paths for model data, financial data, and Faiss database. Update the following variables in the code:

- `MODEL_PATH`: path where the llama model is stored.
- `DATA_PATH`: path to financial data (articles, slides, etc.).
- `DB_FAISS_PATH`: path to store the Faiss database.