# Financial Q/A Chatbot

## Overview

This project is a chatbot designed for question and answer interactions over financial data. At the moment chat history
is not supported.

## Installation

Create a virtualenv and activate it:

```bash
python3.10 -m venv myenv && source myenv/bin/activate # using venv
conda create -n myenv python=3.10 && conda activate myenv # or using conda

```

To get started, ensure you have the required packages installed. You can do this by running the following command:

```bash
pip install -r requirements.txt
```

### Preprocessing Data

To parse articles in pdf format use [nougat](https://github.com/facebookresearch/nougat). Ensure you run it on GPU, e.g.
in [colab](https://colab.research.google.com/drive/1OBhH3Nf2G3nFzZ963HzStUL9iN0PiKTn?usp=sharing).
The preprocessed provided articles are available in `data_mmd`.

### Configuration

Before running the chatbot, make sure the necessary paths for model data, preprocessed data, and Chroma database are
correct. Update the following variables in the `script/configuration.py`:

- `MODEL_PATH`: path where the llama model is stored.
- `DATA_PATH`: path to the preprocessed data.
- `CHROMA_PATH`: path to store the Chroma database.

Export environment variables:

```angular2html
export MODEL_PATH=$(python -c "from configuration import MODEL_PATH; print(MODEL_PATH)")
export MODEL_DOWNLOAD_URL=$(python -c "from configuration import MODEL_DOWNLOAD_URL; print(MODEL_DOWNLOAD_URL)")
```

### Model Download

To use the chatbot, you need to download the llama model. Run the following command to download the model:

```bash 
wget $MODEL_DOWNLOAD_URL --output-document=$MODEL_PATH
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python # if you want to run via llama-cpp-python, otherwise use [fastchat](https://github.com/lm-sys/FastChat)
```

Llama2 is used as an example due to hardware limitations. In case of fastchat use one of the [supported models](https://github.com/lm-sys/FastChat/blob/main/README.md#supported-models), e.g. Vicuna.

### Ingesting Data

Run the script to ingest preprocessed data into the local database:

```bash
cd script/
python ingest_data.py
```

### Running the Server

Once the database is populated, run the server for the retrieval chatbot:

```bash
chainlit run retrieval_chatbot.py -w
```

Access the chatbot via the specified endpoint