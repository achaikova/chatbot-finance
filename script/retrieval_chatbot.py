import chainlit as cl

from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
    load_index_from_storage
)
import logging
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.indices.postprocessor import SentenceTransformerRerank
from llama_index.core.base.response.schema import Response, StreamingResponse
from llama_index.vector_stores.chroma import ChromaVectorStore

from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.utils import truncate_text

from langchain.chat_models import ChatOpenAI

from langchain_community.llms import LlamaCpp
from langchain_community.chat_models import ChatOpenAI

import subprocess
import threading
import time
import os
import openai

import chromadb
import nest_asyncio
import time
from configuration import MODEL_PATH, CHROMA_PATH, CHROMA_COLLECTION, MODEL_NAME

nest_asyncio.apply()

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

Settings.embed_model = embed_model

ROOT_LOG_LEVEL = "DEBUG"

PRETTY_LOG_FORMAT = (
    "%(asctime)s.%(msecs)03d [%(levelname)-8s] %(name)+25s - %(message)s"
)
logging.basicConfig(level=ROOT_LOG_LEVEL, format=PRETTY_LOG_FORMAT, datefmt="%H:%M:%S")
logging.captureWarnings(True)


def load_automerging_index():
    """Load the persisted Chroma database"""
    logging.info("Loading Index")
    service_context = ServiceContext.from_defaults(
        embed_model=embed_model,
        llm=None
    )
    db2 = chromadb.PersistentClient(path=CHROMA_PATH)
    chroma_collection = db2.get_or_create_collection(CHROMA_COLLECTION)
    chroma_database = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=chroma_database, persist_dir=CHROMA_PATH)

    index = load_index_from_storage(storage_context=storage_context, service_context=service_context)
    return index


@cl.cache
def instantiate_llm(): # comment the function if you want to use fastchat
    """Serve the Llama2 model using llama-cpp"""
    print("Loading LLM...")
    n_batch = (
        512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    )
    llm = LlamaCpp(
        model_path=MODEL_PATH,
        n_gpu_layers=-1,  # in case of Metal -- 1
        n_batch=n_batch,
        n_ctx=2048,
        f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
        verbose=True,
        streaming=True,
        # mlock=True
    )
    return llm


def run_fastchat_controller():
    subprocess.run(['python3', '-m', 'fastchat.serve.controller'])


def run_fastchat_model():
    subprocess.run(['python3', '-m', 'fastchat.serve.model_worker', '--model-name', MODEL_NAME, '--model-path',
                    "lmsys/vicuna-7b-v1.5"])


def run_fastchat_api_server():
    subprocess.run(['python3', '-m', 'fastchat.serve.openai_api_server', '--host', 'localhost', '--port', '8888'])


# @cl.cache
# def instantiate_llm(): # uncomment the function if you want to use fastchat
#     """Serve the local LLM using fastchat"""
#     openai.api_base = f"http://localhost:8888/v1"
#     openai.api_key = "EMPTY"
#
#     fastchat_controller_thread = threading.Thread(target=run_fastchat_controller)
#     fastchat_model_thread = threading.Thread(target=run_fastchat_model)
#     fastchat_api_server_thread = threading.Thread(target=run_fastchat_api_server)
#
#     fastchat_controller_thread.start()
#     fastchat_model_thread.start()
#     fastchat_api_server_thread.start()
#
#     time.sleep(30)
#     llm = ChatOpenAI(model=MODEL_NAME)
#     return llm


llm = instantiate_llm()
Settings.llm = llm


@cl.cache
def get_automerging_query_engine(
        automerging_index,
        similarity_top_k=12,
        rerank_top_n=6,
):
    """Create query engine from index"""
    logging.info("Loading Query Engine")
    base_retriever = automerging_index.as_retriever(
        similarity_top_k=similarity_top_k,
    )
    retriever = AutoMergingRetriever(
        base_retriever, automerging_index.storage_context, verbose=True
    )
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n
    )
    merging_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
    )
    auto_merging_engine = RetrieverQueryEngine.from_args(
        retriever, node_postprocessors=[rerank], service_context=merging_context, streaming=True, use_async=True
    )
    return retriever, auto_merging_engine


def factory():
    logging.info("Creating factory")
    index = load_automerging_index()
    retriever, automerging_query_engine = get_automerging_query_engine(index)
    return retriever, automerging_query_engine


@cl.on_chat_start
async def initialize_bot():
    logging.info("Initializing bot")
    welcome_message = cl.Message(content="Starting the bot...")
    await welcome_message.send()
    welcome_message.content = (
        "Welcome to Chat With Financial Documents using LlamaIndex."
    )
    await welcome_message.update()
    _, query_engine = factory()
    cl.user_session.set("query_engine", query_engine)


@cl.on_message
async def main(message: cl.Message):
    query_engine = cl.user_session.get("query_engine")

    response = await cl.make_async(query_engine.query)(message.content)
    response_message = cl.Message(content="")

    if isinstance(response, Response):
        response_message.content = str(response)
        await response_message.update()
    elif isinstance(response, StreamingResponse):
        gen = response.response_gen
        for token in gen:
            await response_message.stream_token(token=token)
        if response.response_txt:
            response_message.content = response.response_txt
        await response_message.update()

    logging.info(f"source_nodes: {[i.text for i in response.source_nodes]}")
