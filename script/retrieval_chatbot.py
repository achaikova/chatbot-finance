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
from llama_index.core.query_engine import SubQuestionQueryEngine, RetrieverQueryEngine
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.utils import truncate_text
from llama_index.core.schema import MetadataMode, NodeWithScore
# from llama_index.llms import Ollama

from langchain_community.llms import LlamaCpp
from langchain_community.chat_models import ChatOpenAI

import subprocess
import threading
import time
import os

import chromadb
import nest_asyncio
import time
from configuration import MODEL_PATH, CHROMA_PATH, CHROMA_COLLECTION, MODEL_NAME

nest_asyncio.apply()

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Using the LlamaDebugHandler to print the trace of the sub questions
# captured by the SUB_QUESTION callback event type
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])

Settings.callback_manager = callback_manager
Settings.embed_model = embed_model

ROOT_LOG_LEVEL = "DEBUG"

PRETTY_LOG_FORMAT = (
    "%(asctime)s.%(msecs)03d [%(levelname)-8s] %(name)+25s - %(message)s"
)
logging.basicConfig(level=ROOT_LOG_LEVEL, format=PRETTY_LOG_FORMAT, datefmt="%H:%M:%S")
logging.captureWarnings(True)


def load_automerging_index():
    print("Loading Index...")
    service_context = ServiceContext.from_defaults(
        embed_model=embed_model,
        llm=None
    )
    # load from disk
    db2 = chromadb.PersistentClient(path=CHROMA_PATH)
    chroma_collection = db2.get_or_create_collection(CHROMA_COLLECTION)
    chroma_database = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=chroma_database, persist_dir=CHROMA_PATH)

    # storage_context = StorageContext.from_defaults(persist_dir=DB_FAISS_PATH)
    index = load_index_from_storage(storage_context=storage_context, service_context=service_context)
    # index = VectorStoreIndex.from_vector_store(vector_store, service_context=service_context, storage_context=storage_context)
    return index


# @cl.cache
# def instantiate_llm():
#     print("Loading LLM...")
#     n_batch = (
#         512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
#     )
#     llm = LlamaCpp(
#         model_path=MODEL_PATH,
#         n_gpu_layers=-1,  # in case of Metal -- 1
#         n_batch=n_batch,
#         n_ctx=2048,
#         f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
#         verbose=True,
#         streaming=True,
#         # mlock=True
#     )
#     return llm

def run_fastchat_controller():
    subprocess.run(['python3', '-m', 'fastchat.serve.controller'])


def run_fastchat_model():
    subprocess.run(['python3', '-m', 'fastchat.serve.model_worker', '--model-name', MODEL_NAME, '--model-path',
                    'lmsys/vicuna-7b-v1.5', "--load-8bit", "--device", "mps"])


def run_fastchat_api_server():
    subprocess.run(['python3', '-m', 'fastchat.serve.openai_api_server', '--host', 'localhost', '--port', '8888'])


@cl.cache
def instantiate_llm():
    os.environ["OPENAI_API_BASE"] = "http://localhost:8888/v1"
    os.environ["OPENAI_API_KEY"] = "empty"

    fastchat_controller_thread = threading.Thread(target=run_fastchat_controller)
    fastchat_model_thread = threading.Thread(target=run_fastchat_model)
    fastchat_api_server_thread = threading.Thread(target=run_fastchat_api_server)
    # Start the thread
    fastchat_controller_thread.start()
    fastchat_model_thread.start()
    fastchat_api_server_thread.start()

    time.sleep(30)
    llm = ChatOpenAI(model=MODEL_NAME)
    return llm


llm = instantiate_llm()
Settings.llm = llm


@cl.cache
def get_automerging_query_engine(
        # llm,
        automerging_index,
        similarity_top_k=12,
        rerank_top_n=6,
        mmr_threshold=0.6
):
    print("Loading Query Engine...")
    base_retriever = automerging_index.as_retriever(
        # vector_store_query_mode="mmr",
        similarity_top_k=similarity_top_k,
        # vector_store_kwargs={"mmr_threshold": mmr_threshold},
    )
    # base_retriever = automerging_index.as_retriever(similarity_top_k=similarity_top_k)
    retriever = AutoMergingRetriever(
        base_retriever, automerging_index.storage_context, verbose=True
    )
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n
        # , model="BAAI/bge-reranker-base"
    )
    merging_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
    )
    auto_merging_engine = RetrieverQueryEngine.from_args(
        retriever, node_postprocessors=[rerank], service_context=merging_context, streaming=True, use_async=True
    )

    # query_engine_tools = [
    #     QueryEngineTool(
    #         query_engine=auto_merging_engine,
    #         metadata=ToolMetadata(
    #             name="pg_essay",
    #             description="Paul Graham essay on What I Worked On",
    #         ),
    #     ),
    # ]
    # query_engine = SubQuestionQueryEngine.from_defaults(
    #     query_engine_tools=query_engine_tools,
    #     use_async=True,
    # )
    return retriever, auto_merging_engine


def display_node(source_node,
                 source_length: int = 100,
                 show_source_metadata: bool = False,
                 metadata_mode: MetadataMode = MetadataMode.NONE):
    source_text_fmt = truncate_text(
        source_node.node.get_content(metadata_mode=metadata_mode).strip(), source_length
    )
    text_md = (
        f"**Node ID:** {source_node.node.node_id}<br>"
        f"**Similarity:** {source_node.score}<br>"
        f"**Text:** {source_text_fmt}<br>"
    )
    if show_source_metadata:
        text_md += f"**Metadata:** {source_node.node.metadata}<br>"
    logging.info(f"{text_md}")


def display_nodes(nodes):
    for node in nodes:
        display_node(node, source_length=10000)


def get_source_nodes(retriever, query):
    nodes = retriever.retrieve(query)
    logging.info(f"Length of nodes from automerging retriever: {len(nodes)}")
    logging.info("Nodes from automerging retriever:")
    display_nodes(nodes)


def factory():
    print("Creating factory...")
    index = load_automerging_index()
    retriever, automerging_query_engine = get_automerging_query_engine(index)
    return retriever, automerging_query_engine


@cl.on_chat_start
async def initialize_bot():
    print("Initializing bot...")
    welcome_message = cl.Message(content="Starting the bot...")
    await welcome_message.send()
    welcome_message.content = (
        "Hi, Welcome to Chat With Financial Documents using Llama2 and LangChain."
    )
    await welcome_message.update()
    retriever, query_engine = factory()
    cl.user_session.set("retriever", retriever)
    cl.user_session.set("query_engine", query_engine)


@cl.on_message
async def main(message: cl.Message):
    query_engine = cl.user_session.get("query_engine")
    retriever = cl.user_session.get("retriever")
    get_source_nodes(retriever, message.content)
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

# """
# Behave as a professor on a chair of Finance. Answer the question given based on this text:
# NewRocket GmbH is an aerospace company founded by two researchers from the Technical University of Munich. Currently, the company employs 18 people, including the founders. To finance the first small batch production line, the founders decide to raise their seed round from professional investors. To cover the expected expenses of the next 24 months, the founding team decides to raise 6 Mn. € of external capital. After several weeks of close exchange and negotiations with potential investors, the single seed investor AlphaStar is willing to sign a term sheet at a 18 Mn. € premoney valuation. The 30.000 founding shares are distributed among the founders according to the initial capital they've provided to fund the first months of research and product development prior to the seed round. Since the second founder provided only a third of the initial capital, 10.000 of the founding shares are assigned to her. Eighteen months after raising their seed round, the team has grown to 38 people and is eager to start the serial production of their production line. To fund the upcoming cost, the founders decide to raise a series A financing round. After assessing multiple term sheets, the founders sign a term sheet with Betaplus Capital over an investment amount of 12 Mn. € at a 40 Mn. € pre-money valuation. Now assume NewBios' efforts after the seed round wouldn't have been successful. Therefore, to save the company, instead of the A-round, they would have raised a down round from Delta Invest of 4 Mn. € at a pre-money valuation of 10 Mn. €. Moreover, assume in this case that AlphaStar has negotiated a weighted average ratchet anti-dilution protection.
#
# Which rate does the Venture Capital Method apply to discount the future value of the investment?
#
# The return of high yield bonds.
# The venture capitalists target rate of return.
# The risk free rate.
# The return of a large cap market index.
# """
