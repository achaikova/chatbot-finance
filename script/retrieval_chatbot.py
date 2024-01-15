import chainlit as cl
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index import StorageContext, load_index_from_storage
from llama_index.retrievers import AutoMergingRetriever
from llama_index.indices.postprocessor import SentenceTransformerRerank
from llama_index.query_engine import RetrieverQueryEngine
from llama_index import (
    ServiceContext,
)
from llama_index.response.schema import Response, StreamingResponse
from langchain_community.llms import LlamaCpp

from configuration import DB_FAISS_PATH, MODEL_PATH

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


def load_automerging_index():
    print("Loading Index...")
    service_context = ServiceContext.from_defaults(
        embed_model=embed_model,
        llm=None
    )
    storage_context = StorageContext.from_defaults(persist_dir=DB_FAISS_PATH)
    index = load_index_from_storage(storage_context=storage_context, service_context=service_context)
    return index


@cl.cache
def instantiate_llm():
    print("Loading LLM...")
    n_batch = (
        1024  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    )
    llm = LlamaCpp(
        model_path=MODEL_PATH,
        n_gpu_layers=1,  # in case of Metal -- 1
        n_batch=n_batch,
        n_ctx=2048,
        f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
        verbose=True,
        streaming=True
    )
    return llm


llm = instantiate_llm()

@cl.cache
def get_automerging_query_engine(
        # llm,
        automerging_index,
        similarity_top_k=6,
        rerank_top_n=6,
):
    print("Loading Query Engine...")
    base_retriever = automerging_index.as_retriever(similarity_top_k=similarity_top_k)
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
        retriever, node_postprocessors=[rerank], service_context=merging_context, streaming=True
    )
    return auto_merging_engine


def factory():
    print("Creating factory...")
    index = load_automerging_index()
    automerging_query_engine = get_automerging_query_engine(index)
    return automerging_query_engine


query_engine = factory()


@cl.on_chat_start
async def initialize_bot():
    print("Initializing bot...")
    welcome_message = cl.Message(content="Starting the bot...")
    await welcome_message.send()
    welcome_message.content = (
        "Hi, Welcome to Chat With Financial Documents using Llama2 and LangChain."
    )
    await welcome_message.update()

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
