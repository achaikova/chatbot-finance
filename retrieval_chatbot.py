import chainlit as cl
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index import StorageContext, load_index_from_storage
from llama_index.retrievers import AutoMergingRetriever
from llama_index.indices.postprocessor import SentenceTransformerRerank
from llama_index.query_engine import RetrieverQueryEngine
from llama_index import (
    ServiceContext,
)
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp

from configuration import DB_FAISS_PATH, MODEL_PATH

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


def load_automerging_index():
    service_context = ServiceContext.from_defaults(
        embed_model=embed_model,
        llm=None
    )
    storage_context = StorageContext.from_defaults(persist_dir=DB_FAISS_PATH)
    index = load_index_from_storage(storage_context=storage_context, service_context=service_context)
    return index


def load_llm():
    n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    # # Make sure the model path is correct for your system!
    return LlamaCpp(
        model_path=MODEL_PATH,
        n_gpu_layers=-1,  # in case of Metal -- 1
        n_batch=n_batch,
        n_ctx=2048,
        f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
        callback_manager=callback_manager,
        verbose=True,
        streaming=True
    )
    #
    # return LlamaCPP(
    #     model_path=MODEL_PATH,
    #     # temperature=0.1,
    #     # max_new_tokens=256,
    #     # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    #     context_window=2048,
    #     # kwargs to pass to __call__()
    #     generate_kwargs={},
    #     # kwargs to pass to __init__()
    #     # set to at least 1 to use GPU
    #     model_kwargs={"n_gpu_layers": -1, "f16_kv": True, "n_batch": n_batch},
    #     # transform inputs into Llama2 format
    #     messages_to_prompt=messages_to_prompt,
    #     completion_to_prompt=completion_to_prompt,
    #     verbose=True,
    # )


def get_automerging_query_engine(
        llm,
        automerging_index,
        similarity_top_k=6,
        rerank_top_n=6,
):
    base_retriever = automerging_index.as_retriever(similarity_top_k=similarity_top_k)
    retriever = AutoMergingRetriever(
        base_retriever, automerging_index.storage_context, verbose=True
    )
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )
    merging_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
    )
    auto_merging_engine = RetrieverQueryEngine.from_args(
        retriever, node_postprocessors=[rerank], service_context=merging_context
    )
    return auto_merging_engine


def factory():
    index = load_automerging_index()
    llm = load_llm()
    query_engine = get_automerging_query_engine(llm, index)
    # memory = ConversationBufferMemory(memory_key="chat_history")
    # tools = [
    #     Tool(
    #         name="LlamaIndex",
    #         func=lambda q: str(index.as_query_engine().query(q)),
    #         description="Useful for when you want to answer questions about the documents.",
    #         return_direct=True,
    #     ),
    # ]
    # agent_executor = initialize_agent(
    #     tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, memory=memory
    # )
    return query_engine


@cl.on_chat_start
async def initialize_bot():
    query_engine = factory()
    welcome_message = cl.Message(content="Starting the bot...")
    await welcome_message.send()
    welcome_message.content = (
        "Hi, Welcome to Chat With Financial Documents using Llama2 and LangChain."
    )
    await welcome_message.update()

    cl.user_session.set("query_engine", query_engine)


@cl.on_message
async def main(message: cl.Message):
    query_engine = cl.user_session.get("query_engine")  # type: RetrieverQueryEngine
    response = await cl.make_async(query_engine.query)(message.content)

    response_message = cl.Message(content="")

    for token in response.response_gen:
        await response_message.stream_token(token=token)

    if response.response_txt:
        response_message.content = response.response_txt

    await response_message.send()