import chromadb
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext, StorageContext,
)
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.vector_stores.chroma import ChromaVectorStore
import time
import logging
from configuration import CHROMA_COLLECTION, CHROMA_PATH, DATA_PATH

ROOT_LOG_LEVEL = "DEBUG"
PRETTY_LOG_FORMAT = (
    "%(asctime)s.%(msecs)03d [%(levelname)-8s] %(name)+25s - %(message)s"
)
logging.basicConfig(level=ROOT_LOG_LEVEL, format=PRETTY_LOG_FORMAT, datefmt="%H:%M:%S")
logging.captureWarnings(True)

def build_automerging_index(
        embed_model="local:BAAI/bge-small-en-v1.5",
        chunk_sizes=None,
):
    start = time.time()
    documents = SimpleDirectoryReader(DATA_PATH).load_data()
    read_documents = time.time()
    logging.info(f'Time taken to load docs: {read_documents - start}')
    db = chromadb.PersistentClient(path=CHROMA_PATH)
    chroma_collection = db.get_or_create_collection(CHROMA_COLLECTION)

    chunk_sizes = chunk_sizes or [2048, 512, 128]
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)
    g_leaf_nodes = time.time()
    logging.info(f'Time taken to get leaf nodes: {g_leaf_nodes - read_documents}')
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    merging_context = ServiceContext.from_defaults(
        llm=None,
        embed_model=embed_model,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    storage_context.docstore.add_documents(nodes)
    add_documents = time.time()
    logging.info(f'Time taken to get add documents: {add_documents - g_leaf_nodes}')
    automerging_index = VectorStoreIndex(
        leaf_nodes, storage_context=storage_context, service_context=merging_context
    )
    automerging_index.storage_context.persist(persist_dir=CHROMA_PATH)
    # return automerging_index


if __name__ == "__main__":
    build_automerging_index()
