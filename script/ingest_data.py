from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext, StorageContext,
)
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.node_parser import HierarchicalNodeParser, get_leaf_nodes

from script.configuration import DB_FAISS_PATH, DATA_PATH



def build_automerging_index(
    embed_model="local:BAAI/bge-small-en-v1.5",
    chunk_sizes=None,
):
    documents = SimpleDirectoryReader(DATA_PATH).load_data()

    chunk_sizes = chunk_sizes or [1024, 512, 128]
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)
    merging_context = ServiceContext.from_defaults(
        llm=None,
        embed_model=embed_model,
    )
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)
    automerging_index = VectorStoreIndex(
        leaf_nodes, storage_context=storage_context, service_context=merging_context
    )
    automerging_index.storage_context.persist(persist_dir=DB_FAISS_PATH)
    return automerging_index



def create_vector_database_langchain():
    loader = PyPDFDirectoryLoader(DATA_PATH)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024 , chunk_overlap=100)
    texts = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')

    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(DB_FAISS_PATH)

def create_vector_database_llamaindex():
    # if not os.path.exists(DB_FAISS_PATH):
        # load the documents and create the index
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    service_context = ServiceContext.from_defaults(
        embed_model=embed_model,
        chunk_size=1024,
        chunk_overlap=100,
        llm=None
    )
    documents = SimpleDirectoryReader(DATA_PATH).load_data()
    index = VectorStoreIndex.from_documents(documents=documents, service_context=service_context)
    # store it for later
    index.storage_context.persist(DB_FAISS_PATH)
    # else:
        # # load the existing index
        # storage_context = StorageContext.from_defaults(persist_dir=DB_FAISS_PATH)
        # index = load_index_from_storage(storage_context)

if __name__ == "__main__":
    build_automerging_index()