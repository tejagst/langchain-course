import os
from dotenv import load_dotenv

from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == '__main__':
    print("Ingesting...")
    loader = UnstructuredLoader("C:\\Users\\sivat\\Projects\\langchain-course\\mediumblog1.txt")
    document = loader.load()

    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    print("ingesting to pinecone...")
    PineconeVectorStore.from_documents(
        texts,
        embeddings,
        index_name=os.environ.get("INDEX_NAME")
    )
    print("finished ingesting to pinecone")
