# Databricks notebook source
# MAGIC %sql
# MAGIC USE CATALOG uc_demos_chris_crawford;
# MAGIC USE SCHEMA mydata;

# COMMAND ----------

# MAGIC %pip install -U protobuf==3.20.1 beautifulsoup4==4.11.1 transformers==4.30.2 langchain faiss-cpu pypdf unstructured sentence_transformers accelerate pdfminer.six pdf2image chromadb opencv-python tesseract unstructured_pytesseract unstructured_inference

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# import
from langchain.document_loaders import TextLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders import PyPDFLoader

# load the document and split it into chunks
#loader = TextLoader("/Volumes/uc_demos_chris_crawford/mydata/llm_pdfs/LearningSpark2.0.pdf")
loader = PyPDFLoader("/Volumes/uc_demos_chris_crawford/mydata/llm_pdfs/LearningSpark2.0.pdf")
documents = loader.load_and_split()

# split it into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# load it into Chroma
db = Chroma.from_documents(docs, embedding_function)

# query it
query = "What is MLFLow?"
docs = db.similarity_search(query)

# print results
print(docs[0].page_content)

# COMMAND ----------


