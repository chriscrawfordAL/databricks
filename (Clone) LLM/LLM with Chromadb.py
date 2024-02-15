# Databricks notebook source
# MAGIC %pip install -U protobuf==3.20.1 beautifulsoup4==4.11.1 transformers==4.30.2 langchain faiss-cpu pypdf unstructured sentence_transformers accelerate pdfminer.six pdf2image chromadb opencv-python tesseract unstructured_pytesseract unstructured_inference

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from langchain.document_loaders import UnstructuredPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
import os
from langchain.document_loaders import TextLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

# COMMAND ----------

source_pdf_dir = "/Volumes/uc_demos_chris_crawford/mydata/llm_pdfs/"
vector_db_dir = "/Volumes/uc_demos_chris_crawford/mydata/llm_vectordb/"
embedding_model_dir = "/Volumes/uc_demos_chris_crawford/mydata/llm_embedded_modeling_dir/"

# COMMAND ----------

display(dbutils.fs.ls(source_pdf_dir))

# COMMAND ----------

pdf_paths = [os.path.join(source_pdf_dir, file) for file in os.listdir(source_pdf_dir)]

documents = []

#for p in pdf_paths:
#  loader = UnstructuredPDFLoader(p, mode="elements", strategy="fast")
#  data = loader.load()
#  docs.extend(data)

for p in pdf_paths:
  loader = PyPDFLoader(p)
  data = loader.load_and_split()
  documents.extend(data)

documents[0]

# COMMAND ----------

len(documents)

# COMMAND ----------

documents[:3]

# COMMAND ----------

import pandas as pd
categories = pd.Series([d.metadata.get('category') for d in docs])
categories.value_counts()

# COMMAND ----------

# Are the `Title` chunks useful?
[d.page_content for d in docs if d.metadata.get('category')=='Title'][:20]

# COMMAND ----------

# Are the `UncategorizedText` chunks useful?
[d.page_content for d in docs if d.metadata.get('category')=='UncategorizedText'][:20]

# COMMAND ----------

# Are the `ListItem` chunks useful?
[d.page_content for d in docs if d.metadata.get('category')=='ListItem'][:20]

# COMMAND ----------

# MAGIC %md Let's filter the documents down to only those with element categories `NarrativeText` and `ListItem`. 

# COMMAND ----------

filtered_docs = [d for d in docs if d.metadata.get('category') in ['NarrativeText', 'ListItem']]
len(filtered_docs)

# COMMAND ----------

# MAGIC %md We also want to make sure that we don't have text chunks that will be too large to pass into the LLM context. Having predictable chunk sizes is helpful when setting the number of sources to retrieve as context. So we'll use another langchain utility - `TextSplitter`.

# COMMAND ----------

from langchain.text_splitter import RecursiveCharacterTextSplitter

# split it into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# COMMAND ----------

len(docs)

# COMMAND ----------

# MAGIC %md The text splitter settings we used created about 2000 more documents than before, but also helps us ensure that the LLM will not have to truncate the source documents passed to it as context
# MAGIC

# COMMAND ----------

# MAGIC %md ## Prep vector database
# MAGIC
# MAGIC We need to choose an _embedding model_ for converting both the document contents and the user queries into numerical embedding representations, and a _vector database_ which will allow for fast and accurate retrieval of relevant documents based on the queries. Our choices:
# MAGIC - The [`e5-large-v2`](https://huggingface.co/intfloat/e5-large-v2) embedding model hosted on Huggingface, which performs at or near [state-of-the-art](https://huggingface.co/spaces/mteb/leaderboard) on retrieval tasks but is also relatively memory- and compute-efficient
# MAGIC - chromadb, an open-source library for efficient similarity search and clustering of embeddings
# MAGIC
# MAGIC Both Huggingface and Chromadb have strong integrations with langchain, which will prove useful when we build the LLM application in the next notebook.
# MAGIC

# COMMAND ----------

embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")

# COMMAND ----------

db = Chroma.from_documents(docs, embeddings)

# COMMAND ----------

# query it
query = "What is MLFlow?"
docs = db.similarity_search(query)

# print results
print(docs[0].page_content)

# COMMAND ----------

# MAGIC %sql
# MAGIC USE CATALOG uc_demos_chris_crawford;
# MAGIC USE SCHEMA mydata;

# COMMAND ----------

display(dbutils.fs.ls(vector_db_dir))

# COMMAND ----------

#!ls -al dbfs:/Volumes/uc_demos_chris_crawford/mydata/llm_vectordb/
#dbfs:/Users/sarbani.maiti@databricks.com/llm-data/llm-fin/pdf_vector_dbc
#!ls -al /databricks/driver/dbfs:/Volumes/uc_demos_chris_crawford/mydata/llm_vectordb/
#!ls -al /dbfs/Users/chris.crawford@databricks.com/llm_vectordb/chromadb/
!ls -al dbfs\:Volumes/uc_demos_chris_crawford/mydata/llm_vectordb/

# COMMAND ----------

# save to disk
db2 = Chroma.from_documents(docs, embeddings, persist_directory='dbfs:/Volumes/uc_demos_chris_crawford/mydata/llm_vectordb/')
#db2_docs = db2.similarity_search(query)

display(dbutils.fs.ls('dbfs:/Volumes/uc_demos_chris_crawford/mydata/llm_vectordb/'))
#display(dbutils.fs.ls(vector_db_dir))
#print(db2_docs[0].page_content)

# COMMAND ----------

# MAGIC %md Load back the saved vector store to ensure it will work in downstream applications
# MAGIC

# COMMAND ----------

db = FAISS.load_local(folder_path=vector_db_dir, embeddings=embeddings)

# COMMAND ----------

db.similarity_search_with_relevance_scores("What is the difference between bagging and boosting?", k=10)
