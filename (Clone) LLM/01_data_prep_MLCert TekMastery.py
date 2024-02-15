# Databricks notebook source
# MAGIC %md # Data preparation and vector database setup
# MAGIC
# MAGIC Let's consider a scenario in which an organization wishes to provide an instruction support tool for aviation students. The FAA has published very helpful manuals, which are meant to be studied in their entirety. But the manuals are hundreds of pages long, so if students have specific questions, they might spend a lot of time searching for relevant passages and using them to formulate an answer. Enter Large Language Models! We will use LLMs and auxiliary tools to develop a knowledge base Q&A service on these aviation manuals.
# MAGIC
# MAGIC The first step in building the service is to prepare the data that will guide the LLM towards creating clear and accurate answers
# MAGIC
# MAGIC ![test image](files/shared_uploads/tim.lortz@databricks.com/faa_qa/data_prep.png)

# COMMAND ----------

# MAGIC %md ## Library installs & imports

# COMMAND ----------

# MAGIC %pip install -U transformers langchain faiss-cpu pypdf unstructured sentence_transformers accelerate pdfminer.six pdf2image chromadb

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from langchain.document_loaders import UnstructuredPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os

# COMMAND ----------

source_pdf_dir = "/Volumes/uc_demos_chris_crawford/mydata/llm_pdfs"
vector_db_dir = "/Volumes/uc_demos_chris_crawford/mydata/llm_vectordb"
embedding_model_dir = "/Volumes/uc_demos_chris_crawford/mydata/llm_embedded_modeling_dir"

# COMMAND ----------

# MAGIC %md ## Load FAA instruction manuals from the web
# MAGIC
# MAGIC The rapidly expanding capabilities and availability of open-source LLMs has been joined and assisted by auxiliary tools such as langchain and unstructured. These facilitate preparing data for use in LLMs and orchestrating the necessary interactions between LLMs and data sources. 
# MAGIC
# MAGIC ![langchain logo](files/shared_uploads/tim.lortz@databricks.com/faa_qa/langchain_logo.png)
# MAGIC ![unstructured logo](files/shared_uploads/tim.lortz@databricks.com/faa_qa/unstructured_logo.png)
# MAGIC
# MAGIC The manuals block downloads using standard code-based download protocols, so they were downloaded previously and stored in DBFS.

# COMMAND ----------

display(dbutils.fs.ls(source_pdf_dir))

# COMMAND ----------

pdf_paths = [os.path.join(source_pdf_dir, file) for file in os.listdir(source_pdf_dir)]

docs = []
for p in pdf_paths:
  loader = UnstructuredPDFLoader(p, mode="elements")
  data = loader.load()
  docs.extend(data)

# COMMAND ----------

len(docs)

# COMMAND ----------

docs[:3]

# COMMAND ----------

# MAGIC %md Using the `elements` mode of the `UnstructuredPDFLoader` generates a lot of useful metadata about each chunk of extracted text. In particular, the `category` element can help identify whether the content contains actual narrative text, or something less useful like a title. 
# MAGIC
# MAGIC Let's use this generated metadata to filter out text that likely has no utility in a retrieval scenario, in which we will provide the LLM with a very limited number of source documents as context. If the source documents were, say, titles that matched the search only because they contained keywords, then they would reduce the amount of potentially helpful context available to the LLM.

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

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 100)
filtered_split_docs = text_splitter.split_documents(filtered_docs)

filtered_split_docs[:20]

# COMMAND ----------

len(filtered_split_docs)

# COMMAND ----------

# MAGIC %md The text splitter settings we used created about 2000 more documents than before, but also helps us ensure that the LLM will not have to truncate the source documents passed to it as context

# COMMAND ----------

# MAGIC %md ## Prep vector database
# MAGIC
# MAGIC We need to choose an _embedding model_ for converting both the document contents and the user queries into numerical embedding representations, and a _vector database_ which will allow for fast and accurate retrieval of relevant documents based on the queries. Our choices:
# MAGIC - The [`e5-large-v2`](https://huggingface.co/intfloat/e5-large-v2) embedding model hosted on Huggingface, which performs at or near [state-of-the-art](https://huggingface.co/spaces/mteb/leaderboard) on retrieval tasks but is also relatively memory- and compute-efficient
# MAGIC - [Faiss](https://github.com/facebookresearch/faiss), an open-source library for efficient similarity search and clustering of embeddings, created by Facebook Research. Faiss has maintained popularity over the past several years due to its reliability and good performance
# MAGIC
# MAGIC Both Huggingface and Faiss have strong integrations with langchain, which will prove useful when we build the LLM application in the next notebook.

# COMMAND ----------

embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")

# COMMAND ----------

max_split_size_mb = 1
db = FAISS.from_documents(documents=filtered_split_docs, embedding=embeddings)

# COMMAND ----------

db.similarity_search_with_relevance_scores("What ML library uses a single node cluster?", k=10)

# COMMAND ----------

db.save_local(vector_db_dir)

# COMMAND ----------

display(dbutils.fs.ls(vector_db_dir))

# COMMAND ----------

# MAGIC %md Load back the saved vector store to ensure it will work in downstream applications

# COMMAND ----------

db = FAISS.load_local(folder_path=vector_db_dir, embeddings=embeddings)

# COMMAND ----------

db.similarity_search_with_relevance_scores("What is the difference between bagging and boosting?", k=10)

# COMMAND ----------


