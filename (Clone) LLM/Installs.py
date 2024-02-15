# Databricks notebook source
# MAGIC %pip install dbdemos

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import dbdemos
dbdemos.install('llm-rag-chatbot', catalog='main', schema='rag_chatbot')

# COMMAND ----------


