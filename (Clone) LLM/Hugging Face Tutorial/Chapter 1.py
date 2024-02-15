# Databricks notebook source
# MAGIC %pip datasets evaluate

# COMMAND ----------

from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier("I've been waiting for a HuggingFace course my whole life.")

# COMMAND ----------

from transformers import pipeline

generator = pipeline("text-generation")
generator("In this course, we will teach you how to")

# COMMAND ----------

from transformers import pipeline

classifier = pipeline("zero-shot-classification")
classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)

# COMMAND ----------


