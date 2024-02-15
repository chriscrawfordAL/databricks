# Databricks notebook source
# MAGIC %md # Implementing and testing a knowledge base Q&A service on FAA manuals
# MAGIC
# MAGIC A best practice for harnessing the generative language capabilities of LLMs using organization-specific data is a method called Retrieval-Augmented Generation (RAG). With RAG, a user prompts the LLM with a question, as usual. However, rather than relying on only the LLM's internal "understanding", the question is coupled with relevant context: a set of organization-specific documents that have semantic meaning that is relevant to the question. The original question, the contextual documents, along with some guiding instructions are all wrapped in a "prompt" that is fed to the LLM. The whole process looks something like this:
# MAGIC
# MAGIC ![test image](files/shared_uploads/tim.lortz@databricks.com/faa_qa/query.png)

# COMMAND ----------

# MAGIC %md ## Library import and install

# COMMAND ----------

# MAGIC %pip install accelerate==0.21.0 openpyxl alibi einops ninja tokenizers==0.13.3 torch==2.0.0 transformers==4.30.2 triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir#subdirectory=python

# COMMAND ----------

# MAGIC %pip install -U faiss-cpu langchain

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from huggingface_hub import snapshot_download
# from langchain.chains import RetrievalQA
# from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.llms import HuggingFacePipeline
# from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
import os
import torch
import transformers
from transformers import AutoTokenizer, StoppingCriteria

# COMMAND ----------

vector_db_dir = "/Volumes/uc_demos_chris_crawford/mydata/llm_vectordb"

%env TRANSFORMERS_CACHE=/Volumes/uc_demos_chris_crawford/mydata/llm_cache
%env HF_HOME=/Volumes/uc_demos_chris_crawford/mydata/llm_cache
%env HF_HUB_DISABLE_SYMLINKS_WARNING=TRUE
%env HF_DATASETS_CACHE=/Volumes/uc_demos_chris_crawford/mydata/llm_cache

# COMMAND ----------

dbutils.fs.mkdirs(os.environ['HF_HOME'])
dbutils.fs.mkdirs(os.environ['TRANSFORMERS_CACHE'])
dbutils.fs.mkdirs(os.environ['HF_DATASETS_CACHE'])

# COMMAND ----------

# MAGIC %md ## Load vector database and embedding model used to create it from previous step

# COMMAND ----------

embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")

# COMMAND ----------

db = FAISS.load_local(vector_db_dir, embeddings=embeddings)

# COMMAND ----------

# MAGIC %md ## Load open-source LLM
# MAGIC
# MAGIC We'll use the capable MosaicML MPT-7B-instruct model from Huggingface. While a larger model will tend to generate better results, the 7B parameter model is small enough to fit on GPUs accessible to most Databricks customers, and can still be an effective model. The Databricks team has published an optimized mpt-7b-instruct inference example [here](https://github.com/databricks/databricks-ml-examples/blob/master/llm-models/mpt/mpt-7b/01_load_inference.py)
# MAGIC
# MAGIC ![mpt image](files/shared_uploads/tim.lortz@databricks.com/faa_qa/mpt-7b_logo.png)

# COMMAND ----------

# Download the MPT model snapshot from huggingface
snapshot_location = snapshot_download(repo_id="mosaicml/mpt-7b")
#snapshot_location = snapshot_download(repo_id="meta-llama/Llama-2-7b")

torch.cuda.empty_cache()

# Initialize tokenizer and language model
tokenizer = transformers.AutoTokenizer.from_pretrained(
  snapshot_location, padding_side="left")

# Although the model was trained with a sequence length of 2048, ALiBi enables users to increase the maximum sequence length during finetuning and/or inference. 
config = transformers.AutoConfig.from_pretrained(
  snapshot_location, 
  trust_remote_code=True,
  max_seq_len = 4096)

# support for flast-attn and openai-triton is coming soon
#config.attn_config['attn_impl'] = 'triton'

model = transformers.AutoModelForCausalLM.from_pretrained(
  snapshot_location, 
  config=config,
  torch_dtype=torch.bfloat16,
  revision="fb38c7169efd8a78c8e27e0a82cce74578100ee3", # of as 6/16
  trust_remote_code=True)

model.to(device='cuda')

model.eval()

display('model loaded')

# COMMAND ----------

# MAGIC %md `mpt-7b-instruct` can be used out of the box to respond to prompts. However, engineers are finding more efficient ways to call the model to generate text. Below is an implementation that Databricks engineers have found to work exceptionally well in terms of output quality and, especially, latency

# COMMAND ----------

def build_prompt(query, docs):
  """
  This method generates the prompt for the model.
  """
  CONTEXT = "\n".join([d.page_content for d in docs])
  INSTRUCTION_KEY = "### Instruction: using the context above, answer this question:"
  RESPONSE_KEY = "### Response:"
  INTRO_BLURB = (
      "Below are source documents that should be used to answer the question that follows. "
      "Write a response that appropriately answers the question."
  )

  return f"""{INTRO_BLURB}
  {CONTEXT}
  {INSTRUCTION_KEY}
  {query}
  {RESPONSE_KEY}
  """

def mpt7_instruct_generate(prompt, **generate_params):

  # Encode the input and generate prediction
  encoded_input = tokenizer.encode(prompt, return_tensors='pt').to('cuda')

  # do_sample (bool, optional): Whether or not to use sampling. Defaults to True.
  # max_new_tokens (int, optional): Max new tokens after the prompt to generate. Defaults to 256.
  # top_p (float, optional): If set to float < 1, only the smallest set of most probable tokens with
  #     probabilities that add up to top_p or higher are kept for generation. Defaults to 1.0.
  # top_k (int, optional): The number of highest probability vocabulary tokens to keep for top-k-filtering.
  #     Defaults to 50.
  # temperature (int, optional): Adjusts randomness of outputs, greater than 1 is random and 0.01 is deterministic. (minimum: 0.01; maximum: 5)

  if 'max_new_tokens' not in generate_params:
    generate_params['max_new_tokens'] = 256
  if 'temperature' not in generate_params:
    generate_params['temperature'] = 1.0
  if 'top_p' not in generate_params:
    generate_params['top_p'] = 1.0
  if 'top_k' not in generate_params:
    generate_params['top_k'] = 50
  if 'eos_token_id' not in generate_params:
    generate_params['eos_token_id'] = 0
    generate_params['pad_token_id'] = 0
  if 'do_sample' not in generate_params:
    generate_params['do_sample'] = True
  
  generate_params['use_cache'] = True

  output = model.generate(encoded_input, **generate_params)

  # Decode the prediction to text
  generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

  # Removing the prompt from the generated text
  # prompt_length = len(tokenizer.encode(wrapped_prompt, return_tensors='pt')[0])
  prompt_length = len(tokenizer.encode(prompt, return_tensors='pt')[0])
  generated_response = tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)

  return generated_response


def custom_doc_qa(query, num_docs):
  docs = db.similarity_search(query=query, k=num_docs)
  prompt = build_prompt(query=query, docs=docs)
  response = mpt7_instruct_generate(prompt=prompt)
  return {"response":response, "sources":docs}

# COMMAND ----------

query = "What is ther difference between bagging and boosting?"

results = custom_doc_qa(query,10)
results["response"]

# COMMAND ----------

results["sources"]

# COMMAND ----------

# MAGIC %md ## Evaluation
# MAGIC
# MAGIC With LLM applications, the developer has many options for optimizing performance. Some of the most obvious and impactful parameters include 
# MAGIC - the language in the prompt instruction
# MAGIC - the number of source documents to retrieve for context
# MAGIC - the max length of the response
# MAGIC - the temperature, top_p and top_k values for the text generation process
# MAGIC
# MAGIC Below is a very rudimentary analysis of the model outputs and model inference latency as a function of the number of source documents provided as context. We'll capture this information for each of several sample questions. 

# COMMAND ----------

questions = [
  "What is the difference between Bagging and boosting?",
  "When should I use SparkTrials vs Trials?",
  "What is an example of a single node machine learning library?",
  "Explain RMSE to me"
]

docs_to_retrieve = [2,4,8]
max_split_size_mb = 10

question_asked = []
docs_retrieved = []
response = []
time_elapsed = []

import time
import pandas as pd

for q in questions:
  for d in docs_to_retrieve:
    question_asked.append(q)
    docs_retrieved.append(d)
    start_time = time.time()
    response.append(custom_doc_qa(q,d)["response"])
    end_time = time.time()
    elapsed_time = end_time - start_time
    time_elapsed.append(elapsed_time)

pd_eval_results = pd.DataFrame({"question_asked": question_asked, "docs_retrieved": docs_retrieved,
                                "response": response, "time_elapsed": time_elapsed})

# COMMAND ----------

display(pd_eval_results)

# COMMAND ----------


pd_eval_results.to_markdown("/Volumes/uc_demos_chris_crawford/mydata/llm_output/output.md")


# COMMAND ----------

questions = [
  "How do I register a pkl file as a model in Databricks?"
]

docs_to_retrieve = [2,4,8]
max_split_size_mb = 10

question_asked = []
docs_retrieved = []
response = []
time_elapsed = []

import time
import pandas as pd

for q in questions:
  for d in docs_to_retrieve:
    question_asked.append(q)
    docs_retrieved.append(d)
    start_time = time.time()
    response.append(custom_doc_qa(q,d)["response"])
    end_time = time.time()
    elapsed_time = end_time - start_time
    time_elapsed.append(elapsed_time)

pd_eval_results = pd.DataFrame({"question_asked": question_asked, "docs_retrieved": docs_retrieved,
                                "response": response, "time_elapsed": time_elapsed})

# COMMAND ----------

display(pd_eval_results)

# COMMAND ----------

import mlflow

mlflow.register_model(model, "mlcert_llm")

# COMMAND ----------

mlflow.register_model("/Volumes/uc_demos_chris_crawford/mydata/llm_vectordb/index.pkl", "mlcert_llm_pkl")

# COMMAND ----------

# MAGIC %md In terms of LLM response time, we don't see any significant degradation by adding more source documents to the context, at least up to 16 documents. Response time is on the order of 1-3 seconds, which is excellent even compared to leading commercial LLMs. So we can focus on response quality as a function of source documents. 
# MAGIC
# MAGIC To the amateur eye of this author, the responses using 8 source documents all seem plausible, whereas those using 2, 4 or 16 documents seem to be incorrect in at least one instance each. However, we would ideally want to test more questions for factuality. For LLM application development, having domain experts involved in the evaluation process is critical.

# COMMAND ----------

# MAGIC %md ## Conclusions and next steps
# MAGIC
# MAGIC We have shown that it is possible to develop a simple yet effective Q&A service for a very niche set of documents without much code, using open-source language models and a vector database. The features used in this demo are available to all Databricks customers as of August 2023. 
# MAGIC
# MAGIC While the results in this demo are interesting and compelling, a team tasked with building a production LLM application might find that the demo doesn't meet their standards for quality or accuracy. In such a case, they might wish to improve the application using methods such as (in increasing order of difficulty):
# MAGIC - Modifying the language in the prompt
# MAGIC - Adding more source documents to the context
# MAGIC - Changing the LLM to one that has performed better on benchmark tests, such as one in the 30B+ parameter range
# MAGIC - Fine-tuning the model on curated question-answer pairs
# MAGIC
# MAGIC Once the team has developed a model that meets their standards for quality, accuracy, reliability, etc., they will want to deploy it in production. To do so typically requires a few more features than was shown in this demo. Databricks laid out the vision for these features during Data & AI Summit 2023 under the heading of "[Lakehouse AI](https://www.databricks.com/blog/lakehouse-ai)". These features include:
# MAGIC - Model Serving, GPU-powered and optimized for LLMs
# MAGIC - Hosted Vector Search for indexing, integrated with tables in the lakehouse
# MAGIC - Curated models, backed by optimized Model Serving for high performance
# MAGIC - Unified Data & AI Governance with Unity Catalog
# MAGIC - MLflow AI Gateway, a workspace-level API gateway that allows organizations to create and share routes, which then can be configured with various rate limits, caching, cost attribution, etc. to manage costs and usage
# MAGIC - and much more!
# MAGIC
# MAGIC <img style="float:right" width="1000px" src="https://live-databricksinc.pantheonsite.io/sites/default/files/inline-images/image1_3.png">
# MAGIC
# MAGIC You can signup to join previews for these features [here](https://docs.google.com/spreadsheets/d/1grSh-VKaOn_ka1N2WwAdiBJmUI2ODBSBRo32hKBzLt0/edit#gid=1291877295). 
