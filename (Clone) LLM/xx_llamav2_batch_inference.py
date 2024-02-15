# Databricks notebook source
# MAGIC %md
# MAGIC # Llama 2 chat Batch Inference on Databricks
# MAGIC
# MAGIC Adapted from Databricks [ml-examples repo notebook](https://github.com/databricks/databricks-ml-examples/blob/master/llm-models/llamav2/llamav2-7b/01_load_inference.py)
# MAGIC
# MAGIC [Llama 2](https://huggingface.co/meta-llama) is a collection of pretrained and fine-tuned generative text models ranging in scale from 7 billion to 70 billion parameters. It is trained with 2T tokens and supports context length window upto 4K tokens. [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) is the 7B fine-tuned model, optimized for dialogue use cases and converted for the Hugging Face Transformers format.
# MAGIC
# MAGIC Environment for this notebook:
# MAGIC - Runtime: 13.2 GPU ML Runtime
# MAGIC - Instance: `g5.4xlarge` on AWS, `Standard_NV36ads_A10_v5` on Azure
# MAGIC
# MAGIC Requirements:
# MAGIC - To get the access of the model on HuggingFace, please visit the [Meta website](https://ai.meta.com/resources/models-and-libraries/llama-downloads) and accept the license terms and acceptable use policy before submitting this form. Requests will be processed in 1-2 days.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load model

# COMMAND ----------

from huggingface_hub import notebook_login

# Login to Huggingface to get access to the model
notebook_login()

# COMMAND ----------

# Load model to text generation pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

# it is suggested to pin the revision commit hash and not change it for reproducibility because the uploader might change the model afterwards; you can find the commmit history of llamav2-7b-chat in https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/commits/main
model = "meta-llama/Llama-2-7b-chat-hf"
revision = "0ede8dd71e923db6258295621d817ca8714516d4"

tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left")
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    revision=revision,
    return_full_text=False
)

# Required tokenizer setting for batch inference
pipeline.tokenizer.pad_token_id = tokenizer.eos_token_id

# COMMAND ----------

# MAGIC %md ## (Small) batch inference pipeline example

# COMMAND ----------

# MAGIC %md #### Get text examples from an existing dataset

# COMMAND ----------

df_amazon_reviews_sample = spark.read.parquet("dbfs:/databricks-datasets/amazon/test4K").sample(.01)
print(f"{df_amazon_reviews_sample.count()} records")
display(df_amazon_reviews_sample)

# COMMAND ----------

# MAGIC %md #### Convert the Spark dataframe to a pandas dataframe for local inference. Then extract the customer reviews as a list.

# COMMAND ----------

pd_amazon_reviews_sample = df_amazon_reviews_sample.select("review").toPandas()

reviews = pd_amazon_reviews_sample["review"].tolist()
reviews[:5]

# COMMAND ----------

# MAGIC %md #### Define functions to create prompts and generate text

# COMMAND ----------

def gen_text(prompts, use_template=False, **kwargs):
    if use_template:
        full_prompts = [
            PROMPT_FOR_GENERATION_FORMAT.format(instruction=prompt)
            for prompt in prompts
        ]
    else:
        full_prompts = prompts

    if "batch_size" not in kwargs:
        kwargs["batch_size"] = 1
    
    # the default max length is pretty small (20), which would cut the generated output in the middle, so it's necessary to increase the threshold to the complete response
    if "max_new_tokens" not in kwargs:
        kwargs["max_new_tokens"] = 512

    # configure other text generation arguments, see common configurable args here: https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    kwargs.update(
        {
            "pad_token_id": tokenizer.eos_token_id,  # Hugging Face sets pad_token_id to eos_token_id by default; setting here to not see redundant message
            "eos_token_id": tokenizer.eos_token_id,
        }
    )

    outputs = pipeline(full_prompts, **kwargs)
    outputs = [out[0]["generated_text"] for out in outputs]

    return outputs

# COMMAND ----------

# MAGIC %md Form the prompts sequentially first, so we can send them to the LLM pipeline in batch mode

# COMMAND ----------

def form_feature_list_prompts(reviews:list) -> list:
  
  def form_prompt(review):
    SYSTEM_PROMPT = """\
    You are a text extractor. Always output your answer in a comma-separated list on a single line. No preamble. No narrative. No chat. No numbering."""

    INTRO_BLURB = "Identify any specific product features or qualities listed in the following review."
    return """
      <s>[INST]<<SYS>>
      {system_prompt}
      <</SYS>>
      {instruction}
      {context}
      [/INST]
      """.format(
          system_prompt=SYSTEM_PROMPT,
          instruction=INTRO_BLURB,
          context=review
      )

  prompts = [form_prompt(r) for r in reviews]

  return prompts

# COMMAND ----------

prompts = form_feature_list_prompts(reviews=reviews)
prompts[:3]

# COMMAND ----------

pd_amazon_reviews_sample["features"] = gen_text(prompts=prompts, max_new_tokens=128, use_template=False, batch_size=8)
pd_amazon_reviews_sample.head()

# COMMAND ----------

# MAGIC %md #### Write the new dataframe with extracted text back to a Delta Lake table in UC

# COMMAND ----------

df_reviews_with_features = spark.createDataFrame(pd_amazon_reviews_sample)

# COMMAND ----------

# df_reviews_with_features.write(<PATH TO YOUR STORAGE LOCATION>)
