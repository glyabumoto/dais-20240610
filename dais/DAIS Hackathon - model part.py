# Databricks notebook source
# MAGIC %sh
# MAGIC
# MAGIC python --version

# COMMAND ----------

!pip install --upgrade pip

# COMMAND ----------

!pip install langchain openai pandas

# COMMAND ----------

!pip install openai

# COMMAND ----------

import openai
import os

# private account
os.environ["OPENAI_API_KEY"] = mapbox_api_key = dbutils.secrets.get(scope="dais", key="openai")

openai.api_key = os.environ["OPENAI_API_KEY"]
gpt_model = "gpt-4o-2024-05-13"

# COMMAND ----------

import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(model=gpt_model)


# COMMAND ----------

import json
from langchain.retrievers.document_compressors import LLMChainExtractor

llm = ChatOpenAI(temperature=0)
compressor = LLMChainExtractor.from_llm(llm)

def ask(prompt):
    response = openai.ChatCompletion.create(
            model=gpt_model,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
    print(json.dumps(response, ensure_ascii=False, indent = 2))

# COMMAND ----------

sdf = spark.read.table("aterio_io_us_census_data_zip_code_insights_free_dataset.delievery.us_census_insights_zipcode")
pdf = sdf.toPandas()

display(pdf)

# COMMAND ----------

from langchain.vectorstores import VectorStore
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
selected_columns = ['ZIP_CODE', 'COUNTY_NAME', 'CITY_NAME', 'CBSA_TITLE']

pdf['combined_text'] = pdf[selected_columns].astype(str).agg(' '.join, axis=1)
text_data = pdf['combined_text'].tolist()

# ベクター化
vectors = [embeddings.embed(text) for text in text_data]

# VectorStoreに保存
vectorstore = VectorStore.from_vectors(vectors, metadata=census_data.to_dict(orient='records'))
