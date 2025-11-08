import numpy as np
import pandas as pd
from datasets import load_dataset

print("-------------------LOADING DATASET")

dataset = load_dataset("Johnnyeee/Yelpdata_663")
train_df=dataset["train"].to_pandas()
test_df=dataset["test"].to_pandas()

print("----------------COMBINING TRAIN AND TEST DF")
df = pd.concat([train_df, test_df], ignore_index=True)

print("----------------FILTERING")
df_open = df[df['is_open']==1]
df=df_open

sample = df.groupby("stars_x", group_keys=False).apply(lambda x: x.sample(min(len(x), 100)))
sample.to_csv("rag/qa_testing/chunk_sample.csv", index=False)

"""
You are a LLM that will help to create synthetic question-answer pairs for evaluating chunk size and chunk overlap for a restaurant recommendation chatbot. 
The chatbot uses reviews to provide personalised, location specific, and preference based recommendations. 
Each QA should reflect realistic user intent and rely on information that can be found in the reviews such as cuisine, quality, service, ambiance, location, and other factors.
"""