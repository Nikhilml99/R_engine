import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings("ignore")
from fastapi import FastAPI, HTTPException

from data_preprocessing import *
wn = nltk.WordNetLemmatizer()
stopword = nltk.corpus.stopwords.words('english')

# import functions for candidate recommendation
from typing import List, Dict
app = FastAPI()

def cr(resume_df_input, job_text, num):
    ...
    # Function code here for candidate recommendation

# import functions for job recommendation
def givejob(resume_df, jobs_df, num):
    ...
    # Function code here for job recommendation

def givejob2(resume_df, jobs_df, num):
    ...
    # Function code here for job recommendation

@app.get("/")
async def read_root():
    return {"msg": "testing api!"}

@app.post("/recommend_jobs/")
async def recommend_jobs(resume_df: Dict[str, List[str]], jobs_df: List[str], num: int):
    # Input format for resume_df should be a dictionary with keys for each column
    # Output format should be a list of recommended job titles
    jobs = givejob2(resume_df, jobs_df, num)
    return {"recommended_jobs": jobs}

@app.post("/recommend_candidates/")
async def recommend_candidates(resume_df_input: Dict[str, List[str]], job_text: str, num: int):
    # Input format for resume_df_input should be a dictionary with keys for each column
    # Output format should be a dictionary of recommended candidates with their details
    candidate = cr(resume_df_input, job_text, num)
    return {"recommended_candidates": candidate}

#  typeerror : list indices must be interger or sloces not str
# resolve this issue give example in python


# read items





