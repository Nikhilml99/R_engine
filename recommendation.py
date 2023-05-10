import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
wn = nltk.WordNetLemmatizer()
from nltk.corpus import wordnet
import warnings
warnings.filterwarnings("ignore")
from flask import Flask, render_template, request, url_for, flash, redirect
import os
import pathlib
import string
from data_preprocessing import *
wn = nltk.WordNetLemmatizer()
stopword = nltk.corpus.stopwords.words('english')
################################################################################
base_dir = pathlib.Path(__name__).parent.absolute()
resume_df = pd.read_csv(os.path.join(base_dir,'data_csv', 'client.csv'))
jobs_df = pd.read_csv(os.path.join(base_dir,'data_csv', 'naukari.csv'))
resume_df40 = pd.read_csv('/Users/sharmanikhi/Walfly_main_project/data_csv/client22.csv')
#################################################################################

# for candidate recommendation

def cr(resume_df_input, job_text, num):
    resumeprocessed = resume_df_input.copy()
    resumeprocessed[['Designation', 'Skills', 'education', 'cities', 'countries', 'regions']] = resumeprocessed[['Designation', 'Skills', 'education', 'cities', 'countries', 'regions']].applymap(data_list)
    resumeprocessed = education_email(resumeprocessed)
    resumeprocessed[['Designation', 'Skills', 'education', 'cities', 'countries', 'regions']] = resumeprocessed[['Designation', 'Skills', 'education', 'cities', 'countries', 'regions']].applymap(remove_punct).applymap(tokenization).applymap(lambda x: [word for word in x if word not in stopword]).applymap(lemmatizer)
    
    resume_text = [' '.join(resume) for resume in resumeprocessed[['Designation', 'Skills', 'education', 'cities', 'countries', 'regions']].values]
    # make vectors
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), min_df=0, stop_words='english')
    job_features = vectorizer.fit_transform([job_text])
    resume_features = vectorizer.transform(resume_text)
    top_candidate_indices = cosine_similarity(job_features, resume_features).ravel().argsort()[::-1][:num]
    candidate = resume_df_input.iloc[top_candidate_indices, ['Name', 'Mobile', 'Email', 'Resume_path']]
    return candidate

# for job recommendation
def givejob(resume_df,jobs_df,num):
    resume_df[['Designation', 'Skills', 'education', 'cities', 'countries', 'regions']]= resume_df[['Designation', 'Skills', 'education', 'cities', 'countries', 'regions']].applymap(data_list)
    resume_df = education_email(resume_df)
    resume_df[['Designation', 'Skills', 'education', 'cities', 'countries', 'regions']] = resume_df[['Designation', 'Skills', 'education', 'cities', 'countries', 'regions']].applymap(remove_punct).applymap(tokenization).applymap(remove_stopwords).applymap(lemmatzer)

    resume_text = resume_merge(resume_df)
    resume_text = resume_text[0]
    job_text = jobs_df_merge(jobs_df)
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), min_df=0, stop_words='english')
    resume_features = vectorizer.fit_transform([resume_text])
    print(resume_features.shape)
    job_features = vectorizer.transform(job_text)
    print(job_features.shape)
    # Compute the cosine similarity between the resume and job data
    top_job_indices = cosine_similarity(resume_features, job_features)
    prd_arr = sorted(top_job_indices[0], reverse=True)
    top_job_indices = top_job_indices.argsort()[0][::-1][:num]
    jobs = jobs_df.iloc[top_job_indices]['Job Title'].tolist()
    return jobs

# without extraction
def givejob2(resume_df,jobs_df,num):
    resume_text = resume_df
    job_text = jobs_df_merge(jobs_df)
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), min_df=0, stop_words='english')
    resume_features = vectorizer.fit_transform(resume_text)
    print(resume_features.shape)
    job_features = vectorizer.transform(job_text)
    print(job_features.shape)
    # Compute the cosine similarity between the resume and job data
    top_job_indices = cosine_similarity(resume_features, job_features)
    prd_arr = sorted(top_job_indices[0], reverse=True)
    top_job_indices = top_job_indices.argsort()[0][::-1][:num]
    jobs = jobs_df.iloc[top_job_indices]['Job Title'].tolist()
    return jobs














