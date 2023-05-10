import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('wordnet')
nltk.download('omw-1.4')
wn = nltk.WordNetLemmatizer()
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet
import warnings
warnings.filterwarnings("ignore")
from flask import Flask, render_template, request, url_for, flash, redirect
import os
import pathlib
import string
wn = nltk.WordNetLemmatizer()
stopword = nltk.corpus.stopwords.words('english')
import time
from multiprocessing import Pool
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



################################################################################
base_dir = pathlib.Path(__name__).parent.absolute()
resume_df = pd.read_csv(os.path.join(base_dir,'data_csv', 'client.csv'))
jobs_df = pd.read_csv(os.path.join(base_dir,'data_csv', 'naukari.csv'))
###############################################################################



#Preprocessing
def data_list(x):
    try:
        return 'Nan' if x == '[]' else ','.join(x[1:-1].split(','))
    except:
        return 'Nan'

##education_____________
def education_email(resume_df):
    # resume_df["Name"]=resume_df['Name'].apply(lambda x: str(x).lower().replace('CURRICULUM VITAE'.lower(),'Nan'))
    resume_df["Name"]=resume_df['Name'].str.lower().str.replace('curriculum vitae', 'Nan')
    resume_df['education'].replace(to_replace=[r"\\t|\\n|\\r", r'\r+|\n+|\t+',r"\t|\n|\r"], value=["","",""], regex=True, inplace=True)
    #ad = r'Created\s+with\s+an\s+evaluation\s+copy\s+of\s+Aspose\.Words\.\s+To\s+discover\s+the\s+full\s+versions\s+of\s+our\s+APIs\s+please\s+visit:\s+https://products\.aspose\.com/words/'
    ad = re.compile(r'Created\s+with\s+an\s+evaluation\s+copy\s+of\s+Aspose\.Words\.\s+To\s+discover\s+the\s+full\s+versions\s+of\s+our\s+APIs\s+please\s+visit:\s+https://products\.aspose\.com/words/')
    resume_df['education'] =resume_df['education'].apply(lambda x : re.sub(ad, "", x))
    a,b,c,d,e= '\\x0c','\\xa0','\\u200b','Education','EDUCATION'
    dat = [i.replace(a, '').replace(b,'').replace(c,'').replace(d,'').replace(e,'') for i in resume_df['education'].values]
    resume_df['education'] = dat

    #email
    dat = [ str(x).lower().replace('e-mail:-','').replace('e-mail :','').replace('email :-','').replace('email:','').replace('email :','').replace('e_mail :-','').replace('e_mail :','').replace('id:','').replace('mail :-','') for x in  resume_df['Email'].values]
    resume_df['Email'] =dat
    return resume_df
#######################################################################################


# remove_Punctuation___________
# def remove_punct(text):
#     text = "".join([char for char in text if char not in string.punctuation])
#     text = re.sub('[0-9]+', '', text)
#     return text

def remove_punct(text):
    text = np.char.translate(text, str.maketrans('', '', string.punctuation))
    text = re.sub('[0-9]+', '', text)
    return text


def tokenization(text):
    text = re.split('\W+', str(text))
    return text

# it consume less memory that's why we used for loop instead of compreshion ------------------>>>>->>>>>
# def remove_stopwords(text):
#     text = [word for word in text if word not in stopword]
#     return text

def remove_stopwords(text):
    return (word for word in text if word not in stopword)


# get_wordnet_pos function--------------------->>>>>>>>>>>>>>>>>>>>>>>>>
# def get_wordnet_pos(treebank_tag):
#     treebank_tag = str(treebank_tag)
#     if treebank_tag.startswith('V'):
#         return wordnet.VERB
#     elif treebank_tag.startswith('N'):
#         return wordnet.NOUN
#     elif treebank_tag.startswith('R'):
#         return wordnet.ADV
#     else:
#         return wordnet.NOUN
def get_wordnet_pos(treebank_tag):
    return [wordnet.VERB if t.startswith('V') else wordnet.NOUN if t.startswith('N') else wordnet.ADV if t.startswith('R') else wordnet.NOUN for t in treebank_tag]


def lemmatzer(text):
    words_and_tags = nltk.pos_tag(text)
    lem = []
    for word, tag in words_and_tags:
        lemma = wn.lemmatize(word, pos=get_wordnet_pos(tag))
        lem.append(lemma)
    return lem

########################################################################################

def resume_merge(resume_df):
    #resume_df['Experience_Period']= resume_df['Experience_Period'].apply(lambda x: str(x)+' years')-------------->>>>>>>>
    resume_df['Experience_Period'] = pd.to_numeric(resume_df['Experience_Period'], errors='coerce').fillna(0).astype(np.int32)
    cols = ['Skills', 'education', 'Experience_Period','Designation']
    resume_df['Resume'] = resume_df[cols].astype(str).apply(lambda row: '_'.join(row.values.astype(object)), axis=1)
    resume_df['Resume'].iloc[0].replace("'","").replace("[",'').replace("]",'')
    resume_text = resume_df['Resume']
    return resume_text


def jobs_df_merge(jobs_df):
    jobs_df['Key Skills'] =jobs_df['Key Skills'].apply(lambda x: str(x))
    jobs_df['Job Title'] =jobs_df['Job Title'].apply(lambda x: str(x))
    jobs_df['Role Category'] =jobs_df['Role Category'].apply(lambda x: str(x))
    job_text = jobs_df[['Job Title', 'Job Experience Required', 'Key Skills']].apply(lambda x: ' '.join(x.astype(str)), axis=1)
    return job_text
job_text = jobs_df_merge(jobs_df)
###########################################################################################

# for candidate recommendation --------->>>>>>>>>>>>>>>>>>>>>>>>
# def cr(resume_df_input,job_text):
#     resumeprocessed = resume_df_input.copy()
#     lst = ['Designation', 'Skills', 'education', 'cities', 'countries', 'regions']
#     for i in lst:
#         resumeprocessed[f'{i}'] = resumeprocessed[f'{i}'].apply(lambda x: data_list(x))
#     resumeprocessed = education_email(resumeprocessed)
       
    
    
#     for i in lst:
#         resumeprocessed[f'{i}'] = resumeprocessed[f'{i}'].apply(lambda x: remove_punct(x)).apply(lambda x: tokenization(x)).apply(
#             lambda x: remove_stopwords(x)).apply(lambda x: lemmatzer(x))
#     resume_text = resume_merge(resumeprocessed)
#     vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
#     # job_features = vectorizer.fit_transform([job_text])-------------->>>>>>>>>>>>>>>>>>>>>>>>>>>
#     # resume_features = vectorizer.transform(resume_text)----------------->>>>>>>>>>>>>>>>>>>>>>>>>>>

#     job_features = vectorizer.fit_transform([job_text]).tocsc()
#     resume_features = vectorizer.transform(resume_text).tocsc()
    
#     # Compute the cosine similarity between the resume and job data
    
#     top_candidate_indices = cosine_similarity(job_features, resume_features)
#     pred_arr = sorted(top_candidate_indices[0], reverse=True)
#     pred = sorted(top_candidate_indices[0], reverse=False)
#     top_candidate_indices=top_candidate_indices.argsort()[0][:3][::-1]
#     candidate = resume_df.iloc[top_candidate_indices][['Name','Mobile','Email']]
#     return candidate


def preprocess_resume(resume_df):
    lst = ['Designation', 'Skills', 'education', 'cities', 'countries', 'regions']
    resumeprocessed = resume_df.copy()
    for col in lst:
        resumeprocessed[col] = data_list(resumeprocessed[col])
    resumeprocessed = education_email(resumeprocessed)
    for col in lst:
        resumeprocessed[col] = resumeprocessed[col].apply(tokenization).apply(remove_stopwords).apply(lemmatzer)
    return resume_merge(resumeprocessed)

def compute_similarity(job_text, resume_text):
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
    job_features = vectorizer.fit_transform([job_text])
    resume_features = vectorizer.transform(resume_text)
    return cosine_similarity(job_features, resume_features)

def cr(resume_df_input, job_text):
    resume_text = preprocess_resume(resume_df_input)
    with Pool() as pool:
        top_candidate_indices = pool.starmap(compute_similarity, [(job_text, [text]) for text in resume_text])
    top_candidate_indices = csr_matrix(top_candidate_indices)
    top_candidate_indices = top_candidate_indices.argsort(axis=1)[:, -3:][:, ::-1]
    candidate = resume_df_input.iloc[top_candidate_indices][['Name', 'Mobile', 'Email']]
    return candidate

job_text = 'data science, nlp , finetuning'