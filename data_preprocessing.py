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
import os
import pathlib
import string
wn = nltk.WordNetLemmatizer()
stopword = nltk.corpus.stopwords.words('english')
from data_extraction import *
##############################################################################
base_dir = pathlib.Path(__name__).parent.absolute()
# resume_df = pd.read_csv(os.path.join(base_dir,'data_csv', 'client22.csv'))
# jobs_df = pd.read_csv(os.path.join(base_dir,'data_csv', 'naukari.csv'))
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

def remove_stopwords(text):
    return (word for word in text if word not in stopword)


def get_wordnet_pos(treebank_tag):
    return [wordnet.VERB if t.startswith('V') else wordnet.NOUN if t.startswith('N') else wordnet.ADV if t.startswith('R') else wordnet.NOUN for t in treebank_tag]


def lemmatzer(text):
    words_and_tags = nltk.pos_tag(text)
    lem = []
    for word, tag in words_and_tags:
        lemma = wn.lemmatize(word, pos=get_wordnet_pos(tag))
        lem.append(lemma)
    return lem

#########################################################################################################

def resume_merge(resume_df):
    resume_df['Experience_Period']= resume_df['Experience_Period'].apply(lambda x: str(x)+' years')
    cols = ['Skills', 'education', 'Experience_Period','Designation']
    resume_df['Resume'] = resume_df[cols].astype(str).apply(lambda row: '_'.join(row.values.astype(object)), axis=1)
    # resume_df['Resume'].iloc[0].replace("'","").replace("[",'').replace("]",'')
    resume_text = resume_df['Resume']
    return resume_text

def jobs_df_merge(jobs_df):
    jobs_df['Key Skills'] =jobs_df['Key Skills'].apply(lambda x: str(x))
    jobs_df['Job Title'] =jobs_df['Job Title'].apply(lambda x: str(x))
    jobs_df['Role Category'] =jobs_df['Role Category'].apply(lambda x: str(x))
    job_text = jobs_df[['Job Title', 'Job Experience Required', 'Key Skills']].apply(lambda x: ' '.join(x.astype(str)), axis=1)
    return job_text

# can you add changes in this code regarding to code optimization to improve the computation speed in this code 