# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 12:53:56 2021

@author: ca007843
"""

# import libraries for data processing and visualization
import glob
import matplotlib.pyplot as plt
import numpy as mp
import os
import pandas as pd
import seaborn as sns

# import libraries for natural language processing (NLP)
import nltk
from nltk.corpus import stopwords
import re
import unicodedata

def load_and_concat_csvs(csv_path):
    '''
    Load and concatenate the corpus of Indeed csvs containing job data
    Input   = csv_path   : string - points to the csv location
    Output  = csv_concat : dataframe - df containing the raw concatenated csvs
    '''
    # load and concatenate all Indeed csvs while adding a field for each record's parent csv name
    all_csvs = glob.glob(csv_path + "/*.csv")
    df_raw = pd.concat([pd.read_csv(fp).assign(csv_name=os.path.basename(fp)) for fp in all_csvs])

    return df_raw


def raw_csv_stats(df_raw):
    '''
    Generate statistics for the raw csv imports
    '''
    print('*****Import Statistics*****')
    print(f'Records imported: {df_raw.shape[0]} \n')
    print(f'Unique job titles: {df_raw.job_title.nunique()} \n')
    print(f'Nulls are present:\n{df_raw.isna().sum()} \n')
    print(f'Records missing job_title field: {(df_raw.job_title.isna().sum() / df_raw.shape[0] * 100).round(3)}%')
    print(f'Records missing job_Description field: {(df_raw.job_Description.isna().sum() / df_raw.shape[0] * 100).round(3)}% \n')
    print(f"Count of duplicates based on company, location, title and description: {df_raw.duplicated(subset=['job_title', 'company', 'location', 'job_Description']).sum()}")
    print(f"Duplication rate: {((df_raw.duplicated(subset=['job_title', 'company', 'location', 'job_Description']).sum()) / df_raw.shape[0] * 100).round(3) }% \n")


def clean_raw_csv(df_raw):
    # drop unnecessary fields and repair job_Description field name
    df_clean = df_raw.drop(['URL', 'page_count', 'post_date', 'reviews'], axis=1)
    df_clean.rename(columns={'job_Description':'job_description'}, inplace=True)

    # drop duplicates based on key fields and assert that the drop duplicates artithmetic worked corrrectly
    df_clean = df_clean.drop_duplicates(subset=['job_title', 'company', 'location', 'job_description'])
    assert (len(df_raw) - len(df_clean))   ==  (df_raw.duplicated(subset=['job_title', 'company', 'location', 'job_Description']).sum())
    
    # drop records that have NaN for job_title, company, location or job_description
    df_clean.dropna(subset = ['job_title', 'company', 'location', 'job_description'], inplace=True)
    
    # reset the index
    df_clean.reset_index(inplace=True, drop=True)
    
    print(f'{len(df_clean)} records remaining after intial data cleaning')
    
    return df_clean


def parse_date_scraped_field(df_clean):
    # convert csv_name field to a string 
    df_clean['csv_name'] = df_clean['csv_name'].astype(str)
    
    # from the csv_name field, create fields for the state, scraped job title and date scraped 
    df_clean['state'] = df_clean['csv_name'].str.slice(3, 5)
    df_clean['scrape_job_title'] = df_clean['csv_name'].str.slice(0, 2)
    df_clean['scrape_date'] = df_clean['csv_name'].str.slice(10, 20)
    
    # make a copy of df_clean as df, drop the now unnecessary csv_name field and delete the df_clean dataframe
    df = df_clean.copy()
    df = df.drop(['csv_name'], axis=1)
    
   
    return df

def clean_for_nlp(series_for_nlp):
    '''
    Execute stopword removal, lowercasing, encoding/decoding, normalizing and 
    lemmatization in preparation for NLP.
    '''
    print('\nCleaning data for nlp...')
    
    # convert parsed series to a list
    text = ''.join(str(series_for_nlp.tolist()))

    # add additional stopwords to nltk default stopword list
    extra_stopwords = ['sexual', 'orientation', 'equal', 'opportunity', 'origin', 'gender', 'identity', 'marital',
                       'status', 'applicant', 'religion', 'sex', 'race', 'color', 'without', 'regard', 'reasonable',
                       'accomodation', 'protected', 'veteran', 'consideration', 'employment', 'receive', 'consideration',
                       'applicant', 'receive', 'united', 'state', 'job', 'description', 'york', 'disability', 'age',
                       'candidate', 'fully', 'vaccinated', 'covid19', 'affirmative', 'action', 'employer', 'discriminate',
                       'arrest', 'conviction', 'please', 'visit', 'every', 'day', 'san', 'francisco', 'around', 'world',
                       'applicable', 'law', 'applicant', 'criminal', 'history', 'etc']
    
    benefits_stopwords = ['medical', 'dental', 'vision', 'pregnancy', 'childbirth', 'life', 'insurance']
    
    
    stopwords = nltk.corpus.stopwords.words('english') + extra_stopwords + benefits_stopwords
    
    # normalize, split and lowercase the parsed text
    text = (unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore').lower())
    words = re.sub(r'[^\w\s]', '', text).split()
    
    # initialize lemmatizer and execute lemmatization
    wnl = nltk.stem.WordNetLemmatizer()
    terms_for_nlp = [wnl.lemmatize(word) for word in words if word not in stopwords]
    
    return terms_for_nlp

def identify_n_grams(terms_for_nlp, n_gram_count, n_gram_range_start, n_gram_range_stop):
    print('\nIdentifying n_grams...')
    n_grams = (pd.Series(nltk.ngrams(terms_for_nlp, n_gram_count)).value_counts())[n_gram_range_start:n_gram_range_stop]
    print(n_grams)
    return n_grams

def create_word_cloud():
    pass

####### !!!!!!!! START HERE NEXT  #########
# clean up doc strings and comments
# build word cloud function

# define universal variables
csv_path = r'C:\Users\ca007843\Documents\100_mine\nlp\data'

# execute cleaning and field parsing
df_raw        = load_and_concat_csvs(csv_path)
raw_csv_stats(df_raw)
df_clean      = clean_raw_csv(df_raw)
df            = parse_date_scraped_field(df_clean)
series_for_nlp = df['job_description']
terms_for_nlp  = clean_for_nlp(series_for_nlp)

# execute nlp
n_gram_count = 2
n_gram_range_start, n_gram_range_stop  = 0, 50
n_grams = identify_n_grams(terms_for_nlp, n_gram_count, n_gram_range_start, n_gram_range_stop)





# clean up intermediate dataframes and variables
del df_raw, df_clean
















#######  ARCHIVE ######
# csv_list = []

# for csv in all_csvs:
#     csv_temp = pd.read_csv(csv, index_col=None, header=0)
#     csv_list.append(csv_temp)

# # concatnate csvs into a single dataframe
# df_raw = pd.concat(csv_list, axis=0, ignore_index=True)


