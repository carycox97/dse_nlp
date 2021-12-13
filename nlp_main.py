# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 12:53:56 2021

@author: ca007843
"""

# import libraries for admin
import os
import warnings

# import libraries for data processing
import glob
import numpy as np
import pandas as pd

# import libraries for visualization
import matplotlib.pyplot as plt
import PIL
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# import libraries for natural language processing (NLP)
import nltk
from nltk.corpus import stopwords
import re
import unicodedata

# remove row and column display restrictions in the console
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def load_and_concat_csvs(csv_path):
    '''
    Loads and concatenates the corpus of Indeed csvs containing job data.

    Parameters
    ----------
    csv_path : string
        Parameter pointing to the location of the stored csvs.

    Returns
    -------
    df_raw : dataframe
        Contains the raw concatenated csvs.

    '''
    # load and concatenate all Indeed csvs while adding a field for each record's parent csv name
    all_csvs = glob.glob(csv_path + "/*.csv")
    df_raw = pd.concat([pd.read_csv(fp).assign(csv_name=os.path.basename(fp)) for fp in all_csvs])

    return df_raw


def raw_csv_stats(df_raw):
    '''
    Generate statistics for the raw csv imports.

    Parameters
    ----------
    df_raw : dataframe
        Contains the raw concatenated csvs.

    Returns
    -------
    None.

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
    df_clean['state_abbrev'] = df_clean['csv_name'].str.slice(3, 5)
    df_clean['scrape_job_title'] = df_clean['csv_name'].str.slice(0, 2)
    df_clean['scrape_date'] = df_clean['csv_name'].str.slice(10, 20)
    
    # create a field to capture the full state name in sentence case

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
                       'applicable', 'law', 'applicant', 'criminal', 'history', 'etc', 'eg', 'andor', 'youll', 'including',
                       'u', 'using', 'way', 'set', 'accomodation', 'within', 'nonessential', 'suspended']
    
    benefits_stopwords = ['benefit', 'medical', 'dental', 'vision', 'pregnancy', 'childbirth', 'life', 'insurance']
    
    stopwords = nltk.corpus.stopwords.words('english') + extra_stopwords + benefits_stopwords
    
    # normalize, split and lowercase the parsed text
    text = (unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore').lower())
    words = re.sub(r'[^\w\s]', '', text).split()
    
    # initialize lemmatizer and execute lemmatization
    wnl = nltk.stem.WordNetLemmatizer()
    terms_for_nlp = [wnl.lemmatize(word) for word in words if word not in stopwords]
    
    return terms_for_nlp


def count_n_grams(terms_for_nlp, n_gram_count, n_gram_range_start, n_gram_range_stop):
    '''
    Count the volume of n_grams present within the job_description field of the Indeed data.

    Parameters
    ----------
    terms_for_nlp : list
        List containing scrapped and cleaned job descriptions; created in the clean_for nlp function.
    n_gram_count : integer
        A parameter representing the dimensionality of the n_grams of interest (e.g., 2 = bigram, 3 = trigram, etc.).
    n_gram_range_start : integer
        A parameter indicating the uoper bound (i.e., the n_gram with the highest count) of the n_grams Series.
    n_gram_range_stop : integer
        A parameter indicating the lower bound (i.e., the n_gram with the lowest count) of the n_grams Series.

    Returns
    -------
    n_grams : Series
        Contains the processed n_grams; sorted by count from highest to lowest.

    '''
    # indicate processing status in the console
    print('\nIdentifying n_grams...')
    
    # count n grams in the field of interest, bounding the count according to n_gram_range_start and n_gram_range_stop
    n_grams = (pd.Series(nltk.ngrams(terms_for_nlp, n_gram_count)).value_counts())[n_gram_range_start:n_gram_range_stop]
    print(n_grams)
    
    # visualize n_grams
    visualize_n_grams(n_grams)
    
    return n_grams


def visualize_indeed_data(df):
    
    # configure plot size, seaborne style and font scale
    plt.figure(figsize=(7, 10))
    sns.set_style('dark')
    sns.set(font_scale = 1.30)
    
    ax = sns.countplot(y='state', data=df, palette='gist_gray', 
                       order = df['state'].value_counts().index) # Blues_d, mako_r, ocean, gist_gray, gist_gray_r, icefire
    ax.set_title('Jobs by State')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

def visualize_n_grams(n_grams):
    '''
    Visualize the n_grams created by the count_n_grams function.

    Parameters
    ----------
    n_grams : Series
        Contains the processed n_grams; sorted by count from highest to lowest; converted to a df in this function.

    Returns
    -------
    None. Directly outputs visualizations.

    '''
    # configure plot size, seaborne style and font scale
    plt.figure(figsize=(7, 10))
    sns.set_style('dark')
    sns.set(font_scale = 1.3)
    
    # convert the n_grams Series to a Dataframe for seaborne visualization
    n_grams_cols = ['count']
    n_grams_df = pd.DataFrame(n_grams, columns=n_grams_cols)
    
    # pull the n_grams out of the index and bound the count of records to be visualized
    n_grams_df['grams'] = n_grams_df.index.astype('string')
    n_grams_df_sns = n_grams_df.iloc[:20]
    
    # create a horizontal barplot visualizing n_gram counts from greatest to least
    ax = sns.barplot(x='count', y='grams', data=n_grams_df_sns, orient='h', palette='mako_d') # Blues_d, mako_r, ocean, gist_gray, gist_gray_r, icefire
    ax.set_title('n grams')


def create_word_cloud():
    pass

####### !!!!!!!! START HERE NEXT  #########
# clean up doc strings and comments
# bar plot of count of jobs in states
# for visualization: branded for NLP/ML insights 
# build word cloud function
# expand stopwords
# create searches for key lists
# create list of ds skills
# create list of cloud tech
# create list of soft skills
# create seaborne charts for n-grams
# add timing
# select favorite sns color pallete, and maybe use that for the branding colors!
# determine optimal sns chat size
# think about grouping/stacking compaisons of n_grams based on job type/title
# consider a function to identify and collapse key terms, like 'scientist' and 'science', 'analytics' and 'analysis', 'algorithm' and 'technique', 'oral' and'verbal'
# consider a swarm plot for ....something, with job title or skill along the x_axis and some count/value along the y-axis,
  # maybe count of ds skills FOR THE UNICORN INDEX; yes, count the number of skills cited in each job listing, parsed by job title (which
  # has been collapsed and simplified)
# will need to make an index of key skills based on the n_gram results
# create a functionality to count how many jobs cite a specifc term I searched; probably just search the series with lov=c
# really need to grind out all stop words that aren't relevant

# define universal variables and data paths
csv_path = r'C:\Users\ca007843\Documents\100_mine\nlp\data_ds'
# csv_path = r'C:\Users\ca007843\Documents\100_mine\nlp\data_da'

# execute cleaning and field parsing
df_raw        = load_and_concat_csvs(csv_path)
raw_csv_stats(df_raw)
df_clean      = clean_raw_csv(df_raw)
df            = parse_date_scraped_field(df_clean)
series_for_nlp = df['job_description']
terms_for_nlp  = clean_for_nlp(series_for_nlp)

# execute nlp
n_gram_count = 1
n_gram_range_start, n_gram_range_stop  = 0, 200
n_grams = count_n_grams(terms_for_nlp, n_gram_count, n_gram_range_start, n_gram_range_stop)





# clean up intermediate dataframes and variables
del df_raw, df_clean
















#######  ARCHIVE ######
# csv_list = []

# for csv in all_csvs:
#     csv_temp = pd.read_csv(csv, index_col=None, header=0)
#     csv_list.append(csv_temp)

# # concatnate csvs into a single dataframe
# df_raw = pd.concat(csv_list, axis=0, ignore_index=True)


