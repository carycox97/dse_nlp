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
from PIL import Image
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

# suspend filter warnings
warnings.filterwarnings("ignore")

def load_and_concat_csvs(csv_path):
    '''
    Load and concatenate the corpus of Indeed csvs containing job data.

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


def calculate_raw_csv_stats(df_raw):
    '''
    Generate statistics for the raw csv imports.

    Parameters
    ----------
    df_raw : dataframe
        Contains the raw concatenated Indeed csvs.

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
    '''
    Cleans the raw Indeed dataframe by dropping uncesessary fields, dropping duplicates, cleaning the NaNs
    and resetting the index.

    Parameters
    ----------
    df_raw : dataframe
        Contains the raw concatenated Indeed csvs.

    Returns
    -------
    df_clean : dataframe
        The cleaned version of df_raw, after duplicates dropped, NaNs cleaned, etc..

    '''
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
    '''
    From the date_scraped field, parse the state abbreviation and date, and create a field
    containing the full state name in sentence case.

    Parameters
    ----------
    df_clean : dataframe
        The cleaned version of df_raw, after duplicates dropped, NaNs cleaned, etc.

    Returns
    -------
    df : dataframe
        The primary dataframe for the concatenated, cleaned and parsed Indeed csv data..

    '''
    # convert csv_name field to a string 
    df_clean['csv_name'] = df_clean['csv_name'].astype(str)
    
    # from the csv_name field, create fields for the state, scraped job title and date scraped 
    df_clean['state_abbrev'] = df_clean['csv_name'].str.slice(3, 5)
    df_clean['scrape_job_title'] = df_clean['csv_name'].str.slice(0, 2)
    df_clean['scrape_date'] = df_clean['csv_name'].str.slice(10, 20)
    
    # create a dictionary to convert state abbreviations to full state names
    state_name_to_abbrev = {
        "Remote" : "re",
        "Alabama": "al",
        "Alaska": "ak",
        "Arizona": "az",
        "Arkansas": "ar",
        "California": "ca",
        "Colorado": "co",
        "Connecticut": "ct",
        "Delaware": "de",
        "Florida": "fl",
        "Georgia": "ga",
        "Hawaii": "hi",
        "Idaho": "id",
        "Illinois": "il",
        "Indiana": "in",
        "Iowa": "ia",
        "Kansas": "ks",
        "Kentucky": "ky",
        "Louisiana": "la",
        "Maine": "me",
        "Maryland": "md",
        "Massachusetts": "ma",
        "Michigan": "mi",
        "Minnesota": "mn",
        "Mississippi": "ms",
        "Missouri": "mo",
        "Montana": "mt",
        "Nebraska": "ne",
        "Nevada": "nv",
        "New Hampshire": "nh",
        "New Jersey": "nj",
        "New Mexico": "nm",
        "New York": "ny",
        "North Carolina": "nc",
        "North Dakota": "nd",
        "Ohio": "oh",
        "Oklahoma": "ok",
        "Oregon": "or",
        "Pennsylvania": "pa",
        "Rhode Island": "ri",
        "South Carolina": "sc",
        "South Dakota": "sd",
        "Tennessee": "tn",
        "Texas": "tx",
        "Utah": "ut",
        "Vermont": "vt",
        "Virginia": "va",
        "Washington": "wa",
        "West Virginia": "wv",
        "Wisconsin": "wi",
        "Wyoming": "wy",
        "District of Columbia": "dc",
    }

    # invert the state_name_to_abbrev dictionary and create the state_name field
    state_abbrev_to_state_name = dict(map(reversed, state_name_to_abbrev.items()))
    df_clean['state_name'] = df_clean['state_abbrev']
    df_clean['state_name'] = df_clean['state_name'].replace(state_abbrev_to_state_name)
    

    # make a copy of df_clean as df, drop the now unnecessary csv_name field and delete the df_clean dataframe
    df = df_clean.copy()
    df = df.drop(['csv_name'], axis=1)
    
    return df


def clean_for_nlp(series_of_interest):
    '''
    Execute stopword removal, lowercasing, encoding/decoding, normalizing and lemmatization in preparation for NLP.

    Parameters
    ----------
    series_of_interest : Series
        A variable set in the main program, series_of_interest contains the series of interest for NLP processing.

    Returns
    -------
    terms_for_nlp : list
        A list containing all terms (fully cleaned and processed) extracted from the series_of_interest Series.

    '''
    
    
    print('\nCleaning data for nlp...')
    
    # convert parsed series to a list
    text = ''.join(str(series_of_interest.tolist()))

    # add additional stopwords to nltk default stopword list
    permanent_stopwords = ['sexual', 'orientation', 'equal', 'opportunity', 'origin', 'gender', 'identity', 'marital',
                       'status', 'applicant', 'religion', 'sex', 'race', 'color', 'without', 'regard', 'reasonable',
                       'accomodation', 'protected', 'veteran', 'consideration', 'employment', 'receive', 'consideration',
                       'applicant', 'receive', 'united', 'state', 'job', 'description', 'york', 'disability', 'age',
                       'candidate', 'fully', 'vaccinated', 'covid19', 'affirmative', 'action', 'employer', 'discriminate',
                       'arrest', 'conviction', 'please', 'visit', 'every', 'day', 'san', 'francisco', 'around', 'world',
                       'applicable', 'law', 'applicant', 'criminal', 'history', 'etc', 'eg', 'andor', 'youll', 'including',
                       'u', 'using', 'way', 'set', 'accomodation', 'within', 'nonessential', 'suspended', 'genetic']
    
    benefits_stopwords = ['benefit', 'medical', 'dental', 'vision', 'pregnancy', 'childbirth', 'life', 'insurance']
    
    other_hr_stopwords = ['qualified', 'applicant', 'related', 'field', 'ability']
    
    stop_words = nltk.corpus.stopwords.words('english') + permanent_stopwords + benefits_stopwords + other_hr_stopwords
    
    # normalize, split and lowercase the parsed text
    text = (unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore').lower())
    words = re.sub(r'[^\w\s]', '', text).split()
    
    # initialize lemmatizer and execute lemmatization
    wnl = nltk.stem.WordNetLemmatizer()
    terms_for_nlp = [wnl.lemmatize(word) for word in words if word not in stop_words]
    
    return terms_for_nlp


def count_n_grams(terms_for_nlp, n_gram_count, n_gram_range_start, n_gram_range_stop):
    '''
    Count the volume of n_grams present within the job_description field of the Indeed data.

    Parameters
    ----------
    terms_for_nlp : list
        List containing scrapped and cleaned terms from the series of interest; created
        in the clean_for nlp function.
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


def filter_terms_for_nlp():
    value_list = ['data','science', 'python']
    boolean_series = pd.Series(terms_for_nlp).isin(value_list)
    filtered_series = pd.Series(terms_for_nlp)[boolean_series]
    print(len(filtered_series))
    print(filtered_series.value_counts())
    print(filtered_series[:10])
    
    # get only rows without the key terms - not sure if this works
    inverse_boolean_series = ~pd.Series(terms_for_nlp).isin(value_list)
    inverse_filtered_df = pd.Series(terms_for_nlp)[inverse_boolean_series]



def visualize_indeed_data(df):
    '''
    Generate basic visualizations for data exploration of the scraped Indeed csv data.

    Parameters
    ----------
    df : dataframe
        The primary dataframe for the concatenated, cleaned and parsed Indeed csv data.

    Returns
    -------
    None. Directly outputs visuaizations.

    '''
    # configure plot size, seaborne style and font scale
    plt.figure(figsize=(7, 10))
    sns.set_style('dark')
    sns.set(font_scale = 1.30)
    
    # create countplot for state counts
    ax = sns.countplot(y='state_name', data=df, palette='gist_gray', 
                       order = df['state_name'].value_counts().index) # Blues_d, mako_r, ocean, gist_gray, gist_gray_r, icefire
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
    ax = sns.barplot(x='count', y='grams', data=n_grams_df_sns, orient='h', palette='mako_d') # crest, mako, Blues_d, mako_r, ocean, gist_gray, gist_gray_r, icefire
    ax.set_title('n grams')


def visualize_word_clouds(terms_for_nlp):
    '''
    Generate masked and unmasked word clouds from the processed terms extracted from the 
    series of interest (e.g., job_description, company, etc.)

    Parameters
    ----------
    terms_for_nlp : list
        List containing scrapped and cleaned terms from the series of interest; created
        in the clean_for nlp function..

    Returns
    -------
    None. Directly outputs and saves visualizations as pngs.

    '''
    # convert the terms_for_nlp list into a string, which is what WordCloud expects
    word_cloud_terms = ' '.join(terms_for_nlp)
       
    # create a WordCloud object and optimize the terms for display; tune the Dunning collocation_threshold
    # to increase or decrease bigram frequency (low threshold=more bigrams)
    
    word_cloud = WordCloud(max_font_size=50,
                           max_words=100,
                           background_color='lightgray',      # whitesmoke, gainsboro, lightgray, silver
                           colormap='mako',                  # mako, crest
                           collocations=False).generate(word_cloud_terms)
    
    # display the word cloud
    plt.figure()
    plt.imshow(word_cloud, interpolation='lanczos') # bilinear, sinc, catrom, bessel, lanczos
    plt.axis('off')
    plt.show()
    
    # save the word cloud to a png
    word_cloud.to_file(f'word_clouds/word_cloud_{series_of_interest.name}.png')
    
    # create a word cloud that allows for bigrams
    word_cloud_bigrams = WordCloud(max_font_size=50,
                                   max_words=100,
                                   background_color='lightgray',      # whitesmoke, gainsboro, lightgray, silver
                                   colormap='mako',                  # mako, crest
                                   collocation_threshold=30).generate(word_cloud_terms)
    
    # display the bigram word cloud
    plt.figure()
    plt.imshow(word_cloud_bigrams, interpolation='lanczos') # bilinear, sinc, catrom, bessel, lanczos
    plt.axis('off')
    plt.show()
    
    # save the bigram word cloud to a png
    word_cloud.to_file(f'word_clouds/word_cloud_bigram_{series_of_interest.name}.png')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # read in mask for a word cloud canvassed on a mask outline
    word_cloud_mask = np.array(Image.open('bear.png'))

    # if needed, use this function and the lines below to convert a mask to white=255, black=0
    # def transform_mask_pixel_values(val):
    #     if val == 0:
    #         return 255
    #     else:
    #         return val   
    # Transform your mask into a new one that will work with the function:
    # transformed_word_cloud_mask = np.ndarray((word_cloud_mask.shape[0], word_cloud_mask.shape[1]), np.int32)
    # for i in range(len(word_cloud_mask)):
    #     transformed_word_cloud_mask[i] = list(map(transform_mask_pixel_values, word_cloud_mask[i]))    

    # create a masked WordCloud object and optimize the terms for display
    word_cloud_masked = WordCloud(max_font_size=50,
                                  max_words=100,
                                  background_color='lightgray',
                                  colormap='mako',                # crest
                                  mask=word_cloud_mask,
                                  contour_width=2, 
                                  contour_color='black').generate(word_cloud_terms)    

    # display the masked word cloud
    plt.figure()
    plt.imshow(word_cloud_masked, interpolation='lanczos') # bilinear, sinc, catrom, bessel, lanczos
    plt.axis('off')
    plt.show()
    
    # save the masked cloud to a png
    word_cloud_masked.to_file(f'word_clouds/word_cloud_masked_{series_of_interest.name}.png')
    
  

####### !!!!!!!! START HERE NEXT  #########
# finalize bar plot of count of jobs in states
# for visualization: branded for NLP/ML insights 
# find a better mask for the word cloud
# parse the date_scraped field for the parent scrape (e.g., ds, ml, etc.)
# expand stopwords, aggressively
# create searches for key lists
# create list of ds skills
# create list of cloud tech
# word clouds based only on the verbs in the job descriptions
# hold = df[df['job_description'].str.contains('attention to detail')]
# hold = df[df['job_description'].str.contains('|'.join(['passion','collaborate','teamwork','team work', 'interpersonal','flexibility','flexible','listening','listener','listen','empathy','empathetic']))]
# create list of soft skills
# create seaborne charts for n-grams
# add timing
# figure out how to brand-color the word clouds
# select favorite sns color pallete, and maybe use that for the branding colors!
# determine optimal sns chat size
# think about grouping/stacking compaisons of n_grams based on job type/title
# consider a function to identify and collapse key terms, like 'scientist' and 'science', 'analytics' and 'analysis', 'algorithm' and 'technique', 'oral' and 'verbal',
#     'model' and 'modeling', 'strong', 'excellent', 'writing' and 'written', 'discipline' and 'field',
#     'collaborate' and 'collaboratively' and 'work together' and 'work closely'; all numbers (1 vs one)
# consider a swarm plot for ....something, with job title or skill along the x_axis and some count/value along the y-axis,
  # maybe count of ds skills FOR THE UNICORN INDEX; yes, count the number of skills cited in each job listing, parsed by job title (which
  # has been collapsed and simplified)
# will need to make an index of key skills based on the n_gram results
# create a functionality to count how many jobs cite a specifc term I searched; probably just search the series with lov=c
# really need to grind out all stop words that aren't relevant
# can break skill lists into why/how/what subsets later


def main_program():
    pass


# define universal variables and data paths
csv_path = r'C:\Users\ca007843\Documents\100_mine\nlp\data_ds'
# csv_path = r'C:\Users\ca007843\Documents\100_mine\nlp\data_da'


# establish lists for nlp filtering
ds_cred_terms = ['ability', 
                 '1',
                 '2',
                 '3',
                 '4',
                 '5',
                 '6',
                 '7',
                 '8',
                 '10',
                 '12',
                 '35',
                 'advanced',
                 'analyst',
                 'analytics',
                 'bachelor',
                 'clearance',
                 'college', 
                 'combination',
                 'computer',
                 'data',
                 'degree',
                 'demonstrated', 
                 'discipline',
                 'education',
                 'electrical',
                 'engineer',
                 'engineering',
                 'equivalent',
                 'experience',
                 'field',
                 'graduate',
                 'handson',
                 'industry',
                 'knowledge',
                 'master',
                 'mathematics', 
                 'operation', 
                 'phd',
                 'portfolio',
                 'practical',
                 'prior',
                 'quantitative', 
                 'record',
                 'related',
                 'relevant',
                 'research',
                 'science',
                 'scientist',
                 'security', 
                 'social',
                 'software',
                 'solid', 
                 'statistic',
                 'statistics',
                 'system', 
                 'technical',
                 'track',
                 'understanding',
                 'university',
                 'work',
                 'working',
                 'year'] # degrees, work and job titles

ds_tech_skill_terms = ['ab',
                       'agile',
                       'ai',
                       'airflow',
                       'advanced',
                       'algorithm',
                       'amazon',
                     'analysis',
                     'analytical',
                     'analytics',
                     'analyze',
                     'analyzing',
                     'anomaly', 
                     'applied',
                     'applying',
                     'architecture',
                     'artificial',
                     'aws', 
                     'azure',
                     'bi',
                     'big',
                     'build', 
                     'c',
                     'cloud', 
                     'code', 
                     'collection'
                     'complex',
                     'computer',
                     'computing',
                     'concept',
                     'confidence',
                     'control',
                     'cutting', 
                     'data',
                     'datadriven',
                     'dataset',
                     'datasets',
                     'database',
                     'decision', 
                     'deep',
                     'deploy',
                     'design', 
                     'designing',
                     'detection',
                     'develop',
                     'development',
                     'distributed', 
                     'django',
                     'ecosystem',
                     'edge',
                     'emerging',
                     'engineer',
                     'engineering',
                     'enterprise',
                     'experimental', 
                     'exploratory',
                     'fastapi',
                     'feature', 
                     'flask',
                     'flow',
                     'forest',
                     'framework',
                     'generation',
                     'google',
                     'hadoop', 
                     'hidden',
                     'image',
                     'imagery',
                     'implement',
                     'implementation',
                     'information',
                     'infrastructure',
                     'integration',
                     'integrity',
                     'intelligence',
                     'java',
                     'lake',
                     'language',
                     'large',
                     'largescale',
                     'learning',
                     'lifecycle',
                     'machine',
                     'maintain',
                     'manipulation',
                     'mathematics',
                     'method',
                     'microsoft', 
                     'mining',
                     'ml',
                     'model',
                     'modeling',
                     'natural',
                     'network',
                     'next',
                     'neural', 
                     'office',
                     'open', 
                     'pipeline',
                     'platform',
                     'power',
                     'predictive',
                     'procedure',
                     'processing'
                     'programming', 
                     'python',
                     'pytorch',
                     'query',
                     'relational', 
                     'r',
                     'random', 
                     'reinforcement',
                     'relationship',
                     'research',
                     'review',
                     'sa',
                     'scale',
                     'science',
                     'scripting',
                     'series',
                     'server',
                     'service',
                     'set',
                     'signal', 
                     'software',
                     'source',
                     'spark',
                     'sql',
                     'statistic',
                     'statistics',
                     'statistical',
                     'structured',
                     'supervised', 
                     'system',
                     'tableau',
                     'technical',
                     'technique',
                     'technology',
                     'tensorflow', 
                     'testing',
                     'time', 
                     'tool',
                     'transformation',
                     'tree',
                     'uncover', 
                     'unstructured',
                     'unsupervised',
                     'version', 
                     'visualization',
                     'warehouse',
                     'warehousing',
                     'web']

ds_soft_skill_terms = ['ad', 
                       'ability',
                       'attention', 
                       'audience',
                     'best',
                     'business',
                     'cause',
                     'challenging', 
                     'closely',
                     'cognitive',
                     'collaborate',
                     'collaborative',
                     'collaboratively',
                     'communicate',
                     'communication',
                     'continuous', 
                     'critical', 
                     'crossfunctional',
                     'decision', 
                     'detail',
                     'dynamic',
                     'effectively',
                     'environment',
                     'excellent',
                     'experience',
                     'fastpaced',
                     'flexible',
                     'high', 
                     'highly', 
                     'hoc', 
                     'idea',
                     'impact',
                     'improvement',
                     'independently',
                     'intelligence',
                     'interpersonal',
                     'level',
                     'making',
                     'management',
                     'member',
                     'motivated',
                     'multidisciplinary',
                     'nontechnical', 
                     'oral,'
                     'organization',
                     'people',
                     'player',
                     'practice',
                     'presentation',
                     'problem', 
                     'problemsolving',
                     'quality',
                     'question',
                     'root', 
                     'skill',
                     'solve',
                     'solver',
                     'solving',
                     'strong',
                     'structure',
                     'team',
                     'technical',
                     'thinking',
                     'time', 
                     'together',
                     'user', 
                     'verbal',
                     'writing',
                     'written',
                     'work'
                     'working']

ds_prof_skill_terms = ['ability',
                       'actionable', 
                       'acumen',
                       'agile',
                       'build',
                       'business',
                       'career', 
                       'case',
                       'challenge',
                       'client',
                       'complex', 
                       'cross', 
                       'crossfunctional',
                       'customer', 
                       'data',
                       'dataset',
                       'datasets',
                       'deep', 
                       'demonstrated', 
                       'decision',
                       'development',
                       'difference',
                       'digital', 
                       'drive',
                       'driven',
                       'efficiency',
                       'end', 
                       'experience',
                       'expert',
                       'expertise',
                       'generate',
                       'functional',
                       'governance',
                       'growth',
                       'help',
                       'identify', 
                       'impact',
                       'improve',
                       'improvement',
                       'innovative',
                       'insight',
                       'lead',
                       'leader',
                       'leading',
                       'leadership',
                       'make', 
                       'manage',
                       'management',
                       'managing',
                       'matter',
                       'member',
                       'multidisciplinary',
                       'need',
                       'objective',
                       'operational', 
                       'opportunity',
                       'partner',
                       'problem',
                       'problemsolving',
                       'process',
                       'product',
                       'professional',
                       'project', 
                       'proven',
                       'question',
                       'record',
                       'requirement',
                       'risk',
                       'science',
                       'service',
                       'skill',
                       'solution',
                       'solve',
                       'solving',
                       'stakeholder',
                       'strategy',
                       'subject',
                       'success',
                       'team',
                       'thought',
                       'track',
                       'transform',
                       'transformation',
                       'understanding',
                       'use',
                       'user',
                       'value',
                       'work',
                       'working']


ds_network_skill_terms = []



stops_permanent_hold = list(set(['internal','external', 'new', 'capital', 'part', 'across', 'multiple', 'plus', 'clinical', 'trial',
                        'strong', 'high', 'quality', 'additional', 'unit', 'would', 'like', 'wide', 'variety',
                        'may', 'include', 'core', 'like', 'qualification', 'fair', 'chance', 'include', 'limited',
                        'per', 'small', 'least', 'key', 'support', 'best', 'policy', 'real', 'estate', 'may', 'required',
                        'world', 'extensive', 'senior', 'looking', 'globe', 'monday', 'friday', 'hand',
                        'million', 'lockheed', 'martin', 'leverage', 'comfortable', 'maintaining', 'broad', 'good',
                        'sr', 'long', 'term', 'must', 'ensure', 'following', 'area', 'social', 'medium', 'chance',
                        'ordinance', 'conversational', 'getty', 'highly', 'motivated', 'may', 'also', 'largest',
                        'committed', 'translate', 'essential', 'duty', 'due', 'drive', 'distributed', 'primary',
                        'committed', 'multiple', 'amount', 'provide', 'assurance', 'vast', 'trove', 'disruption',
                        'fortune', '500', 'drive', 'level', 'shape', 'least', 'los', 'angeles', 'ensure',
                        'credit', 'card']))


stops_other_hr_hold = ['senior', 'closely', 'federal', 'state', 'local', 'laws', 'preferred', 'qualification',
                       'join', 'u', 'applicant', 'national', 'location', 'duty', 'responsibility',
                       'least', 'one', 'individual', 'disability', 'change', 'healthcare', 'minimum', 'qualification',
                       'benefit', 'package', 'law', 'health', 'care', 'type', 'fulltime', 'employee', 'applicant',
                       'diversity', 'inclusion', 'responsibility', 'include', 'physical', 'mental', 'wide', 'range',
                       'must', 'able', 'following', 'consider', 'qualified', 'request', 'accommodation',
                       'diverse', 'inclusive', 'basic', 'paid', 'parental', 'leave', 'application', 'process',
                       'full', 'committed', 'providing', 'remote', 'required', 'basis', 'national',
                       'health', 'safety', 'place', 'competitive', 'salary', 'range', 'perform', 'essential', 'key', 'responsibility',
                       'creed', 'national', 'ancestry', 'federal', 'contractor', 'preferred', 'worklife', 'balance',
                       'contact', 'accommodation', 'request', 'able', 'responsible', 'travel', 'requirement',
                       'offer', 'competitive', 'pay', 'transparency', 'committed', 'creating', 'start', 'date',
                       'background', 'check', 'duty', 'assigned', 'public', 'health', 'retirement', 'plan',
                       'expression', 'citizen', 'authorized', 'discrimination', 'harassment', 'regulation',
                       'workplace', 'remotely', 'equity', 'inclusion', 'role', 'company', 'previous', 'position', 'requires',
                       'location', 'paid', 'holiday', 'tuition', 'reimbursement', 'opportunity', 'hire',
                       'human', 'resource', 'cover', 'letter', 'home', 'reporting', 'need', 'assistance',
                       'comprehensive', 'benefit', 'citizenship', 'compensation', 'history', 'spending',
                       'position', 'summary', 'compensation', 'benefit', 'workforce', 'benefit', '401k', 'role',
                       'candidate', 'legally', 'recruiting', 'religious', 'belief', 'match', 'paid', 'parental',
                       'assistance', 'accommodation', 'visa', 'sponsorship', 'access', 'compensation', 'package',
                       'culture', 'genetics', 'notice', 'opportunityaffirmative', 'seeking', 'recruitment', 'hiring',
                       'authorization', 'characteristic', 'offer', 'scratch', 'group', 'employee', 'navigate',
                       'provide', 'proof', 'providing', 'opportunity', 'request', 'personal', 'health', 'saving',
                       'eligible', 'employee', 'currently', 'seeking', 'requires', 'north', 'america'] 

consideration = ['building', 'environment'] # financial, service | experience, building | work, environment | supply, chain | essential, function

c19 = ['proof', 'vaccination', 'executive', 'order', 'requirement']

ds_task_terms = ['processing', 'model']
python_library_terms = ['numpy', 'pandas']




# execute cleaning and field parsing
df_raw        = load_and_concat_csvs(csv_path)
calculate_raw_csv_stats(df_raw)
df_clean      = clean_raw_csv(df_raw)
df            = parse_date_scraped_field(df_clean)
series_of_interest = df['job_description']
terms_for_nlp  = clean_for_nlp(series_of_interest)
visualize_indeed_data(df)

# execute nlp
n_gram_count = 2
n_gram_range_start, n_gram_range_stop  = 800, 900
n_grams = count_n_grams(terms_for_nlp, n_gram_count, n_gram_range_start, n_gram_range_stop)

visualize_word_clouds(terms_for_nlp)




# clean up intermediate dataframes and variables
del df_raw, df_clean, n_gram_count, n_gram_range_start, n_gram_range_stop











#######  ARCHIVE ######
# csv_list = []

# for csv in all_csvs:
#     csv_temp = pd.read_csv(csv, index_col=None, header=0)
#     csv_list.append(csv_temp)

# # concatnate csvs into a single dataframe
# df_raw = pd.concat(csv_list, axis=0, ignore_index=True)


