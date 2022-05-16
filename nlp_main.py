# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 12:53:56 2021

@author: Cary Cox, Ph.D.

This program performs a range of natural language processing and visuzalization tasks for a large dataset comprised 
of job listings scraped from the Indeed.com website. The general data flow is as follows:
    
                       raw csv Indeed data ingested ->
    raw csv Indeed data cleaned and readied for nlp ->
      visualizations generated for the raw csv data ->
                        natural language processing -> 
                      visualization of nlp findings    
    
"""

# import libraries for admin tasks
import os
import time
import warnings

# import libraries for data processing
import glob
import numpy as np
import pandas as pd
from progressbar import progressbar
from tqdm import tqdm

# import libraries for visualization
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import textwrap
from wordcloud import WordCloud #, STOPWORDS, ImageColorGenerator

# import libraries for nlp
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize  #  nltk.download('punkt') in shell after import nltk
import re
import string
import unicodedata

# initiate processing time calculation
start_time = time.time()

# remove row and column display restrictions in the console
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# suspend filter warnings
warnings.filterwarnings("ignore")

# configure tqdm progress bars for pandas commands
tqdm.pandas()

def load_and_concat_csvs(csv_path):
    '''
    Load and concatenate the corpus of Indeed csvs containing job data. Generates and displays high-level stats for 
    the imported data.

    Parameters
    ----------
    csv_path : string
        Parameter pointing to the location of the stored csvs.

    Returns
    -------
    df_raw : dataframe
        Contains the raw concatenated csvs.

    '''
    print('\nLoading and concatenating Indeed csvs...')
    
    # load and concatenate all Indeed csvs while adding a field for each record's parent csv name
    all_csvs = glob.glob(csv_path + "/*.csv")
    df_raw = pd.concat([pd.read_csv(fp).assign(csv_name=os.path.basename(fp)) for fp in progressbar(all_csvs)])

    # calculate and display high-level statistics for the imported data
    print('\n***** Data Ingest Statistics ***** \n')
    print(f'csvs imported: {len(all_csvs)} \n')
    print(f'Job listings imported: {df_raw.shape[0]} \n')
    print(f'Unique job titles: {df_raw.job_title.nunique()} \n')
    print(f'Nulls are present:\n{df_raw.isna().sum()} \n')
    print(f'Records missing job_title field: {(df_raw.job_title.isna().sum() / df_raw.shape[0] * 100).round(3)}%')
    print(f'Records missing job_Description field: {(df_raw.job_Description.isna().sum() / df_raw.shape[0] * 100).round(3)}% \n')
    print('***** Data Cleaning ***** \n')
    print(f"Count of duplicates based on company, location, title and description: {df_raw.duplicated(subset=['job_title', 'company', 'location', 'job_Description']).sum()}")
    print(f"Duplication rate: {((df_raw.duplicated(subset=['job_title', 'company', 'location', 'job_Description']).sum()) / df_raw.shape[0] * 100).round(3) }% \n")

    return df_raw


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
    df : dataframe
        The primary dataframe for the concatenated, cleaned and parsed Indeed csv data.

    '''
    # drop unnecessary fields and repair job_Description field name
    df_clean = df_raw.drop(['URL', 'page_count', 'post_date', 'reviews'], axis=1)
    df_clean.rename(columns={'job_Description':'job_description'}, inplace=True)

    # drop duplicates based on key fields and assert that the drop duplicates artithmetic worked corrrectly
    df_clean = df_clean.drop_duplicates(subset=['job_title', 'company', 'location', 'job_description'])
    assert (len(df_raw) - len(df_clean))   ==  (df_raw.duplicated(subset=['job_title', 'company', 'location', 'job_Description']).sum())
    
    # drop records that have NaN for job_title, company, location or job_description
    df_clean.dropna(subset = ['job_title', 'company', 'location', 'job_description'], inplace=True)
    
    # reset the index and report the number of unique records remaining after initial cleaning
    df_clean.reset_index(inplace=True, drop=True)
    print(f'{len(df_clean)} records remaining after intial data cleaning')
    
    def clean_and_parse_date_scraped_field(df_clean):
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
            The primary dataframe for the concatenated, cleaned and parsed Indeed csv data.

        '''
        # convert csv_name field to a string 
        df_clean['csv_name'] = df_clean['csv_name'].astype(str)
      
        # fix single zero in csv name
        df_clean['csv_name'] = [x.replace('_0_', '_000_') if (len(x) != 24) else x for x in df_clean['csv_name']]
        
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
    
    # parse the date_scraped field from the Indeed csvs, and create the ready-for-nlp dataframe, df     
    df = clean_and_parse_date_scraped_field(df_clean)
    
    return df


def clean_terms_for_nlp(series_of_interest):
    '''
    Execute stopword removal, lowercasing, encoding/decoding, normalizing and lemmatization in preparation for NLP.
    
    Data Flow:
        series_of_interest converted into a single string *text_as_string > 
        *text-as_string normalized, split and lowercased >
        *text-as_string converted to a list of single *words >
        
        *additional_stopwords list created to capture industry-specific stopwords >
        standard English *stop_words created from NLTK >
        
        *words cleansed of terms in *stop_words >
        *words lemmatized to create the *terms_for_nlp list, a list of lemmatized terms >
        *additional_stopwords dropped from *terms_for_nlp >
        
        *term_fixes dictionary created to map misspellings, ambiguities, etc. to preferred terms >
        *terms_for_nlp converted to a single-series dataframe, *df_term_fixes >
        *df_term_fixes remapped according to the *term_fixes dictionary as a final NLP cleaning step >
        *df_term_fixes converted back into a list as the final instantiation of *terms_for_nlp <>

    Parameters
    ----------
    series_of_interest : series
        A variable set in the main program, series_of_interest contains the targeted job listing data for NLP processing.

    Returns
    -------
    terms_for_nlp : list
        A list containing all terms (fully cleaned and processed) extracted from the series_of_interest Series.
    
    additional_stopwords : list
        A list to capture all domain-specific stopwords and 'stop-lemma'
    
    term_fixes : dictionary
        A dictionary for correcting misspelled, duplicated or consolidated terms in the series of interest

    '''    
    print('\nCleaning data for nlp:')
    
    # convert parsed series to a single string; not that this destroys the integrity between term and job listing
    text_as_string = ''.join(str(series_of_interest.tolist()))
    
    # normalize, split and lowercase the parsed text
    print('   Normalizing, splitting and lowercasing...')
    text_as_string = (unicodedata.normalize('NFKD', text_as_string).encode('ascii', 'ignore').decode('utf-8', 'ignore').lower())
    words = re.sub(r'[^\w\s]', '', text_as_string).split()
    
    # add additional stopwords to nltk default stopword list; counts between 20 and 50 are evaluated but not included in the lists
    # counts of 19 and below are neither evaluated nor included in the stopword list or skill lists
    additional_stopwords = sorted(list(set(['14042', '3rd', '401k', '50', '500', 'a16z', 'able', 'accepted',
                                            'accommodation', 'accomodation', 'account', 'across',
                                            'action', 'additional', 'adhering', 'affiliation', 'affirmative',
                                            'age', 'allen', 'also', 'amazecon', 'america', 'amount', 'ancestry',
                                            'andor', 'angeles', 'another', 'applicable', 'applicant',
                                            'apply', 'area', 'around',
                                            'arrest', 'assigned', 'assistance', 'assurance', 'authentic', 'authorization',
                                            'authorized', 'background', 'balance', 'base', 'based', 'basic', 'basis',
                                            'belief', 'belonging', 'benefit', 'beyond', 'billion', 'bonus',
                                            'booz', 'broad', 'california', 'call', 'candidate', 'cannot', 'capital',
                                            'card', 'care', 'chance', 'characteristic', 'chase', 'check',
                                            'chicago', 'childbirth', 'citizen', 'citizenship', 'city', 'civil',
                                            'classified', 'click', 'clinical', 'closely', 'color', 'colorado', 'come',
                                            'comfortable', 'commitment', 'committed', 'commuter', 'company', 'compensation',
                                            'competitive', 'complaint', 'compliance', 'comprehensive', 'confidential',
                                            'consider', 'consideration', 'considered', 'consistent', 'contact', 'contractor',
                                            'conversation', 'conversational', 'conviction', 'core', 'cover', 'covid',
                                            'covid19', 'creating', 'credit', 'creed', 'criminal', 'culture', 'current',
                                            'currently', 'date', 'day', 'dc', 'december', 'dedicated', 'defense',
                                            'demand', 'dental', 'deploying', 'description', 'disability', 'disclose',
                                            'disclosed', 'disclosure', 'discriminate', 'discrimination', 'discussed',
                                            'disruption', 'diverse', 'diversity', 'domestic', 'drive',
                                            'drugfree', 'drugtesting', 'due', 'duty', 'eeo', 'eg', 'eligibility',
                                            'eligible', 'email', 'embracing', 'employee', 'employeeled', 'employer',
                                            'employment', 'encouraged', 'enjoy', 'ensure', 'equal', 'equity', 'essential',
                                            'estate', 'etc', 'every', 'everyone', 'existing', 'expression',
                                            'extensive', 'external', 'fair', 'family', 'fargo', 'federal', 'feel', 'following',
                                            'fortune', 'francisco', 'friday', 'fulltime', 'fully', 'furnish',
                                            'furtherance', 'gender', 'genetic', 'genetics', 'getty', 'globe', 
                                            'good', 'group', 'growing', 'hand', 'harassment', 'health', 'healthcare',
                                            'hearing', 'high', 'highly', 'hire', 'hiring', 'history', 'holiday', 'home',
                                            'host', 'hour',                                            
                                            'human', 'ibm', 'id', 'identity',
                                            'il', 'include', 'including', 'inclusion', 'inclusive', 'indepth',
                                            'inquired', 'inside', 'insurance', 'internal', 'job', 'johnson', 'join',
                                            'jpmorgan', 'kept', 'key', 'kpmg', 'largest', 'law', 'laws', 'least', 'leave',
                                            'legally', 'letter', 'leverage', 'life', 'lightspeed', 'like', 'limited',
                                            'local', 'location', 'lockheed', 'long', 'looking', 'los', 'love', 'm', 'made',
                                            'maintaining', 'make', 'mandate', 'manner', 'marital', 'martin', 'match',
                                            'matching', 'mature', 'may', 'medical', 'medium', 'mental', 'million', 'minimum',
                                            'monday', 'multiple', 'must', 'national', 'navigate', 'need',
                                            'new', 'nondiscrimination', 'nonessential', 'north', 'notice', 'offer', 'one',
                                            'opportunity',  'order', 'ordinance', 'orientation',
                                            'origin', 'outside', 'overview', 'package', 'paid', 'pandemic', 'parental',
                                            'part', 'participate', 'party', 'pay', 'per', 'perform', 'performed',
                                            'perk', 'personal', 'phone', 'place', 'plan', 'please', 'plus',
                                            'point', 'policy', 'political', 'position', 'posse', 'poster', 'preemployment',
                                            'preferred', 'pregnancy', 'premier', 'prescribe', 'previous', 'primary', 'prior',
                                            'privacy', 'privilege', 'proceeding', 'proof', 'protected', 'proud',
                                            'provide', 'providing', 'public', 'puerto', 'purchase', 'qualification', 'qualified',
                                            'race', 'range', 'rapidly', 'real', 'reasonable', 'receive', 'recruiter',
                                            'recruiting', 'recruitment', 'referral', 'regard', 'regarding', 'regardless',
                                            'regulation', 'regulatory', 'reimbursement', 'relic', 'religion', 'religious',
                                            'relocation', 'remotely', 'reporting', 'report', 'req', 'request', 'required',
                                            'requires', 'resource', 'responsibility', 'responsible', 'resume',
                                            'retirement', 'reward', 'rico', 'role', 'safety', 'salary', 'salesforcecom',
                                            'salesforceorg', 'san', 'saving', 'schedule', 'scratch', 'secret', 'seeking',
                                            'self', 'sending', 'senior', 'sense', 'sequoia', 'sex', 'sexual', 'shape',
                                            'show', 'sincerely', 'small', 'social', 'someone', 'sound', 'spending',
                                            'sponsorship', 'sr', 'standard', 'start', 'state', 'statement', 'status', 'stay',
                                            'stock', 'suite', 'summary', 'supplemental', 'supply', 'support', 'sure',
                                            'suspended', 'talented', 'teladoc', 'tenure', 'term', 'therapeutic', 'third',
                                            'total', 'toughest', 'transgender', 'translate', 'transparency', 'travel', 'trial',
                                            'trove', 'tuition', 'type', 'u', 'union', 'unit', 'united', 'unitedhealth', 'vaccine',
                                            'unsolicited', 'upon', 'using', 'vaccinated', 'vaccination', 'variety', 'vast',
                                            'veteran', 'visa', 'visit', 'washington', 'way', 'wed', 'well', 'live',
                                            'wellbeing', 'wellness', 'whats', 'wide', 'within', 'without', 'workforce',
                                            'worklife', 'workplace', 'world', 'would', 'york', 'youll', 'zone', 'view', 'note',
                                            'achieve', 'goal', 'organization', 'future', 'sourcing', 'offering', 'throughout',
                                            'choice', 'let', 'know', 'immigration', 'available', 'important',
                                            'government', 'agency', 'financial', 'institution', 'resolve', 'issue', 'active',
                                            'leveraging', 'drug', 'free', 'monitor', 'successful', 'completion', 'community', 'serve',
                                            'hired', 'accenture', 'chief', 'officer', 'investigation', 'otherwise', 'unless',
                                            'right', 'thing', 'better', 'response', 'formal', 'charge', 'b', '2021',
                                            'conducted', 'legal', 'placing', 'manager', 'talent', 'firm', '100', 'ongoing',
                                            'ethnicity', 'conference', 'resident', 'submitting', 'acknowledge', 'mix', 'building',
                                            'celebrates', 
                                            'vacation', 'sick', 'january', '2022', 'tiger', 'global', 'get', 'done', 'via',
                                            'top', 'internally', 'externally', 'performance', 'indicator', 'thrive',
                                            'continue', 'grow', 'faculty', 'staff', 'bring', 'closer', 'result', 'space',
                                            'virtual', 'assistant', 'approved', 'save', 'money', 'create',
                                            'various', 'production', 'activity', 'take', 'department', 'provides',
                                            'familiarity', 'others', 'assist', 'needed', 'enable', 'believe', 'effective',
                                            'different', 'planning', 'task', 'want', 'supporting', 'appropriate', 'consumer',
                                            'effort', 'define', 'conduct', 'potential', 'used', 'patient', 'inc',
                                            'find', 'documentation', 'finance', 'similar', 'first', 'specific', 'share',
                                            'deployment', 'includes', 'require', 'focused', 'act', 'implementing', 'desired',
                                            'organizational', 'person', 'many', 'brand', 'content', 'address',
                                            'directly', 'driving', 'execution', 'colleague', 'general', 'online', 'addition',
                                            'asset', 'commercial', 'meaningful', 'purpose', 'ideal', 'today', 'investment',
                                            'monitoring', 'provider', 'interest', 'event', 'seek', 'assessment', 'necessary',
                                            'country', 'option', 'revenue', 'execute', 'corporate', 'ensuring', 'direction',
                                            'youre', 'performing', 'enhance', 'component', 'significant', 'possible', 'give',
                                            'complete', 'site', 'form', 'guide', 'student', 'common', 'contract',
                                            'number', 'changing', 'two', 'week', 'embrace', 'furthering', 'submitted', 'force',
                                            'box', 'annual', 'reinforced', 'maintains', 'accordance', 'protection', 'requesting',
                                            '190', 'chapter', 'coordinated', 'increasingly', 'chapter', 'globally', 'affinity',
                                            'reaching', 'banking', 'junior', 'moving', 'undue', 'hardship', 'assign',
                                            'betterrounded', 'strive', 'assign', 'remain', 'carefully', 'guaranteed', 'mitre',
                                            'shortterm', 'longterm', 'getting', 'started', 'link', 'contains', 'atlanta', 'ga',
                                            'complies', 'category', 'attached', 'actionequal', 'llp', 'international', 'dfj',
                                            'infectious', 'disease', 'fit', 'reach', 'vary', 'depending', 'funded', 'investor',
                                            'vice', 'president', 'remind', 'put', 'according', 'guideline', 'increasingly',
                                            'requiring', 'venture', 'accel', 'men', 'woman', 'sdtm', 'adam', 'austin', 'tx',
                                            'none', 'listed', 'since', '1911', 'pride', 'facebooks', 'facebook', '16', 'copy',
                                            'charlotte', 'nc', 'heart', 'everything', 'contain', 'volunteer', 'charitable',
                                            'promotion', 'termination', 'society', 'reason', 'discount', 'retail', 'ibmibms',
                                            'reinventing', 'biggest', 'invention', 'ibmer', 'journey', 'run', 'signed', 'agreement',
                                            'worldrestlessly', 'worldclass', 'educational', 'liable', 'thirdparty', 'several',
                                            'bringing', 'hybrid', 'relying', 'bringing', 'relying', 'statementibm',
                                            'personalized', 'ibms', 'ibmers', 'restless', 'depth', 'empower', 'depth', 'select',
                                            'facilitate', 'guarantee', 'continued', 'counseling', 'consisting', 'bonding',
                                            'giving', 'obligated', 'sealed', 'expunged', 'rich', 'discriminated', 'asked',
                                            'observance', 'toll', 'speak', 'employed', 'affiliated', 'send', 'employed',
                                            'sealed', 'shall', 'mayo', 'clinic', 'lottery', 'except', 'casual', 'dress',
                                            'lawfacebook', 'goldman', 'sachs', 'jp', 'morgan',
                                            'private', 'sector', 'mutual', 'mount', 'sinai', 'exhaustive', 'list', 'provided',
                                            'interview', 'unlimited', 'pto', 'keep', 'facing', 'silicon', 'valley', 'adequate',
                                            'protocol', 'imperative', 'sustain', 'pwc', 'pwcsponsored', 'signon', 'restricted',
                                            'mayo', 'clinic', 'h1b', 'lottery', 'canada', 'saics', 'saic', 'ie', 'collateral',
                                            'vehicle', 'unwavering', 'aim', 'discharge', 'additionally', 'audit',
                                            'accept', 'headhunter', 'insider', 'threat', 'mobile', 'app',
                                            'described', 'representative', 'weve', 'got', 'st', 'louis', 'skillsexperience',
                                            'leaf', 'absence', 'crucial', 'happiness', 'participates', 'everify',
                                            'striking', 'healthy', 'verizon', '41', 'cfr', 'lot', 'fun', 'sharing',
                                            'recognize', 'strength', 'constantly', 'strength', 'linked', 'vacancy', 'apps',
                                            'announcement', 'salesforce', 'einstein', 'landscape', 'builder', 'constantly',
                                            'expand', 'whether', 'array', 'celebrate', 'stronger', 'twitter', 'instagram',
                                            'kind', 'connects', 'arlington', 'va', 'prepared', 'ready', 'voice', 'carves',
                                            'factor', 'starting', 'influenced', 'path', 'us', 'discover', 'possibility',
                                            'unleash', 'ready', 'express', 'craving', 'prepared', 'actual', 'influenced',
                                            'carves', 'ibmare', 'possibility', 'voice', 'cocreate', 'individuality', 'add',
                                            'alter', 'fabric', 'truly', 'personally', 'hundred', 'thousand', 'month',
                                            'layoff', 'recall', 'ethnic', 'sensory', 'past', 'operate', 'there', 'always',
                                            'edward', 'jones', 'nice', 'have', 'determine', 'exceed', 'expectation', 'harmony',
                                            'predisposition', 'carrier', 'dealing', 'listing', 'figure', 'backwards',
                                            'vigorously', 'hiv', 'past', 'organizationleaders', 'thirteen',
                                            '85000', 'property', 'casualty', 'mass', 'brigham', 'driver', 'license',
                                            'second', 'dose', 'investigator', 'pittsburgh', 'pa', 'attract', 'retain',
                                            'catered', 'lunch', 'booster', 'shot', 'maternity', 'prohibited', 'behavior',
                                            'exceptional', 'aspect', 'expected', 'discovery', 'setting', 'material',
                                            'produce', 'see', 'academic', 'startup', 'exciting', 'recognized',
                                            'contribution', 'answer', 'creation', 'play', 'robust', 'payment', 'integrate',
                                            'class', 'explore', 'overall', 'establish', 'move', 'integrated', 'three', 'desire',
                                            'obtain', 'pricing', 'regular', 'maintenance', 'interested', 'along', 'increase',
                                            'utilize', 'manufacturing', 'built', 'acquisition', 'thats', 'submit',
                                            'coordinate', 'prepare', 'utilizing', 'foundation', 'encourage', 'cycle',
                                            'vendor', 'become', 'device', 'smart', 'ass', 'effectiveness', 'mean',
                                            'adoption', 'located', 'preferably', 'division', 'mindset', 'scope', 'exposure',
                                            'progress', 'transforming', 'lab', 'enabling', 'interaction',  'look', 'promote',
                                            'example', 'foster', 'specification', 'associated', 'daily', 'realtime', 'campaign',
                                             'storage', 'competency', 'website', 'medicine', 'incentive', 'follow',
                                            'facility', 'coverage', 'validate', 'continuously', 'defining', 'bank', 'serving',
                                            'claim', 'entire', 'outstanding', 'successfully', 'excited', 'conducting',
                                            'actively', 'input', 'realworld', 'gain', 'principal', 'onsite', 'towards', 'selection',
                                            'accelerate', 'among', 'output', 'worldwide', 'generous', 'channel',
                                            'video', 'special', 'taking', 'advertising', 'selected', 'custom', 'posting', 'title',
                                            'accountability', 'corporation', 'especially', 'tracking', 'target', 'industrial',
                                            'advantage', 'secure', 'performs', 'game', 'transportation', 'five', 'founded', 'art',
                                            'survey', 'format', 'equipment', 'even', 'specialist', 'creates', 'situation',
                                            'simple', 'nation', 'safe', 'shared', 'delivers', 'step', 'ecommerce', 'visual',
                                            'applies', 'sustainable', 'dod', 'topic', 'personnel', 'dont',
                                            'assignment', 'welcome', 'willing', 'phase', 'preference', 'combine', 'accessible',
                                            'defined', 'back', 'specifically', 'budget', 'evaluating', 'either', 'encourages',
                                            'given', 'rule', 'established', 'gathering', 'familiar', 'connected', 'recommend',
                                            'update',  'transfer', 'context', 'could', 'ensures',  'cancer', 'expect',
                                            'definition', 'mind', 'gather', 'environmental', 'strongly', 'enables', 'skilled',
                                            'pharmaceutical', 'preparation',  'usa', 'single', 'turn', 'submission',
                                            'laboratory', 'usage', 'ca', 'serf', 'posted', '25', 'allows', 'deloitte', 'close',
                                            'kafka', 'brings', 'headquartered', '40', 'sophisticated', 'distribution', 'hr',
                                            'handle', 'stage', 'connection', 'streaming', 'four', 'traditional', 'powerful',
                                            'marketplace', 'specialized', 'kpis', 'electronic', 'capacity', 'file',
                                            'consistently', 'nature', 'valued', 'informed', 'american', 'oversight', 'valuable',
                                            'spectrum', 'campus', 'highest', 'desirable', 'exemption', 'verification', 'supported',
                                            'reduce',  'align', 'broader', 'integrating', 'entity', 'succeed', 'demonstrates',
                                            'economic', 'enhancement', 'executing', 'capable', 'internet', 'allow', 'joining',
                                            'advancement', 'transaction', 'inventory', 'capture', 'involved', 'reference', 'much',
                                            'whole', 'institute', 'industryleading', 'contributing', 'food', 'gap', 'compelling',
                                            'hospital', 'period', 'never', 'timeline', 'proprietary', 'scaling', 'typically',
                                            'monthly', 'engaging', 'sponsor', '2020', 'entertainment', 'ground', 'offered',
                                            'dependent', 'inspire', 'discussion', 'awareness', 'involving', '15', 'profile',
                                            'something', 'advisory', 'running', 'employ', 'programmer', 'tackle',  'alignment',
                                            'boston', 'name', 'regularly', 'sport', 'central', 'le', 'deeply', 'launch', 
                                            'particular', 'accounting', 'received', '30', 'english', 'logistics', 'emphasis',
                                            'however', 'size',  'empowers', 'opening', 'personalization', 'alternative', 
                                            'population', 'liremote', 'rewarding', 'session', 'protect', 'yes', 'short', 'post', 
                                            'maximize',  'thinker', 'release', 'primarily', 'propose', 'forefront', 'return',
                                            'final', 'news', 'rate', 'anywhere', 'empowered', 'descriptive', 'studio',
                                            'accountable', 'influencing', 'massive', 'typical', 'treatment', 'commerce',
                                            'manipulating', 'backend', 'querying', 'handling', 'amazing', 'award', 'hold', 
                                            'weekly', 'retention', 'bias', 'immediate', 'teaching', 'resolution', 'spend',
                                            'hear', 'easy', 'superior', 'tuning', 'nearly', 'promotes', 'demonstrate',
                                            'satisfaction', 'unlock', 'behind', 'screening', 'begin', 'inquiry',  'page',
                                            'particularly', 'bestinclass', 'demonstrating', 'element', 'ask', 'demonstrated',
                                            'availability', 'oncology', 'specialization', 'empowering', 'criterion', 'loan',
                                            'pioneering', 'advancing', 'exploring', 'aptitude', 'stand', 'institutional', 
                                            'comply', 'respond', 'derive', 'message',  'easily', 'recent', 'journal', 'lending',
                                            'networking', 'continually', 'workshop', 'economy', 'completed', 'choose',
                                            'utilization', 'alongside', 'demonstrable',  'administrative', 'named', 'created',
                                            'last', 'dream', 'requisition', 'invest', 'section', 'apple', 'combining', 'committee',
                                            'held', 'authority', 'deal',  'contributes', 'daytoday', 'iot', 'toward',
                                            'achieving', 'learner', 'spirit', 'fee', 'agent', 'certain',  'cc', 'transition',
                                            'roadmaps', 'champion', 'virginia', 'region', 'failure', 'establishing',
                                            'comfort', 'eager',  'combined', 'contingent', 'instruction', 'powered', 'utility',
                                            'functionality', 'worked', 'ideally', 'coordination', 'intended', 'underlying',
                                            'proper', 'operates', 'refine', 'unable', 'represent', 'might', 'advocate',
                                            'youve', 'mobility', 'wealth', 'aligned', 'found', 'known', 'funding', 'embedded',
                                            'fostering', 'fuel', 'intermediate',  'awardwinning', 'often', 'tomorrow', 'carry',
                                            'synthesize', 'smarter', 'quo', 'scalability', 'grant', 'useful', 'routine',
                                            'workload', '200', 'foundational', 'reusable', 'retrieval', 'talk', 'incumbent',
                                            'pursue', 'attribute', 'retailer', 'span', 'disabled', 'board', 'face', 'zeta', 'co',
                                            'met', 'validating', 'shaping', 'tactical', 'sample', 'ambitious', 'hub', 'aid',
                                            'presence', 'command', 'subsidiary', 'participation', 'anticipate', 'therefore',
                                            'supplier', 'enablement', 'backed', 'rating', 'advise', 'preparing', 'little', 
                                            'accessibility', 'incorporate', 'beginning', 'car', 'everywhere', 
                                            'supportive', 'departmental', 'newly', 'autonomy', 'still', 'varying', 'sustainability',
                                            'texas', 'improved', 'included', 'macro', 'approximately', 'difficult', 'interpreting',
                                            'inhouse', 'generating', 'relation', 'fund', 'already', 'implemented', 'highlevel',
                                            'huge', 'exchange', 'inspired', 'net', 'legacy', 'fl', 'continues', 'premium', 
                                            'realize', 'covered', 'reality', 'verify', 'configuration', 'llc', 'physician',
                                            'accomplish', 'headquarters', 'breadth', 'membership', 'trading', 'visibility',
                                            'room', 'migration', 'head', 'parent', 'true', 'pursuit', 'spotify', '13',
                                            'producing', 'md', 'ultimately', 'ever', 'appointment', 'engaged', 'craft',
                                            'ambition', 'fellow', 'fastest', 'yet', 'specialty', 'approval', 'paypal',
                                            'main', 'prospective', 'affordable', 'affect', 'feed', 'tv', 'expanding', 
                                            'continuing', 'completing', 'frequently', 'connecting', 'investigate', 'audio',
                                            'safer', 'display', 'focusing', 'decade', 'liaison', 'onboarding', 'grade', 'enabled',
                                            'ideation', 'led', 'worker', 'transcript', 'participating', 'article', 'ranging',
                                            'rely', 'bold', 'equality', 'fashion', 'integral', 'requested', 'evolution', 
                                            'appropriately', 'valid', 'advice', 'adhere', 'discus', 'exception',
                                            'actuarial', 'collective', 'coordinating', 'addressing', '9', 'motor', 'assisting',
                                            'biomedical', 'targeting', 'screen', 'cultural', 'promise',
                                            'initial', 'wayfair', 'living', 'ops', 'concern', 'cell', 'music', 'stipend', 'invite',
                                            'determined', 'profiling', 'eye', 'merit', 'represents', 'assessing',
                                            'emergency', 'freedom', 'numerous', 'friendly', 'technologist', 'permanent',
                                            'placement', 'insightful', 'low', 'citi', 'formulate', 'allowing', 'deployed',
                                            'involves', 'front', 'manual', 'recruit', 'pioneer', 'increasing', 'happy', 'reflects',
                                            'dedication', 'construct', 'implication', 'identified', 'convey', 'movement',
                                            'selfservice', 'occasionally', 'summarize', 'pursuant', 'involve', 'larger', 'price',
                                            'fresh', 'achievement', 'varied', 'teach', 'competing', 'eoe', 'blue', 'operationalize',
                                            'extraordinary', 'practitioner', 'spanning', 'style', 'readiness', 'prevent', 'pfizer',
                                            'enhancing', 'proposed', 'automotive', 'child', 'ic', 'prohibits', 'keen', 'log',
                                            'limitation', 'selling', 'frame', 'equitable', 'square', 'subscription', 'interesting',
                                            'tradeoff', 'expense', 'describe', 'ranking', 'really', 'reviewing', 'keeping',
                                            'streamline', 'ranked', 'reserve', 'ny', 'nj', 'conflict',   'atmosphere',
                                            'chemical', 'therapy', 'robotic', 'blog', 'targeted', 'searching', 'extend',
                                            'ten', 'recently', 'staffing', 'measuring', 'sop', 'filled', 'exhibit', 'ii',
                                            'transparent', 'representation', 'entry', 'opinion', 'electric', 'speaking',
                                            'consultation', 'planet', 'consumption', 'climate', 'consistency', 'sit', 'sensitive',
                                            'picture', 'dollar', 'fastgrowing', 'arrangement', 'regional', 'closing', 'enjoys',
                                            'knowledgeable', 'attribution', 'break', 'fitness', 'europe', 'thrives', 'increased',
                                            'milestone', 'incredible', 'everyday', 'researching', 'foreign', 'minority',
                                            'nonprofit', 'construction', 'simplify', 'suitable', 'pharmacy', 'whose', 
                                            'exempt', 'item', 'cro', 'native', 'intuitive', 'consult', 'participant', 'prevention',
                                            'official', 'conversion', 'fact', 'ahead', 'diagnostic', 'healthier', 'deloittes',
                                            'moment', 'gsk', 'philadelphia', 'frontend', 'barrier', 'intersection', 'determining',
                                            'directed', 'inperson', 'trade',  'forbes', 'fill', 'enhanced',
                                            'prepares', 'az', 'correct', 'importance', 'significantly', 'educate', 'reputation',
                                            'here', 'effect', 'perfect', 'pet', 'tight', 'guiding', 'uncertainty', 'side',
                                            'scheduling', 'diagnostics', 'merchant', 'interacting', 'labor', 'currency', 'fintech',
                                            'street', 'seeker', 'evaluates', 'standardization', 'reading', 'investing', 'workday',
                                            'circle', 'mode', 'supervisor', 'extremely', 'receiving', 'organizing', 'magazine',
                                            'outlined', 'commensurate', 'florida', 'isnt', 'factory', 'district',
                                            'parttime', 'pressure', 'occasional', 'pas', 'utilizes', 'calculation',
                                            'promoting', 'extended', 'dna', 'quarterly', 'snack', 'deeper', 'courage', 
                                            'acceptance', 'qa', 'perception', 'mitigate', 'nasdaq', 'attend', 'prospect', '24',
                                            'arm', 'air', 'shipping', 'fleet', 'score', 'penn', 'oh', 'gym', 'babs', '11', '60',
                                            'era', 'mitigation', 'introduction', 'civilian', '1st',  'surface', 'league', 'stored',
                                            'resulting', 'setup', 'lighthouse', 'narrative', 'vital', 'going', 'treat',
                                            'substantial', 'comparable', 'lift', 'seattle', 'e', 'highperforming', 'easier', '18',
                                            'urgency', 'delight', 'tailored', 'abreast', 'progressive', 'vibrant', 'book',
                                            'vertical', 'astrazeneca', 'tangible', 'loyalty', 'mechanism', 'lieu', 'competition',
                                            'nationwide', 'invent', 'owning', 'light', 'discovering', 'hope', 'fda', 
                                            'joint', 'pushing', 'generally', 'follows', 'touch', 'spring', 'relevance',
                                            'water', 'discretion', 'extent', 'acquire', 'respected', 'positively', 'corp',
                                            'transformational', 'walk', 'qualify', 'owned', 'defines', 'payer', 'guided',
                                            'popular', '2019', 'standing', 'nyse',  'fulfill', 'adaptable', 'cash', 
                                            'moderate', 'dell', 'customized', 'designated', 'highlight', 'pc', 'selecting',
                                            'messaging', 'outreach', 'pursuing',  'ceo', 'gi', 'agenda', 'elevate', 'historical',
                                            'transactional', 'original', 'sometimes', 'buy', 'stability', 'capgemini', 'modify',
                                            'consultative', 'incident', 'dallas', 'advertiser', 'ci', 'publishing', 'vp',
                                            'tackling', 'apart', 'v', 'tolerate', 'feasibility', 'uptodate', 'productiongrade',
                                            'roi', 'collectively', 'supplement', 'diving', 'guidehouse', 'tax', 'sdlc',
                                            'parameter', 'lasting', 'philosophy', 'balancing', 'cant', 'holding',
                                            'benchmark', 'cdisc', 'adjustment', 'thoughtful', 'publish', 'telework', 'fidelity',
                                            '14', 'shop', 'november', 'medicare', 'considers', 'modernization', 'lower', 'near',
                                            'commission', 'outlook', 'surveillance', 'mi', 'impacting', 'aimed', 'navy',
                                            'affiliate', 'duration', 'leveraged', 'demonstration', 'adept', 'blend', 'norm',
                                            'maintainable', 'buying', 'adherence', 'association', 'intent', 'underwriting',
                                            'interprets', 'clinician', 'shopper', 'imagine', 'confidentiality', 'happen',
                                            'circumstance', 'specializing', 'curriculum', 'denver', 'maryland', 'rare',
                                            'propensity', 'considering', 'dimension', '23', 'index', 'negotiation',
                                            'inspiring', 'speaker', 'crafting', 'tier', 'motion', 'academia', 'truth',
                                            'road', 'briefing', 'forum', 'creator', 'fedex', 'heavily', 'constant',
                                            'upload', 'pilot', 'theyre', 'allocation', 'traveler', 'manuscript', 'shopping',
                                            'limit', 'devise', 'provision', 'forth', 'serious', 'turning', 'catalog', 'competence',
                                            'understandable', 'pound', 'phoenix', 'scoring', 'coming', 'enrich', 'seven',
                                            'disruptive', 'behalf', 'covering', 'individually', 'welcoming', 'attendance',
                                            'strives', 'bar', 'omnichannel', 'afraid', 'brightest', 'holder', 'classical',
                                            'valuation', 'optional', 'obsessed', 'temporary', 'massachusetts', 'normal',
                                            'expansion', 'dig', 'grown', 'indeed', 'protecting', 'annually', 'club', 'consultancy',
                                            'frequent', 'seamlessly', 'epic', 'accelerating', 'disclaimer', 'questionnaire',
                                            'telephone', 'obtaining', 'enough', 'ultimate', 'sell', 'maturity',
                                            'anyone', 'fastestgrowing', 'unparalleled', 'say', 'onthejob', 'patent', 'employing',
                                            'ba', 'reflect', 'visible', 'recommended', 'linguistics', 'efficacy', 'uk', 'ui',
                                            'aviation', 'mandatory', 'graphical', 'simply', 'internship', 'concurrent',
                                            'moody', 'else', 'translates', 'cultivate', 'anccon', 'cognizant',
                                            'strengthen', 'raise', 'drawing', 'affair', 'internationally', 'door', 'portal',
                                            'seller', 'plant', 'buyer', 'accomplishment', '1000', 'traffic', 'intervention',
                                            'honest', 'specified', 'executes', 'w', 'published', 'army',
                                            'relates', 'eight', 'recommends', 'park', 'called', 'sitting',
                                            'fulfillment', 'na', 'walmart', 'administrator', 'ac_consulting21', 'verisk', 'spent',
                                            'michigan', 'slack', 'crime', 'anticipated', 'desk', '5000', 'derived',
                                            'grasp', 'dot', 'west', 'personalize', 'meaning', 'complicated', 'calling', 'bigger',
                                            'reducing', 'seamless', 'module', 'clarity', 'convenient', 'pennsylvania', 'integrates',
                                            'achieved', 'putting', 'rank', 'retaining', 'suggest', 'piece', 'throughput',
                                            'launched', 'loading', 'exist', 'john', 'customercentric', 'relating', 'truck', 
                                            'centered', '90', 'winning', '0', 'facilitation', 'enrollment', 'arise', 'rather',
                                            'inspires', '2018', 'evaluated', 'pro', 'footprint', 'doe', 'finish',
                                            'subcontractor', 'scoping', 'telecommunication', 'raised', 'b2b', 'informal', 'wa',
                                            'lifelong', 'ip', 'psychology', 'marketer', 'relentless', 'prefer', 'abuse',
                                            'export', 'billing', 'commonly', 'stress', 'db', 'south', 'lifetime', 'terraform',
                                            'coupled', 'constructive', 'demo', 'concerning', 'cambridge', 'mortgage',
                                            'half', 'documented', 'distill', 'bridge', 'wave', 'artifact', 'explaining',
                                            'consensus', 'facilitating', 'facilitates', 'steward', 'examine', 'convert', 'gaming',
                                            'performant', 'importantly', 'territory', 'acting', 'cdc', 'sharepoint', 'evangelize',
                                            'breaking', 'maintained', 'grid', 'average', 'cleanse', 'relocate', 'storm',
                                            'involvement', 'offline', 'scheduled', 'usable', 'pose', 'fix', 'paradigm', 'house',
                                            'remove', 'nordstrom', 'completeness', 'thriving', 'mile', 'nielsen', 'horizon',
                                            'manufacturer', 'significance', 'harness', 'thrasio', 'county', 'mentality',
                                            'fairness', 'sufficient', 'usability', 'recognizes', 'template', 'registered',
                                            'missioncritical', 'jersey', 'restful', 'standardize', 'london', 'datarelated',
                                            'cohesive', 'acceptable', 'almost', 'added', 'ship', 'procurement', 'abstract',
                                            'sponsored', 'rewarded', 'interviewing', 'answering', 'financing', 'accordingly',
                                            'skillset', 'fall', 'poly', 'chat', 'endusers', 'relate', 'becoming', 'shifting',
                                            'earth', 'century', 'playing', 'expedia', 'pool', 'etsy', 'jobrelated', 'crosschannel',
                                            'nurture', 'aligns', 'mdm', 'wish', 'conjunction', 'instrument', 'layer',
                                            'destination', 'mo', 'emotional', 'though', 'reimagining', 'd', 'proposition',
                                            'secondary', 'dr', 'autonomously', 'inspiration', 'integrator', 'coe', 'branch',
                                            'sequencing', 'metro', 'later', 'justice', 'income', 'immediately', 'publicly',
                                            'instead', 'alliance', 'matrixed', 'openness', 'tremendous', 'artist', 'resolving',
                                            'granted', 'reported', 'far', 'identityexpression', 'concrete', 'min', 'pension',
                                            'mid', 'crypto', 'pharma', 'max', 'caci', 'philippine', 'translational', 'unified',
                                            'seeing', 'aware', 'stewardship', 'decisioning', 'postdoctoral', 'funnel',
                                            'multimodal', 'intel', 'toolkits', 'lay', 'variance', 'auditing', 'adopt',
                                            'generates', 'budgeting', 'occupational', 'commute', 'solves', 'flight', 'broadly',
                                            'survival', 'kpi', 'try', 'establishes', 'satisfy', 'memory', 'missiondriven',
                                            'aspiration', 'enduring', 'alone', 'separation', 'acquiring', 'recovery', 'intuition',
                                            'competitor', 'stated', 'trillion', 'baseline', 'churn', 'lifestyle', 'incorporating',
                                            'launching', 'acquired', 'erp', 'body', 'tough', 'wider', 'internalexternal',
                                            'wherever', 'addressed', 'latency', 'gas', 'virtually', 'advising', 'black',
                                            'potentially', 'validity', 'voluntary', 'sign', 'pull', 'tactic', 'comprised',
                                            'adjust', 'founding', 'brief', 'resultsoriented', 'respective', 'envision',
                                            'obtained', 'uphold', 'terminology', 'belong', 'diego', 'taken', 'pertaining',
                                            'governing', 'webbased', 'diagram', 'india', 'yield', 'highgrowth', 'insurer',
                                            'carvana',  'bot', 'consists', 'ford', 'arizona', 'earned', 'summarizing',
                                            'dependency', 'formation', 'debt', 'periodic', 'former', 'labeling', 'inception',
                                            'de', 'semester', 'permitted', 'judgement', 'battery', 'fsa', 'mark', 'prescription',
                                            'farmer', 'relative', 'upgrade', 'accident', 'heavy', 'ada', 'indicate', '45',
                                            'motional', 'uniquely', 'hipaa', 'houston', 'companywide', 'determination',
                                            'annotation', 'formula', '150', 'humble', 'observational', 'editing', 'enduser', 
                                            'substituted', 'previously', 'green', 'pain', 'caregiver', 'seasoned', 'weekend',
                                            'unprecedented', 'owns', 'cohort', 'mixed', 'minute', 'equally', 'away', 'inspection',
                                            'vmwares', 'systematic', 'revolutionize', 'watch', 'iso', 'reliably', 'dhs',
                                            'accessing', 'reimagine', 'likely', '2000', 'sequence', 'friend', 'accelerator',
                                            'confirm', 'housing', 'revolutionizing', 'interfacing', 'glassdoor', 'electronics',
                                            'validated', 'mart', 'carolina', 'da', 'accommodate', 'childrens', 'gotomarket',
                                            'eastern', 'organisation', 'obligation', 'extra', 'scorecard', 'gps',
                                            'trained', 'presales', 'appreciation', 'nursing', 'furthermore', 'restriction',
                                            'staying', 'obstacle', 'button', 'responsive', 'heritage', 'heard', 'revolution',
                                            'modification', 'impossible', 'influential', 'powering', 'seriously', 'percent',
                                            'household', 'dissemination', 'dl', 'cm', 'packaging', 'strict', 'asking',
                                            'enforcement', 'extreme', 'intern', 'encompassing', 'reproducibility', '19', 'loop',
                                            'triage', 'aligning', 'temporarily', 'youd', 'regulated', 'brain', 'hq', 'tune',
                                            'distance', 'continual', 'representing', 'solely', 'beneficial', 'biopharmaceutical',
                                            'instrumentation', 'ey', 'critically', 'fan', 'interacts', 'synthesizing',
                                            'sent', 'nyc', 'qualifying', 'hsa', 'centric', 'culturally', 'mexico',
                                            'zero', 'routinely', 'reveal', 'row', 'recommending', 'tasked', 'synthesis', 'trait',
                                            'possessing', 'perceived', 'flink', 'seen', 'merchandising', 'properly', '360',
                                            'count', 'deepen', 'sharp', 'doesnt', 'ex', 'cellular', 'underrepresented',
                                            'discretionary', 'standardized', 'deriving', 'spread', 'determines', 'enrichment',
                                            'limitless', 'nationality', 'consume', 'arent', 'variant', 'contacted', 'zoom',
                                            'independence', 'protein', 'ryder', 'centralized', 'nike', 'demographic',
                                            'deemed', 'longer', 'fulfilling', 'proudly', 'positioned', 'whatever', 'attack',
                                            'encouraging', 'gained', 'providence', 'unmatched', 'pocs', 'entrylevel', 'spouse',
                                            'adding', 'redefining', 'ohio', 'sole', 'mn', 'aipowered', 'modernize', 'tiktok',
                                            'confidently', 'asia', 'appreciate', 'fire', 'guidewire', 'police', 'endless',
                                            'conventional', 'toptier', 'heshe', 'adp', 'backbone', 'stable',
                                            'registry', 'koverse', 'noise', 'viable', 'lifechanging', 'resiliency', 'bottom',
                                            'spoken', 'remediation', '57', 'iii', 'curve', 'publisher', 'democratize', 'portion',
                                            'restaurant', 'prohibit', 'velocity', 'yr', 'advises', 'venue',
                                            'adapting', 'ssrs', 'bay', 'wireless', 'visually', 'caring', 'linking',
                                            'television', 'ease', 'robot', 'rooted', 'ia', 'awesome', 'observation', 'chewy', 
                                            'shareholder', 'usajobs', 'visitor', 'equipped', 'listener', 'intend', 'star',
                                            'updated', 'alert', 'nonrelational', 'retrieve', 'semiconductor', 'correction',
                                            'licensed', 'anything', 'alexa', 'org', 'genuine', 'prevents', 'ar', 'quantity',
                                            'minimize', 'invitation', 'mclean', 'depends', 'enhances', '400', 'hardest',
                                            'generalized', 'accomplished', 'ticket', 'downtown', 'utilized',
                                            'depend', 'ibmibm', '3d', 'poc', 'challenged', 'writes', 'revolutionary', 'majority',
                                            'allowance', 'ally', 'sustainment', 'parking', 'servicing', 'mac', 'thank',
                                            'mantech', 'suggestion', 'compliant', 'instrumental', 'novartis', 'controlled',
                                            'proofofconcept', 'enterprisewide', 'persona', 'humility', 'columbia', '80', 'ssa',
                                            'securely', 'contractual', 'multiyear', 'solved', 'embraced', 'structural', 'middle',
                                            'prominent', 'safely', '22', 'comprehend', 'testdriven', 'telling', 'consecutive',
                                            'verifying', 'docusign', 'beverage', 'resourceful', 'camera', 'length', 'diagnosis',
                                            'productionready', 'roll', 'referred', 'treated', 'responding',
                                            'escalation', 'completely', '5g', 'chronic', 'robustness', 'doctor', 'loved',
                                            'extending', 'gaining', 'oneonone', 'soon', '2014', 'attempt', 'familial', 'navigating',
                                            'commit', 'worth', 'filing', 'modular', 'tn', 'directive', 'defect', 'tested',
                                            'frequency', 'portland', 'experiencing', 'ago', 'east', 'zscaler', 'meal', 'operator',
                                            'mechanical', 'wait', 'recurring', 'endpoint', 'realization', 'awarded', 'unlocking',
                                            'possibly', 'author',  'produced', 'statesnew', 'stop', 'plenty', 'upstart',  
                                            'applicability', 'land', 'ocean', 'overcome', 'breastfeeding', 'vulnerability',
                                            'agreed', 'homeland', 'personality', 'theme', 'knowing', 'summarization', 'redefine',
                                            'issued', 'automatically', 'assure', 'backup', 'ingenuity',  'aircraft', 'surrounding',
                                            'onboard', 'widely', 'aggressive', 'attending', 'maximizing', 'founder', 'immunology',
                                            'walking', 'ec', 'bureau', 'kingdom', 'finally', 'draft', 'trip', 'eliminate',
                                            'admired', 'biometrics', 'flagship', 'qc', 'fundamentally', 'spot', 'advocacy',
                                            'downstream', 'stretch', 'tower', 'scaled', 'waste', 'analog', 'offshore', 'enter',
                                            '70', 'connectivity', 'exploit', 'validates', 'tagging', 'kpmgs', 'superb', 'compare',
                                            'servicenow', 'teacher', 'benchmarking', 'abbvie', 'humanity', 'interos', 'updating',
                                            'crew', 'doer', 'sharpen', 'janssen', 'exposed', 'dei', 'transport', '120', 'formed',
                                            'peace', 'minneapolis', 'delegate', 'reinvent', 'monetization', 'productionize', 'rwe',
                                            'fixed', 'joined', 'round', 'recognizing', 'panel', 'reston', 'consolidate', 'unlawful',
                                            'mountain', 'augment', 'indirect', 'notforprofit', 'night', '21', 'cna', 'incremental',
                                            'indiana', '160', 'australia', 'siri', 'orientationgender', 'payroll', 'investigating',
                                            'medication', 'constituent', 'gm', 'conception', 'formulating', 'cool', '34', 'dataiku',
                                            'examination', 'undertake', 'urban', 'toolset', 'ut', 'microstrategy', 'evidencebased',
                                            'aspire', 'remaining', 'reside', 'o', 'specialize', 'vesting', 'ehr', 'suitability',
                                            'skillsets', 'thrilled', 'cardiovascular', 'floor', 'promising', 'operationalization',
                                            'overtime', 'highvolume', 'carbon', 'renewable', 'bigbearai', 'encounter', '20000',
                                            'unexpected', 'old', 'disseminate', 'santa', 'pressing', 'facet', 'datacentric',
                                            'transit', 'talend', 'productionlevel', 'distancing', 'planned', 'grows', 'attracting',
                                            'stereotype', 'telemetry', '800', 'biologist', 'nuclear', 'intensive', 'welltested',
                                            'veteransindividuals', 'inventing', 'technologydriven', 'articulating', 'pega', 'cto',
                                            'refinement', '49', 'multicultural', 'responsibly', 'la', 'capturing', 'joy',
                                            'diagnose',  'fastchanging', 'wi', 'weather', 'percentage', 'archive', 'hospitality',
                                            'substitute', 'childcare', 'violence', 'tampa', 'constructing', 'coffee',
                                            'grounded', 'wholesale', 'disrupting', 'dialog', 'subcategory', 'thoroughly', 'dev', 
                                            'wall', 'io', 'fort', '300', 'assay', 'facetoface', '2008', 'uf0b7', 'processed',
                                            'appeal', 'observability', 'pinterest', 'penfed', 'physically', 'kitchen', 'academy',
                                            'gateway', 'propel', '170', 'industryspecific', 'takeda', 'comcast', 'sheet',
                                            'compromise', 'describing', 'authoring', 'lucid', 'desktop', 'shield', 'knowhow',
                                            'naval', 'substance', 'sleeve', 'acceleration', 'specializes', 'georgia', 'pivotal',
                                            'impala', 'ccpa', 'usd', 'safeguard',  'salt', 'elegant', '2017', 'grammarly',
                                            'educating', 'installation', 'ct', 'strengthening', 'structuring', 'sw',
                                            'everyones', 'adaptability', 'permissible', 'moderna', 'duke', 'nontraditional',
                                            'straightforward', 'functionally', 'visionary', 'expensive', 'learned', 'iconic',
                                            'operationalizing', 'protective', 'introduce', 'expanded', '15000', 'faced',
                                            'professor', 'ge', 'contracting', 'conceptualize', 'comprehension', 't', 'concurrently',
                                            'planner', 'columbus', 'marketleading', 'october', 'defend', 'youtube', 'molecule',
                                            'margin', 'kansa', '160000', 'stanley', 'mother', 'bug', 'medicaid', 'flat',
                                            'agriculture', 'factbased', 'backing', 'singular', 'bright', 'logging',
                                            'initially', '2016', 'quarter', 'exists', 'served', 'reserved', 'eagerness', 'rotation',
                                            'demanding', 'comparison', 'pick', 'oregon', 'hotel', 'reddit', 'navigation',
                                            'locally', 'ipsoft', 'illinois', 'exclusive', 'marine', 'alternate', 'embark','dog',
                                            'extension', 'highvalue', 'tip', 'measured', 'projection', 'miami', 'mask', 'usaa',
                                            'bloomberg', 'border', 'negotiate', 'striving', 'describes', 'stuff', 'coordinator',
                                            'dbt', 'wont', 'cgi', 'trying', 'empowerment', 'transformed', 'cincinnati', 'harvard',
                                            'signature', 'workspace', 'soft', 'harnessing', 'coop', 'decide', 'specify', 'twilio',
                                            'connecticut', 'twelve', 'fax', 'edit', 'hisher', 'allowed', 'residency', '40000',
                                            'stocked', 'stake', 'pacific', 'oncall', 'knack', 'composed', 'distilling',
                                            'freight', 'weakness', 'inoffice', 'guard', 'atlassian', 'assume', 'premise',
                                            'minnesota', 'bill', 'eo', 'bus', 'toolsets', 'purchasing', 'synapse', 'construed',
                                            'fueled', 'notification', 'uniqueness', 'endeavor', 'calendar', 'flsa', 
                                            'prime', 'animal', 'visiting', 'invested', 'basketball', 'stanford', 'grocery',
                                            'remarkable', 'usdc', 'import', 'schneider', 'armed', 'commercialization', 'snap',
                                            'transitioning', 'lens', 'configure', 'foot', 'mit', 'completes', 'pretax', 'waiting',
                                            'serviceoriented', 'although', 'philip', 'resultsdriven', 'longitudinal', 'oct', 'sc',
                                            'upcoming', 'leap', 'exceptionally', 'pillar', 'inventive', 'visited', 'council',
                                            'employerdisabilityveterans', 'consortium', 'disrupt', 'cpg', '70000', 'september',
                                            'convention', 'comscore', 'directorate', 'n', 'optum', 'cox', 'progression', 'tag',
                                            'corresponding', 'telecommute', 'radio', 'receipt', 'occasion', 'singapore', 'fish',
                                            'exploitation', 'uc', 'diversified', 'character', 'roku', 'berkeley', 'commercially',
                                            'competent', 'accrediting', 'incomplete', 'regulator', 'adopting', 'fearlessly',
                                            'catalyst', 'tailor', 'violation', 'grumman', 'registration', 'airline', 'franchise',
                                            'utah', 'drafting', 'northrop', 'augmented', 'paying', 'celebrating', 'possession',
                                            'deposit', 'unmet', 'prudential', 'messy', 'inclusivity', 'compassionate', 'forge',
                                            'dataintensive', 'breach', 'rock', 'band', 'sits', 'emphasizing', 'serverless', '46',
                                            'coast', 'retaliation', 'versed', 'ta', 'pathway', 'investigative', 'licensing',
                                            'transforms', 'route', 'represented', 'merchandise', 'cd', 'knowledgeskills',
                                            'consent', 'seniorlevel', 'drink', 'nov', 'treasury', 'companypaid',
                                            'costeffective', 'excitement', 'packaged', 'issuing', 'thanks', 'compass', 'assemble',
                                            'sedentary', 'lineage', 'geico', 'engineered', 'refer', 'bert', 'unitibm', 'postdoc',
                                            'arrive', 'mary', 'reuse', 'cleveland', 'ups', 'modality', 'orlando', 'ascension',
                                            'toole', 'conditional', 'automatic', 'versioning', 'fantastic',
                                            'nothing', 'invited', 'dialogue', 'wearing', 'northern', 'satisfactorily', 'river',
                                            'shown', 'trademark', 'flourish', 'station', 'toyota', 'embedding', 'incorporated',
                                            'testable', 'drift', 'remains', 'nevada', 'compassion', 'carrying', 'concentration',
                                            'accepting', 'organizes', 'pathology', 'officebased', 'pertinent', 'boot', 'ride', 
                                            'proposing', 'lmi', 'highthroughput', 'magic', 'codebase', 'followed', 'learns',
                                            'emission', 'david', 'appetite', 'fight', 'hygiene',  'acute',
                                            'gamechanging', 'preferable', 'charter', 'strategize', 'lb', 'delightful', 'ramp',
                                            'surrounded', 'illness', 'characterize', 'characterization', 'affecting', 'racial',
                                            'mistake', 'noblis', 'checking', 'eeoaa', 'continuum', 'pure', 'jmp', 'left', 'cut', 
                                            'arkansas', 'clarify', 'balanced', 'situational', 'followup', 'transcribe', 'bear',
                                            'bargaining', 'escalate', 'raytheon', 'residential', 'roadblock', 'movie', 'biopharma',
                                            'discrepancy', 'k', 'fullservice', 'performer', 'finra', 'universal', 'micron',
                                            'victim', 'manufacture', 'advantagesdrawbacks', 'omics', 'avoid', 'captured',
                                            'coloradobased', 'sociology', 'healthrelated', 'discord', 'technologyenabled', 'argo',
                                            'passing', 'prove', 'relentlessly', 'alike', 'clicking', 'amongst', 'calibration',
                                            'oldest', '810', 'jacob', 'attractive', 'suit', 'fluid', 'ample', 'refining',
                                            'seminar', 'unusual', 'trending', 'interpreted', 'adheres', 'inherent', 'gene',
                                            'carmax', 'productionizing', 'farm', 'continuity', 'podcasts', 'talking', 'lever',
                                            'clarification', '6000', 'ave', 'dataflow', 'compete', 'habit', 'citigroup', 'endorse',
                                            'continent', 'responds', 'simplicity', 'hp', 'clm', 'honesty', 'angular',
                                            'viewpoint', 'blueprint', 'incredibly', 'maybe', 'arising', 'contextual', 'normally',
                                            'richest',  'usbased', '31', 'vanguard', 'actually', 'peraton', 'adult', 'festival',
                                            'oil', '3m', 'jd', 'enormous', 'hazard', 'fisher', 'guest', 'characteristicto',
                                            'avenue', 'digestible', 'peopleno', 'quora', 'whenever',
                                            'multifaceted', 'brighter', 'valuesdriven', 'static', 'trailblazer', 'rental',
                                            'decided', 'cultivating', 'multichannel', 'female', 'arena', 'suited', 'wear',
                                            'causeandeffect', 'telecommuting', 'consolidation', 'traded', 'embody', 'luxury',
                                            'ondemand', 'biden', 'detroit', 'acoustic', 'datapowered', 'estimated', 'pi',
                                            'separate', 'multiplatform', 'selfserve', 'chartered', 'simplest', 'technician',
                                            'grubhub', 'liberty', 'biomarker', 'counterpart', 'dozen', 'nationally', 'wearable',
                                            'june', 'raising', 'montana', 'pl', 'solutionoriented', '10000', 'subsidized',
                                            'historically', 'missing', 'said', 'info', 'adopted', 'showing', 'micro', 'comment',
                                            'scholar', 'uva', 'seat', 'transact', 'sleep', 'debate', 'crossteam', 'beauty',
                                            'succinctly', 'multitude', 'credible', 'dramatically', 'clevel', 'synergy', 'lie',
                                            'cio', 'anticipates', 'reflection', 'adtech', 'mm', 'entitled', 'preclinical',
                                            'radically', 'benefitsperks', 'ordinary', 'considerable', 'mcdonalds',
                                            'crop', 'genome', 'textual', '2012', 'bike', 'bayer', 'encompasses', 'administering',
                                            '180', 'midlevel', 'distinguished', 'multinational', 'storing', 'enjoyable',
                                            'allstate', 'scheme', 'volvo',  'brave', 'humor', 'enriching',
                                            'prototyped', 'greenhouse', 'complement', 'merck', 'remotefirst', 'youth', 'novavax',
                                            'nih', 'privately', 'receives', 'reconciliation', 'press', 'pepsico', 'experian',
                                            'promotional', 'minded', 'inconsistency', 'buyin', 'historic', 'solutioning', 'et',
                                            'readily', 'supervising', 'commuterelocate', 'agricultural', 'weight', 'forensic',
                                            'pollen', 'penske', 'corrective', 'ontime', 'projected', 'scene', 
                                            'selective', '600', 'fundraising', 'lob', 'customize', 'overnight',
                                            'puzzle', 'prep', 'lender', 'onpremise', 'remained', 'ineligible', 'nasa', '100000',
                                            'respecting', 'usercentric', 'developmental', 'lifting', 'g', 'intake', 'baselined',
                                            'crisis', 'alerting', 'employerprotected', 'subscriber', 'photo', 'correctly',
                                            'securing', 'tradition', 'answered', 'respectfully', 'timing', 'loosely', 'changed',
                                            'liability', 'plano', 'tolerance', 'summit', 'commodity', 'entirely', 'reviewer',
                                            'welcomed', 'datainformed', 'convenience', 'mail', 'discounted', 'kla',  'fine',
                                            'heuristic', 'gift', 'systemic', 'latin', 'topnotch', 'explores', 'lifecycles',
                                            'complying', 'paypals', 'casebycase', 'onchain', 'realizing', 'freely', 'disciplined',
                                            'grooming', 'nine', 'ich', 'probationary', 'financially', 'grand', 'gsks', 'counsel',
                                            'favorite', 'streamlining', 'highend', 'wholly', 'cultivates', 'keyboard', '150000',
                                            'assortment', 'kenilworth', 'purposedriven', 'slide', 'motivates', 'imagination',
                                            'qualtrics', 'pulling', 'sandia', 'forever', 'uber', 'smaller', 'identityassignment',
                                            'argus', 'al', 'maintainability', 'attracts', 'progressively', 'intermittent',
                                            'celebrated', 'rated', 'fellowship', 'admission', 'americaunited', '2007', 'ppg', 
                                            'legislation', 'tennessee', 'bid', 'theft', 'gilbert', 'late', 'j', 'novetta',
                                            'ictap', 'profound', 'observed', 'lincoln', 'brought', 'inferentia', 'seed', 'vet',
                                            'maximus', 'allinclusive', 'voicebots', 'distinct', 'outline', 'baltimore', 'alpha',
                                            'who', 'derivative', '23a', 'adapts', 'formatting', 'unrivaled', 'incoming', 'nuance',
                                            'happening', 'playbook', 'mechanicsburg', 'hit', 'indianapolis', 'conceptdrift',
                                            'routing', 'profession', 'fte', 'onprem', 'frontier', 'charged', 'cure', 'servant',
                                            'instore', 'reasonably', 'uncovering', 'clickstream', 'diarization', 'wage', 'default',
                                            'spec', 'prosperity', 'relies', 'accelerated', 'redhorse', 'hierarchy', 'deploys',
                                            'takeaway', 'jurisdiction', '49014920', 'distribute', 'careerscapitalonecom', 'false',
                                            'telematics', 'workfromhome', 'heterogeneous',
                                            'attach', 'copssc', 'finger', 'aml', 'bae', 'china', 'formerly', 'informative',
                                            'calculated', '710', 'occur', 'td', 'pcg', 'harbor', 'educator', 'openended',
                                            'solicit', 'neighborhood', 'bit', 'fertility', 'fail', 'ideate', 'satisfactory',
                                            'tobacco', 'pioneered', '3000', '2013', 'leidos', 'economist', 'nashville', 'summer',
                                            'rwd', 'uipath', 'prompt', 'userfriendly', 'informing', 'thermo', 'eap', 'bespoke',
                                            'nonfunctional', '18003049102', 'fraudulent', 'residence', 'persuasively', 'invests',
                                            'correspondence', 'counter', 'underserved', 'spanish', 'boeing', 'antonio', '2015',
                                            'nonmerit', 'uploaded', 'disorder', 'permission', 'earlystage', 'collibra', 'openly',
                                            'gdit', 'configuring', 'expansive', 'offchain', 'osha', 'dignity', 'poor', 'choosing',
                                            'recreational', 'vulnerable', 'threshold', 'conflicting', 'unpaid', 'cardinal',
                                            'compiling', 'salaried', 'squad', 'distinctive', 'edw', 'geared', 'bottleneck',
                                            'urgent', 'entail', 'mlbased', 'argonne', 'asks', 'riverside', 'extends', 'broker',
                                            'easytounderstand', 'letting', 'consults', 'moderately', 'epa', 'lgbtq', 'repetitive',
                                            'contracted', 'multi', 'elite', 'latitude', 'wellpositioned', 'heor', 'upper',
                                            'weapon', '60135c', 'winner', 'ocr', 'nonclinical', 'august', 'usually',
                                            'personalizing', 'berlin', 'williams', 'valuebased', 'z', 'provisioning',
                                            'siemens', 'olap', 'recurly', 'encryption', 'cx', 'internetnative', 'satisfied',
                                            'highprofile', 'blood', 'intends', 'adverse', 'worksm', 'placed', 'datarich',
                                            'instructor', 'renewal', 'responsiveness', 'realistic', 'toronto', 'differently',
                                            'metropolitan', 'neuraflash', 'patientfocused', 'constitute', 'raleigh', 'rollout',
                                            'mastercard', 'dangerous', 'posture', 'sidebyside', 'pr', 'bidding',
                                            'postgraduate', 'paramount', 'subordinate', 'hesitate', 'directory', 'instance',
                                            'disrupted', '36', 'viacomcbs', 'positioning', 'adjusted', 'ltd', 'ee', 'sponsoring',
                                            'negative', 'overhead', 'linkage', 'productionquality', 'merger', 'jobqualifications',
                                            'icml', '125', 'customization', 'l', 'welfare', 'stimulating', 'volunteering',
                                            'actor', 'surgical', 'hi', 'digitally', 'converting', 'lt', 'enforce', 'vitae',
                                            'executed', 'gurobi', 'divisional', 'dataops', 'folk', 'contained', 'aesthetic',
                                            'innate', 'compiler', 'persuasive', 'immersive', 'predictable', 'closure', 'unofficial',
                                            'credera', 'corner', 'subsequent', 'organic', 'outcomefocused', 
                                            'confirmed', 'epsilon',  'goto', 'regimen', 'paired', 'cpu', 'onpremises',
                                            'f', 'appealing', 'slowing', 'truist', 'comfortably', 'rootcause', 'journalism',
                                            'matched',  'latent', 'nurse', 'interoperability', 'administer', 'embeddings',
                                            'terminal', 'reflected', 'nicetohave', 'interagency', 'tap', 'orleans', 'keywords',
                                            'eliminating', 'productization', 'mulesoft', 'mold', 'prolonged', 'necessarily',
                                            'alaska', 'trusting', 'liveramp', 'streamlined', 'birthday', '26', 'evening', '175',
                                            'charity', 'combat', 'fairly', 'blvd', 'scholarship', 'pmrs', 'inquire', 'mention',
                                            'pagaya', 'chantilly', 'mckesson', 'ul', 'cadence', 'wonder', 'expressing', 'disaster',
                                            'variation', 'definexml', 'hanover', 'levi', 'soc', 'subsystem', 'featured',
                                            'circuit', 'biologics', 'drastically', 'naturally', 'difficulty', 'welldocumented',
                                            'germany', 'promoter', 'duplicating', 'ccb', '58', 'usecases',
                                            'controller', 'retains', 'rail', 'wayfairs', 'indeeds', 'firmwide', 'jump',
                                            'decentralized', 'showcase', 'runtime', 'prestigious', 'enroll', 'preservation',
                                            'rationalization', 'antiracist', 'chair', '208000', 'death', 'dish', 'hair', 'spotifys',
                                            'gathered', 'consequence', 'coinbase', 'acl', 'prescient', 'guru', 'defi', 'expands',
                                            'editorial', 'checkout', 'govern', 'disciplinary', 'wisconsin', 'zt', 'houzz',
                                            'firsthand', 'collins', 'escalating', 'dialogflow', 'fivestar', 'surgery', 'dow',
                                            'fox', 'contacting', 'cor', 'ng', 'aaa', 'western', 'rolling', '112000', 'aclu',
                                            'outsourcing', 'deserve', 'mellon', 'spa', 'writer', 'outing', 'mostly', 'terminated',
                                            'assimilation', 'individualized', 'compiles', 'retrospective',
                                            'demeanor', 'height', 'compound', 'costbenefit', 'crash', 'boundless', 'concert',
                                            'android', 'charging', 'crossplatform', 'caller', 'pnc', 'inviting', 'announced', 'x',
                                            '27', 'illegal', 'notify', 'contemporary', 'thoughtfully', 'plaid', 'older', 'modifying',
                                            'emphasizes', 'embed', 'hazardous', 'adventure', 'philanthropic', 'oscar', 'expose',
                                            'cpa', 'transferring', 'hungry', 'i9', 'dublin', 'worldrenowned', 'deficiency',
                                            'obsession', 'laundering', 'processor', '56', 'multicloud', 'sexgender', 'altamira',
                                            'dreamer', 'soul', 'salesloft', 'corvus', 'broadbased', 'exceeding', 'southern',
                                            'revise', 'po', 'repair', 'federally', 'fis', 'mindful', 'thus', 'highlighting',
                                            'emotion', 'sm', 'numerator', 'actionoriented', 'robinhood', 'interim', 'coherent',
                                            'encrypted', 'operated', 'recharge', 'samsung', 'diplomacy', 'timeframes', 'outset',
                                            'separated', 'reduced', 'sift', 'purposeled', 'feeling', '130', 'oop', 'hawaii',
                                            'duolingo', 'illustrate', 'credited', 'conservation', 'resourcing', 'functioning',
                                            'insatiable', 'opt', 'fpa', 'eu', 'differing', 'protects', 'shopify', 'displaying',
                                            'evangelist', 'warnermedia', 'cleaner', 'universe', 'classroom', 'telecom', 'varies',
                                            'chip', 'transmission', 'coo', 'simplified', 'lesson', 'librarian', 'authentication',
                                            'sustained', 'procedural', 'uf0a7', 'whitepapers', 'fouryear', 'explanatory', 'sf50',
                                            'radiology', 'dnn', 'agree', 'instructional', 'scrappiness', 'hot', 'madison',
                                            'anticipating', 'brainstorm', 'hw', 'nsa', 'mutually', 'incorporates', 'mandated',
                                            'permit', 'humancomputer', 'dexterity', 'eeoc', 'sustaining', '02', '10m', 'beach',
                                            'exercising', 'productionalize', 'nutrition', 'complementary', 'continental',
                                            'afterpay', 'consumable', 'newest', 'ann', 'informs', 'interpretable', 'w2',
                                            'dayton', 'graduation',
                                            'yahoo', 'aienabled', 'chewys', 'lookout', 'monitored', '250', 'handicap', '1988',
                                            'fascinated', 'producer', 'proteomics', '37', 'migrating', 'enrolled', 'amplify',
                                            'directtoconsumer', 'h', 'confidentially', 'sizing', 'undertaken', 'cigna', 'closed', 
                                            'comparative', 'full_time', 'orange', 'minor', 'dq', 'hhs', 'anthem',
                                            '6200', 'virus', 'auction', 'ucsf', 'redesign', 'verified', 'download', 'happens',
                                            'athlete', '30000', 'exclusively', 'sunshine', 'lack', 'undergoing', 'mediamonks',
                                            'webinars', 'interval', 'liquidity', 'hackathons', 'trigger',
                                            'reached', 'employerpaid', 'election', 'compilation', 'nurturing', 'cissp', 
                                            'argument', 'ppo', 'thereby', 'purposeful', 'homeowner', '_', 'fastforward',
                                            'reverse', 'policygenius', 'belt', 'multidomain', 'mahout', 'sydney', 'selfcare',
                                            'vietnam', 'unfamiliar', 'onto', 'manpower', 'wellversed', 'exam', 'patented',
                                            'forgiveness', 'sep', 'disposal', 'donation', 'vevraa', 'noisy', 'overcoming',
                                            'healthfirst', 'browser', 'advocating', 'genuinely', 'strictly', 'cdo', 'soap',
                                            'july', 'cable', 'pair', 'shipped', 'excites', 'p', 'flatiron',
                                            'dual', 'adjacent', 'locate', 'smooth', 'infer', 'revision', 'multiservice',
                                            'evolent', 'nuanced', 'iqvia', 'treating', 'motorola', 'exponential', 'periodically',
                                            'brick',  'offsite', 'commencement', 'doresponsibilities', 'countless', 'hourly',
                                            'neurips', 'bm', 'telco', 'crystal', 'keller', 'hortonworks',
                                            'junction', 'covidrelated', 'glass', 'amplified', 'cosmetic', 'ara',
                                            'discussing', 'modified', 'gold', 'genesys', 'axon', 'distributor', 'paternity',
                                            'helm', 'removing', 'comprises', 'timeoff', 'missile', 'rx', 'agentbased',
                                            'bad', 'hawkeye', 'postcovid19', 'careful', '4th', 'tour',
                                            'forwardlooking', 'indication', 'sourced', 'tumor', 'yearly', 'socioeconomic',
                                            'tissue', 'exponentially',  'ac', 'ea', 'honor', 'headphone', 'dealer',
                                            'icon', 'giant', 'wfh', 'outcomesis', 'walmarts', 'welldefined',  'mainstream',
                                            'forensics', 'fault', 'ix', 'firmware', 'replication', 'envelope', 'decisive',
                                            'con', 'contingency', 'unknown', 'venmo', 'klas', 'switch', 'alias', 'bytedance',
                                            'abide', '365', 'democratizing', 'injury', 'rationale', 'compiled', 'taxonomy',
                                            'sgo', 'designation', 'precedence', 'couple', 'conveying', 'young', 'ce',
                                            'brainstorming', 'unconventional', 'compensates', 'relativity', 'thinktank',
                                            'capitalize', 'attainment', 'darpa', 'industrystandard', 'ego', 'badge',
                                            'bitfury', 'honoring', 'laptop', 'wine', 'paris', 'northwest', 'alienage', 'eliciting',
                                            'sustainably', 'ticketing', 'commutable', 'weighting', 'influenza', 'usaid',
                                            'valueadded', 'counting', 'nonroutine', 'modernizing', 'healthsafety', 'pertains',
                                            'hone', 'unicorn', 'plain', 'carnegie', 'microscopy', 'microsofts', 'phenomenal',
                                            'investigates', 'biomarkers', 'fiscal', 'interpretability', '4year', 'nyu', 'nist',
                                            'printer', 'frictionless', 'tokyo', 'csuite', 'telecommuter', 'queue', 'impose',
                                            'dean', 'alumnus', '55', 'browse', 'sought', 'seniority', 'nbcuniversal', 'copyright',
                                            'calculate', 'blackstone', 'discovered', 'temperature',  'beam', 'cpt', 'versus',
                                            'sold', 'aa', 'hugging', 'fullest', 'reengineering', 'qualcomm', 'ida', 'non',
                                            'elevating', 'hill', 'est', 'collegial', 'closeknit', 'dohmh', 'brown', 'mt',
                                            'morning', 'affected', 'newlyhired', 'blockchains', 'commuting', 'coursera',
                                            'standardizing', 'floating', 'recurrent', 'scan', 'refactor', 'att', 'incubation',
                                            'broadcast', 'easytouse', 'tsa', 'doc', 'renowned', 'usc', 'lived', 'indicated',
                                            'xenai', 'split', 'methodical', 'machinery', 'double', 'momentum', 'cruise', 'prism',
                                            'rcm', 'firmly', 'print', 'danaher', 'voya', 'robotaxi', 'honeywell', 'nearterm',
                                            'deviation', 'maturing', 'nestle', 'studying', 'promptly', 'industrialize', 'amp',
                                            'notion', 'socialize', 'translator', 'ggv', 'epidemiological',
                                            'caterpillar', 'situated', 'reservist', 'sophistication', 'enterpriselevel', 'jointly',
                                            'participated', 'thru', 'recover', 'nonexperts', 'wellrounded', 'appearance', 'notch',
                                            'courageous', 'dispersed', 'regeneron', 'chop', 'layout', 'sf', 'twin', 'appliance', 
                                            'businesscritical', 'atlantic', 'released', 'fearless', 'amenity', 'musthave', '2nd',
                                            'cornell', 'bosch', 'dirty', 'suggesting', 'admin', 'imperfect', 'pharmacology',
                                            'juggle', 'ingredient', 'relay', 'se', 'learningbased', 'broaden', 'tradecraft',
                                            'pharmacist', 'bicycle', 'extracted', 'formulates', 'okrs', 'european', 'crossborder',
                                            'flc', 'arbor', 'humancentric', 'roc', 'kit', 'breakdown', 'bia', 'mixture', 'daysweek',
                                            'humancentered', 'uniformed', 'wellstructured', 'cambium', 'medicaldentalvision',
                                            'tapping', 'cisco', 'upstream', '200000', 'oak', 'viewed',
                                            'centre', 'gaithersburg', 'queuing', 'exactly', 'ireland', 'japan', 'harmonization',
                                            'tfls', 'exact', 'suggests', 'activate', 'emergent', 'goaloriented', 'esg', 'equip',
                                            '71', 'thereof', 'litigation', 'xpo', 'oneyear', 'newer', 'solar', 'e2e', 'multiint',
                                            'uiux', 'curating', 'strongest', 'delighting', 'refers', 'tim', 'displayed', 'dw',
                                            'nci', 'boast', 'sd', 'standalone', 'similarly', 'sort', 'resonate', 'cataloging',
                                            'outofthebox', 'interplay', 'reshape', 'dominated', 'march', 'preconfigured', 'herein',
                                            'impression', 'trader', 'ppd', 'dsd', 'systematically', '68', 'preexisting', 'sony',
                                            'traveling', 'impacted', 'hispanic', 'publicis', 'varo', 'accidental', 'court',
                                            'carta', 'hottest', 'gameplay', 'likeminded', 'uw', 'rethink', 'positivity',
                                            'venturebacked', 'hosting', 'peak', 'alto', 'migrate', 'highvisibility', 'resumecv',
                                            'companysponsored', 'allowable', 'um', 'laic', 'grammarlys', 'orise', 'equivalency',
                                            'mercy', 'fmla', 'std', 'businessesagencies', 'apogee', 'clientowned',
                                            'elicit', 'deepmind', 'fsp', 'ambassador', 'requisite', 'foremost', 'tdd',
                                            'missionfocused', 'princeton', 'bristol', 'serverside',
                                            'apco', 'hairstyle',
                                            'underwriter', 'ending', 'conceptualization', 'ignite', 'ph', 'hca', 'steer',
                                            'hampshire', 'verizons', 'mitigating', 'syndicated', 'ball', 'xandr', '65',
                                            'tabular', 'abstraction', 'adjusting', 'alight', 'vancouver', 'grammar', 'valuing',
                                            'thursdayfriday', 'smile', 'conceive', 'baseball', 'deere', 'eventually', 'vernacular',
                                            'macys', 'differentiate', 'install', 'actuary', 'traditionally', 'meetups', '22000',
                                            '4000', 'earning', 'rsm', 'emory', 'iclr', 'transitioned', 'atmospheric', 'tlfs',
                                            'bd', 'seta', 'shouldnt', 'recording', 'evangelizing', 'indoor', 'mfdv', 
                                            '25000', 'mvc', 'precaution', 'preventing', 'tenuretrack', 'prosper', 'rise', 'mainly',
                                            'soccer', 'immune', 'tone', 'israel', 'notetaking', 'peoplefirst', 'hartford',
                                            'stateofthe', 'emerson', 'biased', 'meta', 'partial', 'crack', 'disney', 'undefined',
                                            'consuming', 'q', 'photography',  'digging', 'silo', 'cloudformation', 'town',
                                            'formally', 'congestion', 'iac', 'redesigning', 'insurtech', 'node', 'programmatically',
                                            'lilly', 'gc', 'eisai', 'allocate', 'iv', 'turnaround', 'unite', 'landing', 'unlike',
                                            'tied', 'fly', 'fiscalnote', 'thereto', 'tesla', 'cheriton', 'celebration', 'bench',
                                            'fastgrowth', 'tutorial', 'differentiating', 'trail', 'knowledgebase', 'integer',
                                            'nonstandard', 'penetration', 'monetize', 'simplifying', 'employeeowned', 'bond',
                                            'ruthlessly', 'celestar', 'embodies', 'proc', 'cold', 'iri', 'banner', 'c3', 'sea',
                                            'clientessential', 'salesdelivery', 'encompass', 'neurodiversity', 'palo', 'dli',
                                            'honored', 'observe', 'epidemiologic', 'freddie', 'skyrocketed',
                                            'viewing', 'minitab', 'antibody', 'historian', '14000', 'reproductive', 'refined',
                                            'wellknown', '2030', 'intense', 'attraction', 'numenta', 'populate', 'ryders',
                                            'endura', 'volunteerism', 'chemours', 'convergence', 'nm', 'forming',
                                            'yale', 'influencers', 'crowd', 'dutiesresponsibilities', 'bowlthe', '10x', 'millennials',
                                            'pediatric', 'commonwealth', 'aggressively', 'enterprisescale', 'customermatrix',
                                            'cognitivescale', 'candid', 'ed', 'marginalized', 'dependable', 'unemployment', 'pmo',
                                            'pretty', 'homebased', '50000', 'hosted', 'experienceknowledge', 'productize',
                                            'radical', 'oracleteradata', '1159', 'renaissance', 'abbott', 'lp', 'advertised',
                                            'containing', 'voted', 'powerapps', 'highfrequency', 'finetuning', 'preliminary',
                                            '26500', 'saiccom', 'pitch', 'vault', 'marsh', 'substantive', 'fascinating', 'trace',
                                            'wallet', 'springfield', 'domestically', 'impressive', 'cumulative', 'encore',
                                            'inefficiency', 'entering', 'behaviour', 'withdraw', 'wellfunded', 'infection',
                                            'devoted', 'userfacing', 'knocking', 'ok', 'anytime', 'mar', 'excluded', 'delve',
                                            'physiology', 'exploiting', '85', 'fusing', 'enjoying', 'eagerly', 'customerdriven',
                                            'cortana', 'sec', 'resonates', 'assembling', 'deepdive', 'hunger', 'uma', 'scholarly',
                                            'mentioned', 'constellation', 'blending', 'trainer', 'accomplishing', 'bolster', 'ft',
                                            'playstation', 'pursues', 'knowledgeexperience', 'bending', 'pre', 'escalates',
                                            'copied', 'firstclass', 'lactation', 'burden', 'alamo', 'wwwgettyimagescom', 'needing',
                                            'incurs', 'financials', 'donor', 'communicative', 'apl', 'conform', 'viability',
                                            'xoom', 'braintree', 'amelia', 'pypl', 'origination', 'ear', 'deductible', 'hubspot',
                                            'friction', 'physiological', 'netflix', 'mf', 'born', 'biquarterly', 'diabetes',
                                            'richmond', 'warranty', 'loving', 'needle', 'electricity', 'attest', 'legallyrequired',
                                            'seismic', 'believing', 'aging', 'runtimes', 'udemy', 'crederas', 'payor', 'li',
                                            'undergo', 'societal', 'lawful', 'appraisal', 'boehringer', 'selects', 'literal',
                                            'turnover', 'legible', 'traceability', 'healthiest', 'adequately', 'absolutely',
                                            'finished', 'sight', 'ea_exphire', 'programme', 'outdoor', 'decker', 'everest',
                                            'irving', 'candidly', 'influencer', 'specie', 'dba', 'lost', 'bullet', 'educates',
                                            'interpreter', '10000000', 'register', 'firstofitskind', 'pepsicola', 'phenomenon',
                                            'bend', 'metabolic', '48', 'statutory', 'bound', 'compensate', 'negotiating',
                                            'businessrelated', 'unrelated', 'resourcefulness', 'headspace', 'chef', 'quant',
                                            'governmental', 'northeastern', 'ri', 'repayment', 'tusimple', 'miner', 'attorney',
                                            'arc', 'bridging', 'myriad', 'solicitation', 'adviser', 'squibb', 'retrieving',
                                            'seeyourselfcignacom', 'colour', 'annotated', 'alcohol', 'bethesda', 'powerhouse',
                                            'crawler', 'slas', 'ref', '24b', 'advantageous', 'softwaretools', 'dsi', 'island',
                                            'moveworks', 'bottomline', 'permanently', 'accessed', 'edc', 'limc1', 
                                            'saint', 'grad', 'pubsub', 'elder', 'onpoint', 'entered', 'cleared', 'upsell',
                                            'nudge', '47', 'transcriptomics', 'preview', 'gadget', 'overarching', 'od',
                                            'frontline', 'pragmatism', 'elasticity', 'markit', 'stellar', 'jefferson',
                                            'calculating', 'safeguarding', 'ridesharing', 'harris', 'beloved', 'ncqa', 'swap',
                                            'ihs', 'bentonville', 'jax', 'nccharlotte', 'outsourced', 'hightech', 
                                            'veteransdisabled', 'revolve', 'approve', 'airport', '21st', 'congress', 'windowlearn',
                                            'faith', 'adequacy', 'lumen', 'negotiable', 'moines', 'rivian', 'steady', 'http',
                                            'lgbtqia', 'waymo', 'exp', 'informational', 'singledose', 'spearhead', 'forwarded',
                                            'authentically', 'lane', 'battelle', 'unmanned', 'triple', '1015', 'shine', '95',
                                            'intels', 'gmp', 'novo', 'championing', 'encouragement', 'henry', 'businessdriven',
                                            'framing', 'optimizely', 'hitech', 'paycheck', 'exl', '18th', 'sub', 'irb',
                                            'organisational', 'displaced', 'offtheshelf', 'surprise', 'employmentbased', 'myers',
                                            'ds', 'ssc', 'apparel', 'acuity', 'politics', 'amd', 'accelerates', 'persuade',
                                            'cart', 'accolade', 'extensible', 'quoras', 'favorable', 'respiratory', 'customizable',
                                            'mateo', 'podcasting', 'sts', 'agony', 'film', 'franklin', 'administered', 'wisdom',
                                            'electronically', 'reshaping', 'accepts', '29', 'corning', 'examining', 'certify',
                                            'researchdriven', 'butterfly', 'tendency', 'ceremony', 'adaptation', 'approachable',
                                            'cvpr',  'childhood', 'bwh', 'groupon',
                                            'banker', 'administers', 'sanction', 'marcus', 'becomes', 'breathe', 'memphis',
                                            'birth', 'performancebased', 'affectional', 'quota', 'bhg', 'driverless',
                                            'affordability', 'volleyball', 'arch', 'str', 'patientcentered', 'macroeconomic',
                                            'rarely', 'multitouch', 'annualized', 'tabulation', 'longstanding', 'challengesand',
                                            'extractionmanipulation', 'programssoftware', 'theater', 'earlier', 'offense',
                                            'expenditure', 'netezza', 'seize', 'extensively', 'ne', 'outlining', 'texture',
                                            'wellestablished', 'faulttolerant', 'cheminformatics', 'hpe', 'ev', 'consultings',
                                            'agronomic', '2010', 'midb', 'invision', 'didnt', 'valueadd', 'cap', 'laserfocused',
                                            'aetna', 'freelancer', 'prospecting', 'ra', 'despite', 'refresh', 'replenishment',
                                            'seasonal', 'codebases',  'ace',
                                            'multiplier', 'b2c', 'reynolds', 'infinite', 'undertaking', 'schwab', '1500',
                                            'omnicell', 'lipost', 'albertsons', '65000','teleworking', 'koch', 'apiai',
                                            'console', 'telehealth', 'columnar', 'metlife', 'mam', 'workrelated',
                                            'preparedness', 'arl', 'com', 'approaching', 'raceethnicity', 'nicetohaves',
                                            'biweekly', 'super', 'kid',
                                            'categorical', 'largely', 'hilton', 'vha', 'skin', 'cb', 'longrange', 'commander', 
                                            'followthrough', 'psychological', 'np', 'c4it', 'amphibious', 'middleware', 'rush',
                                            'erwin', 'httpspearsonbenefitsuscom', 'breakneck', 'digitalfirst', 'mash',
                                            'quizlet', 'reject', 'lowes', 'memorable', 'vested', 'splitting',
                                            'allergan', 'tcpip', 'esteemed', 'steinberg', 'goat', 'protiviti', 'collector',
                                            'devising', 'parse', 'epidemiologist', 'familyfriendly', 'softwareasaservice',
                                            'deserves', 'reconnaissance', 'earliest', 'sculley', 'shoulder', 'shipt',                                            
                                            'jack', 'globalreach', 'deepening', 'pson', 'whilst', 'wwwpearsoncom', 'mgb',
                                            'usageperformance', 'stoop', 'emea', 'serial', 'vcs', 'itil', 'behaviorresults',
                                            'quiet', 'somewhere', 'solidity', 'malicious', 'cecl', 'graduated', 'cst',                                                                                        
                                            'citrine', 'quadrant', '101', 'identifier', '03', 'miningdata', 'infrastructureascode',
                                            'achieves', 'governed', 'resumescvs', 'overlap', 'confusion', 'witai', 'os', 'flock',
                                            'massively', 'singlecell', 'granularity', 'egoless', 'imagined', 'mechanic',
                                            'servicessupport', 'preserve', 'differ', 'labeled', 'ton', 'productionalization',
                                            'stationary', 'siriusxmpandora', 'eur', 'dismemberment', 'census',
                                            'ingelheim', 'xylem', 'wireframes', 'mixing', 'hcp', 'exos', 'ehs', 'executivelevel',
                                            'highmark', 'adjudication', 'azimuth', 'httpsxenai', 'waltham', 'allegiance',
                                            'lifesaving', 'bee', 'signing', 'resultant', 'vigilant', 'dayworking', 'stitch',
                                            'conceptualizing', 'bramble', 'advised', 'di', 'connector', 'mitreand', 'hhsc',
                                            'siriusxm', 'twice', 'tel', 'pandora', 'quantitate', 'productizing', 'comparing',
                                            'workbench', 'foursquare', 'reallife', 'resolved', 'demography', 'soothsayer',
                                            'lease', 'nonus', 'delay', 'laugh', 'testability', 'budgeted', 'selfaware',
                                            'invitae', 'resulted', 'crosssell', 'zillow', 'booking', 'reader', 'postersupplement',
                                            'sysco', 'perficient', 'healthineers', 'soil', 'assures', 'rehabilitation',
                                            'nato', 'equalopportunity', 'favor', 'farfetch', 'silver', 'cdw', 'pwcs',
                                            'consist', 'techdriven', 'precedent', 'africa', 'attends', 'automobile',
                                            'clara', 'vetting', 'returned', 'turned', 'stresstest', 'licensure', 'bu', '19972021',
                                            'reimburse', 'healing', 'recruitinghelpmitreorg', 'phi', 'crowe', 'unites', 'ncu',
                                            'comprising', 'riding', 'casino', 'exemplary', 'facilitator', 'weblog',
                                            'textuallanguage', 'worldleading', 'drill', 'afforded', 'atlas', 'polished', 'cfa',
                                            'owe', 'prerequisite', 'anthropologie', 'dm', 'adversary', 'purdue', 'digitized',
                                            'workstation', 'tester', 'jose', 'effortless', 'distinction', 'claiming', 'cyberspace',
                                            'cedar', 'metal', 'rockville', 'uf', 'barbaricum', 'subsidy', 'missionoriented',
                                            'projectbased', 'incorporation', 'cheaper', 'marketed', 'lgpd', 'technologybased',
                                            'glba', 'unify', 'dla', 'reinsurance', 'vui', 'multibillion', 'referring',
                                            'annuity', 'verifies', 'worksite', 'oci', 'dos', '12000000', 'jj', 'lahsa', 'puppet',
                                            'rfps', 'irvine', 'correlating', 'icd10', 'stressful', 'dataproc', 'asymmetric',
                                            'soundness', 'redundancy', 'roughly', 'wake', 'biogen', 'hl7', 'spam', 'fiber',
                                            'exceeds', 'causality', 'login', 'disadvantaged',
                                            'sacramento', 'feasible', 'uploading', 'sooner', 'roth', 'datacenter',
                                            'egnyte', 'syntax', 'outlet', 'partially', '2005', 'incrementally', 'finalist',
                                            'zuora', 'hivaids', 'universally', 'jr', 'deck', 'scanning', 'intention', '247',
                                            'wolf', 'sciencerelated', '8th', 'oneself', 'tm', 'setback', 'fruit', 'anomalous', 
                                            'simplification', 'edition', 'coalition', 'hypergrowth', 
                                            'guild', 'evernorth', 'moral', 'appfolio', 'picking',  'tempe',
                                            'unity', 'angle', 'el', 'rnaseq', 'lock', 'featuring', 'fueling', 'fortive', 'candor',
                                            'restore', 'rna', 'newr', 'wednesday', 'httpswwwgetsalesforcebenefitscom',
                                            'evidenced', 'oliver', 'port', 'demandshome', 'reputational', 'newreliccom',
                                            'ingested', 'bigbearais', 'importing', 'passport', 'topoftheline', 'cooking',
                                            'playercoach', 'northwestern', 'icf', 'multithreaded', 'jvm', 'golf',
                                            'calibrated', 'touchpoint', 'filling', 'commercialize', '28',
                                            'intentional', 'accrued', 'granular', 'initiation', 'denial', 'subset',
                                            'contributed', 'sam', 'cornerstone', 'adwords', 'mrna', 'pentaho',
                                            'toprated', 'statute', 'disabilitymedical', 'similarity', 'liquid', 'allocated',
                                            'f1', '401', 'atscale', 'consumed', 'warfighter', 'smartest', 'thereafter',
                                            'goaldirected', 'optimum', 'touchpoints', 'blocker', 'busy', 'paperwork',
                                            'highpressure', 'coin', 'remediate', 'anothers', 'streamprocessing', 'upward',
                                            'mastercards', 'nasdaq100', 'steadfast', 'veeva', 'workfusion',
                                            'computerized', 'sized', 'substitution', 'streetlight', 'herndon', 'newark', 'wd',
                                            'committing', 'longevity', 'feeding', 'discriminatory', 'questioning', 'wilmington',
                                            'sporting', 'memo', 'motto', 'taste', 'texttospeech', 'instill', 'charlottesville',
                                            '7000', 'utmost', 'eat', 'minimizing', 'cacis', 'womanowned', 'ojo', 'dispute',
                                            'envisioning', 'predefined', 'creek', 'readable', 'dsw', 'incurring', 'handinhand',
                                            'eks', 'brokerage', 'bedside', 'customizing', 'nontrivial', 'appreciated', 'aig',
                                            'dakota', 'peripheral', 'svp', 'appen', 'tightknit', 'tonal', 'refinancing', 'rowe',
                                            'rocket', '700', 'iam', 'crossbusiness', 'inclusiveness', 'season', 'bulk', '40b',
                                            'kindness', 'grower', 'conveys', 'datainformation', 'hinge', 'augmenting',
                                            'fitforpurpose', 'usual', 'noted', 'naming',  'psychometrics', 'ortec',
                                            'deferred', 'drone', 'huntington', 'adsupported', 'sonar', 'everevolving', 'hta',
                                            'portability', 'preserving', 'antimoney', 'realized', 'tie', 'cdf', 'practicing',
                                            'fulfills', 'quest', 'stoneturn', 'redcap', 'allsource', 'buzzfeed', 'atm', 'relied',
                                            'asneeded', 'refactoring', 'auditor', 'attachment', 'rep', 'cti', 'tailoring', 'par',
                                            'revolutionized', 'maine', 'leasing', 'kneel', 'compatibility', 'installing',
                                            'deadlinedriven', 'strengthens', 'affords', 'galaxy', 'contest', 'arrange',
                                            'neurology', 'hedis', 'landmark', 'branding', 'confluent', 'ice', 'excels',
                                            'readjustment', 'customerfirst', 'recipient', 'pairing', 'simulated', 'amplitude',
                                            'revised', 'legislative', 'sanofi', 'contextualize', 'systemwide', 'opm', 'worry',
                                            'approximate', 'nro', 'infrequent', 'restructure', 'coronavirus', 'prescribed',
                                            'dogfriendly', 'cryptocurrency', 'ner', 'breed', 'seagen', 'staging', 'afb',
                                            'renewables', 'biometric', 'proving', 'inventor', 'businessfocused', 'pooling',
                                            'pdds', 'gdia', 'keyword', 'ctsh', '11000', 'disposition', '185', 'pcr', 'adopts',
                                            'somewhat', 'cofounder', 'projectspecific', 'pulse', 'aptiv',
                                            'laser', 'ou', 'integrative', 'alarm', 'wartsila', 'espp', 'coverent', 'assigning',
                                            'qinetiq', 'aviv', 'ingalls', 'flu', 'rayban', 'biremote', 'flint', 'excite', 'chubb',
                                            'tablet', 'inequality', 'privacyrelated', 'zulily', 'occupation', 'downtime',
                                            'bluehalo', 'proximity', '2009', 'uplift', 'comprehensible', 'prize', 'skus', 
                                            'broken', 'annotate', 'courteous', 'april', 'dentsu', 'meaningfully', 'football',
                                            'lexisnexis', 'masking', 'questionanswering', 'danger', 'farreaching', 'mantechs',
                                            'cerner', 'drop', 'aiops', 'onsiteiq', 'programmed', 'isr', 'boldly', 'mgh',
                                            'composing', 'uat', 'webpage', 'spin', 'textbased', 'appear', 'somerville',
                                            'differentiation', 'baby', '1300', 'sabre', 'rotational', 'entirety', 'homeownership',
                                            'strike', 'resort', 'enjoyed', 'kensho', 'computationally', 'snapshot', 'fifth',
                                            'avanade', 'midsized', 'pending', 'mailing', 'indirectly', 'sparse', 'timesensitive',
                                            'notified', 'stratification', 'ctapictap', 'malware', 'nh', 'wind', 'ridge', 'ot',
                                            'rl', 'craftsmanship',  'autopilot', 'stereotyping', 'defensive', 'ukg',
                                            'decompose', 'kdd', 'ksas', 'earnings', 'password', 'publishes', 'louisiana',
                                            'espark', 'introducing', 'everybody', 'burlington', 'aside', 'flavor', 'outbound',
                                            'upskill', 'relaxed', 'sapient', 'achievable', 'neustar', 'hopkins', 'perceive',
                                            'copier', 'asap', 'transunion', 'slac', 'damage', 'charting', 'teambuilding',
                                            'hancock', 'jobseekers',  'fewer', 'nga', 'cno', 'corelogic',
                                            'amendment', 'suck', 'definitely', 'skillsabilities', 'elk', '1520', 
                                            'reddits', 'determinant', 'incl', 'appreciates', 'sparkstreaming', 'niche', 'proceed',
                                            'futureready', 'aiming', 'simplifies', 'attentive', 'toolbox', 'monica', 'warning',
                                            'france', 'gui', 'er', 'cipp', 'realm', 'canspam', 'telecommuters', 'ffrdc', 'ncr',
                                            '84', '900', 'prometheus', 'cbp', 'socially', 'ambulatory', 'precisely', 'depot', 
                                            'pedestrian', 'fitting', 'newport', 'seemingly', 'usingcreating', 'fulfilled', 
                                            'borrower', 'bestfit', 'scripps', 'federated', 'cisa', 'recreation', 
                                            'establishment', 'gate', 'aka', 'lish1', 'confirming', 'summarizes', 'esop', 'beside',
                                            'motionals', 'rockwell', '01', 'koreai', '1200', 'cardiac', 'ubers', 'fingerprint',
                                            'frog', 'platformenabled', 'reportable', 'dare', 'explorer', 'attestation', 
                                            'orientated', 'generalpurpose', 'impeccable', 'sfl', 'drawback', 'prologis', 'wrong',
                                            'thomas', 'delegating', 'invented', 'pollution',
                                            'unauthorized', 'havent', 'customizations', 'mythic', 'saks', 'countermeasure',
                                            'avantus', '132', 'markdown', 'inbound', 'oe', 'collegiality', '2003', 'decipher',
                                            'commence', 'yoga', 'fragmented', 'aps', 'surgeon', 'redfin', 'transparently',
                                            'workout', 'lend', 'pack', 'breakfast', 'twodose', 'buffalo', 'embarking', '650', 
                                            'inequity', 'intouch', 'introduced', 'upskilling', 'excluding', 'intuit',
                                            'queried', 'lucene', 'vestcom', 'archiving', 'datadog', 'codevelop', 'appointed',
                                            'catalent', 'began', 'magnitude', 'naci', 'staffed', 'trucking', 'catastrophe',
                                            'emphasize', 'otsuka', '2006', 'identifiable', 'hero', 'propulsion', 'lakeland',
                                            'smith', '1100', 'physicist', 'amend', 'excess', 'occurs', 'yorknew',
                                            'snow', 'denied', 'inclination', 'emotionally', 'tutor', 'scam', 'switching',
                                            'enterprisegrade', 'obvious', 'indicating', 'calm', 'emnlp', 'miss', 'simulator',
                                            'exit', 'tapcart', 'reconcile', 'nonexempt', 'msd', '50m', 'gross', 
                                            '5th', 'vibe', 'buried', 'fraction', 'vague', 'anchor', '8000', 'missouri',
                                            'mobilefirst', 'technologically', 'complementing', 'steel', 'bidens', 'breeding',
                                            'solutionsoriented', 'dhei', '2011', 'harrisburg', 'multistakeholder',
                                            'shipment', 'cartcom', 'supplementing', 'opendoor', 'norc', 'recursion', 
                                            'hubbers', 'regulationsstandards', 'saying', 'triplelift', 'adapted',
                                            'disqualify', 'twitch', 'backtested', 'robert', 'or', 'cj', 'crosscountry', 'dia',
                                            'configurable', 'irca', 'boomi', 'multiomics', 'decisiveness', 'microbial', 'flawless',
                                            'figuring', 'cabinet', 'spatiotemporal', 'essence', 'opentext', 'gay', '33',
                                            'microbiology', 'simpler', 'dependability', 'autodesk', 'hm', 'occurred',
                                            'automaker', 'macos',
                                            'charles', '52', 'bandwidth', 'quote', 'struggle', 'starter', 'aug', 'accompanying',
                                            'wyman', 'kroll', 'characterizing', 'entertain', 'unvaccinated', 'principled', 'bet',
                                            'pipeda', 'upholding', 'humana', 'cnbc', 'domino', 'clinically', 'passed', 'preventive',
                                            'apex', 'stateprovince', 'procter', 'eleven', 'projectstasks', 'genius', 'macrolevel',
                                            'diplomaged', 'creatingrunning', 'productionalized', '13000000', 'purposebuilt', 'tb',
                                            'crains', 'aftermarket', '2002',
                                            'elimination', 'itar', '1961', 'technologyfocused', 'munich', '80000', 'releasing',
                                            'meant', 'reform', 'instant', 'rational', 'reflecting', 'grew', 'paul', 'responder',
                                            'hustle', 'statisticallyminded', 'ibmwhat', 'productsservices', 'vr', 'westlake',
                                            'deployable', 'athome', 'tlf', 'unavailable',
                                            'businessolver', 'bio', 'av', 'proposes', 'played', 'postdegree', 'autoscaling',
                                            'ordering', 'asic', 'logic2020', 'conveniently', 'compromising', 'hyderabad',
                                            'iaa', 'snagajob', 'nonstatisticians',
                                            'disco', 'higherlevel', 'rsus', 'menu', 'mle', 'integrationcontinuous', 'tenant',
                                            'tend', 'policymakers', 'wastewater', 'assessed', 'sankyo', 'jean', 'authoritative',
                                            'crossdepartmental', 'garage', 'slow', 'reflective', 'yearround', 'safetysensitive',
                                            'varsity', 'icertis', 'camp', 'objectively', 'impairment', 'injection', 'ob', 
                                            'domicile', 'gapp', 'iq', 'detailing', 'cmc', 'ccar', 'preventative', 'nordisk',
                                            'settlement', 'mx', 'bird', 'advertisement', 'tf', 'eurofins', 'whichever', '263892',
                                            'coppa', 'amyris', 'labcorp', 'justification', 'virology', 'nextroll',
                                            'njoy', 'educationtraining', 'cref', 'experiential', 'oltp', 'menlo', 'reaction',
                                            'organizationally', 'gradeslevels', 'protectionrelated', 'caching', 'mentored',
                                            '75year', 'geo', 'oura', 'forrester', 'arrow', 'accrual',
                                            'multilevel', 'gb', 'secrecy', 'drg', 'eighty', 'differentiator', 'acquires', 'avp',
                                            'ipsos', 'daunting', 'republic', 'carfax', 'outpatient', 'xsell',
                                            '74000', 'lowest', 'cashierless', 'highscale', 'finicity', 'mdg', 'digest',
                                            'stewarding', 'lsi', 'blacksky', '131', 'kelly', 'reset', 'george', 'autoimmune',
                                            'hcpcs', 'reuters', 'coolest', 'salient', 'daiichi', 'achiever',
                                            'lateral', 'hobby', 'shehe', 'wired', 'alexis', 'sarah',
                                            'freeportmcmoran', 'sia', '13000',  'fundbox', 'coupa', '707', 'versatility',
                                            'aitecharch', 'bmo', 'bmw', 'bdm', 'ap', 'encountered', 'iat', 'computerelectrical',
                                            'hrbp', 'aon', 'ministry', 'foundry', 'dea', 'marijuana', 'industrybased', 'dcri',
                                            'digitalization', 'gastroenterology', '1year', 'recorded', 'shelf', 'rosslyn', 
                                            'fieldbased', 'mississippi', 'thrill', 'weighing', 'bigtable', 'reusability',
                                            'sabbatical', 'cat', 'dry', 'brooklyn', '110000', 'insured', 'elicitation', 'ironic',
                                            'metrology', 'appropriateness', 'recipe', 'rei', 'manulife', 'itatms',
                                            'inaccuracy', 'directhire', 'steering', 'branded', 'pd', 'stride', 'acorn',
                                            'valuedriven', 'merrick', 'perkins', 'activision', 'kleiner', 'reconstruction',
                                            'tredence', 'staple', 'gamble', 'abbvies', 'uniform', 'kaiser', 'accessory',
                                            'imagining', 'immunization', 'aca', 'golden', 'switzerland', 'replace',
                                            'ledger', 'responded', 'qualifies', 'medidatas', 'stormsamza', '1b', 'ornl',
                                            'acquirer', 'tpm', 'durham', 'evergrowing', 'businessoriented', 'shrinkage', 'avaya',
                                            'interdepartmental', 'preeminent', 'messenger', 'wing', 'drilling', 'jackson', 'uas', 
                                            'deciding', 'regarded',  'fbi', 'nd', 'ctos', 'replicable', 'trackrecord',
                                            'clery', 'quite', 'inscom', 'passenger', 'sla', 'symbol', 'reinforces', 'systemlevel',
                                            'constructively', 'yesterday', 'rose',  'selfawareness', 'okay', 'liaising',
                                            'nexus', 'vista', 'mouse', 'predictability', 'underperforming', 'leaving',
                                            'trouble', 'mirror', 'controlling', 'sweden', 'intimate', 'alexion', 'circlers',
                                            'farming', 'skillsqualifications', 'edr', 'lighting', 'archival', 'designthinking',
                                            'celonis', 'everexpanding', 'ftes', 'indigenous', 'analyzer', 'mentalphysical',
                                            'dubai', 'fruition', 'manually', 'mandating', 'hazmat', 'ln', 'handled', 'negligible',
                                            'underpinning', 'underpinned', 'sf15', 'justify', 'chegg', 'conferencing', 'honorable',
                                            'outsized', 'ratio', '110', 'disambiguation', 'superset', 'wyndham', 'issuer', 
                                            'complimentary', 'gilead', 'tightly', 'accompanied', 'gauge', 'realtorcom', 
                                            'assimilate', 'medal', 'originally', 'decline', 'prefect', 'figma', 'cutoff', 
                                            'homomorphic', 'interconnected', 'cube', 'renter', 'profitsharing', 'actigraph', 
                                            'harrison', 'remotework', 'writtenqualifications', 'hbo', 'settle', 'wholeperson',
                                            'relish', 'proofofconcepts', 'passivelogic', 'vertex', 'web3', 'immunotherapy',
                                            'martech', 'mainframe', 'permanente', 'worcester', 'facilitated', 'lading', 'absent',
                                            '7500000', 'costco',  'concierge', 'und', 'translated', 'rescinded', 'thinking',
                                            'jdcom', 'verisks', 'stooping',
                                            'aberdeen', 'liverampers', 'wideranging', 'soliciting', 'invaluable', 'afford',
                                            'hyundai', 'highway', 'backtesting', 'obw', 'binder', '1990', 'fixing',
                                            'climb', 'kroger', 'relatively', 'usac', 'compensationbenefits',
                                            'asd', 'composable', 'warner', '10point', 'airbnb', 'productized',
                                            'tea', 'educated', 'alternatively', 'frederick', 'reassignment', 'nonstatistical',
                                            'csr', 'besides', 'dmp', 'bxds', 'usjobscognizant',
                                            'betterment', 'didi', 'globalized', 'promoted', 'mindfulness', 'examines', 'amended',
                                            'dha', 'correctness', 'mechanistic', 'affirmativeaction', 'whatatms', 'logo', 'ir', 
                                            'hedge', 'pixel', 'sandbox', 'offset', 'quants', 'tds', 'prudentials', 'newsroom',
                                            'french', 'lesbian', 'threaten', 'impediment', 'noaa', 'intercept', '75144k', 'fed',
                                            '2040', 'blizzard', 'countrybased', 'machinedriven', 'trident', 'humandriven',
                                            'labelling', 'combatant', 'tandem', 'aspen', 'communitybased', 'castle', 'alwayson', 
                                            'aaai', 'consumercentric', 'definitive', 'im', 'inapp', 'kentucky', 'assuring', 'tiny',
                                            'highfidelity', 'aledade', 'characterbased', 'gogetter', 
                                            'hardwaresoftware', 'leaguers', 'tfl', 
                                            'atlassians', '2025', 'universitywide', 'luis', 'guardrail', 'pleasant', 'at', 
                                            'maritime', 'lover', 'satisfying', 'zeppelin', 'fidelitycareerscom',
                                            'generic', 'ky', 'roche', 'defending', 'swe', 'cytel', 'rev', 'synchronization',
                                            'classic', 'receivable', 'tactically', 'incubate', 'operationally', 'wy', 
                                            'infertility', 'consumerfacing', 'guardian', 'eeoaffirmative', 'zoetis', 'dinner',
                                            'revolut', 'egencia', 'vote', 'sum', 'exec', 'tucson', 'jewelry', 'fighting',
                                            'probe', 'hathaway', 'mostestablished', '39',
                                            'tesseract', 'adf', 'altice', 'ind123', 'immigrant', 'amherst', 'categoriesunited',
                                            'phenotypic', 'journalist', 'harm', 'lastly', 'airborne', 'detected',
                                            'explosive', 'province', 'subtype', 'glossary', 'eeo2', 'conduit', 'partitioning',
                                            'collision', 'antidiscrimination', 'outage', 'lukos', 'inbranch', 'father',
                                            'drivetime', 'transportationrelated', 'convince', 'prognostic', 'hellofresh',
                                            'knit', 'willamette', 'pod', 'privacyops', 'decrease', 'cameo', 'gsn', 'frontdoor', 
                                            'devbusinessops', 'oig', 'adoptive', 'reactive', 'addressable', 'prohibiting',
                                            'longerterm', 'ado', 'streamlines', '1400', 'packet', 'datorama', 'thatatms', 
                                            'everyoneatms', 'mumbai', 'parson', 'merges', 'delhaize',  'scrub', 'opportunity',
                                            'statewide', 'ahold', 'happier', '350', 'correcting', 'anaplan', 'erpi', 'broadband'
                                            'crown', 'dailypay', 'ergonomic', 'inmemory', 'nfl', 'ripple', 'peacock', 'readout',
                                            'pitfall', 'lipfe', 'lululemon', 'unifying', 'praescient',
                                            'fmrcom', 'kneeling', 'concurrency', 'criticism', 'watching', '800am', 'behave',
                                            'richer', 'firewall', 'wow', 'assumes', 'lecture', '300000', 'dd214', 'shortage',
                                            'pickup', 'deepest', 'interdependency', 'disruptors',
                                            '3564940', 'pave',  'pdd', 'fidelitywe', 'noninternship', 'mlb', 'shy',
                                            'stripe', 'cac', 'firstparty', 'indigo', 'fanatic', 'asu', 'restricts', 'wmd',
                                            'surfacing', 'hacker', 'interventional', '320', 'monetary', 'boulder', 'kyiv', 'cfo',
                                            'hyper', 'peertopeer', 'reviewtrackers', 'vc', 'coinnovation', 'expressed',
                                            'productsanalytic', 'welldesigned', 'ew', 'hematology', 'deems', 'geopolitical',
                                            'wifi', '1700', 'nondiscriminatory', 'gallagher', 'posed', 'alion', 'misconduct',
                                            'gear', 'exhibiting', 'absolute', 'circulation', 'wet', 'impress', 'countering',
                                            'pairprogramming', '1020', 'filed', 'usaf', 'cae', 'pregnancyrelated', '140', 
                                            'chainlink', 'ended', 'celebrity', 'wildfire', '1800', 'eoeaa', 'travisci', 'paytv', 
                                            'dohme', 'chandler', 'biomedicine', 'modeled', 'enova', 'thornton', 'drawn', 'shi',
                                            'bisexual', 'mason', 'versa', 'gleaned', 'timeframe', 'fanduel', 'stevens', 'tmobile',
                                            'incorrect', 'crouch', 'sincerelyheld', 'garden', 'unsolved', 'predictably', 'mayor',
                                            'internetscale', 'optimistic', 'workhuman', 'centrally', 'caciwgi', 'cannon',
                                            'objectorientedobject', 'crown', 'hall', 'brochure', 'broadband', 'ttec', 'neiman',
                                            'stealth', 'sketch', 'carvanas', 'residing', 'articulated', 'ccai', 'conversant',
                                            'shoe', 'hmo', 'blob', 'freewheel', 'cdao', 'milliman', 'instinct',
                                            'brazil', 'continuation', 'wyoming', 'suspicious', 'pharmacovigilance', 'relax',
                                            'dec', 'knot', 'powercenter', 'kemper', 'pacing', 'hello', 'servicedisabled', 
                                            'alexandria', 'immense', 'topology', '8a', 'ring', 'clientssponsors', 'plymouth',
                                            'warfighting', 'insatiably', 'transplant', 'notably', 'cafeteria', 'accountant',
                                            'productionized', 'pass', 'dilemma', 'returning', 'liveperson', 'proofsofconcepts',
                                            'cater', 'wild', 'mosaic', 'leaning', 'broadening', 
                                            'recognizable', 'bigpicture', 'synchronize', '5pm', 'orthodontia', 'wri', '120000', 
                                            'tobe', 'irad', 'smallest', 'fulfil', 'soar', 'welldeveloped', 'dispersion',
                                            'museum', 'sar', 'bennett', 'tracked', 'sequential', 'failing', 'fuzzy','meditation',
                                            'composition', 'hoursweek', 'mockups', 'nnsa', 'calibrate', 'fto', 'horizontal',
                                            'bcg', '1996', 'neither', 'kbi', 'onshore', 'espouse', 'longlasting', 'macbook',
                                            'ntt', 'sunpower', 'anthropology', 'numerate', 'divestiture', 'imaginative',
                                            'nccs', 'graphically', 'praxis', 'refinery', 'postoffer', 'immerse', 'nonimmigrant', 
                                            'greenplum', 'ctc', 'citis', 'spgi', 'printed', 'predictiveprescriptive',
                                            'computerworld', '8008355099', 'stephen', 'photocopier', 'variability', 'inappropriate',
                                            'storagecomputedb', 'cleanlogical', 'intrinsic', 'platformsolutions', 'reignite',
                                            'measurably', 'dmi', 'nextbest', 'organizationwide', 'massage', 'multithreading',
                                            'faq', 'straight', 'spd', 'civic', '711', 'circleci', 'assembled', 'medifast',
                                            'en', 'objectivity', 'renewed', 'diner', 'xerox', 'bny', 'farthest', 'cognite',
                                            'chime', 'paint', 'thoughtprovoking', 'in', 'bought', '32', 'actionemployer',
                                            'suntrust', 'swp', 'suspected', 'fostered', 'discriminating', 'ae', 
                                            'worldchanging', 'catch', 'inpatient', 'f35', 'brunswick', 
                                            'decisionsupport', 'reinforce', 'thimble', 'catalogue', 'xpress', '100k', 'thirty',
                                            'counterintelligence', 'carried', 'internals', 
                                            'fuse', 'realty', 'digitization', 'highlyskilled', 'revenuegenerating', '440',
                                            'clientexternal', 'modifier', 'rack', 'spain', '55000', 'rebate', 'fhir', 'opened', 
                                            'beer', 'trainee', 'bilingual', 'walnut', 'widen', 'returntooffice', 'metamorphosis',
                                            'probably', '38', 'barclays', 'importantsome', 'sri', 'unsecured', 'perhaps',
                                            'servicenowaffiliated', 'visioning', '55a', 'hippo', 'animation', 'arming', 'viz', 
                                            'asis', 'simulink', 'iccv', 'summation', 'addicted', 'ayco', 'bbt', 'faa', 
                                            'visanet', 'cashless', 'viewer', 'compact', 'delaware', 'hughes', 'sdtmadam',
                                            'instantly', 'gen', 'abroad', 'ddt', 'seoul', 'aarrr', 'pirate', 'selflearning',
                                            'businessagency', 'stalking', 'marketo', 'fdic', '9001', '97', 'leisure', 'upbeat',
                                            'enduras', 'degradation', 'gainesville', 'northeast', 'synthesizes', 'substantially',
                                            'paidtime', 'hunting', 'changer', 'starbucks', 'ariba', 'bama', 'iai', 'harvest',
                                            'intelligenesis', 'signet', 'pak', 'glean', 'nbc', 'horiba', 'rfp', 'milwaukee',
                                            'auditable', 'wwwaccenturecom', 'ro', 'neon', 'englishspanish', 'rewire', 'wafer',
                                            'seedinvest', 'cohesion', '43', 'sondermind', 'aerial', 'lime', 'recurlys', '42',
                                            'moore', 'ulta', 'cbs', 'americorps', 
                                            'essentially', 'passive', 'dentalvision', 'noncompete', 'lumedic', 'groupons',
                                            'disparity', 'visualized', 'touched', 'juggling', 'datto', '18000', 'interactivity', 
                                            'versioned', 'salesforces', 'cdp', 'biorad', 'ben', 'disseminating', 'readability',
                                            'calculator', 'male', 'icd', 'ppt', 'stash', 'kidney', 'msu', 'cited', 'tempus', 
                                            'infuse', 'critique', 'uis', 'multilabel', 'detects', 'jupiter', 'betting', 
                                            'productionization', 'stopping', 'netimpact', 'andela', 'dearborn', 'burning',
                                            '16000', 'singularity', 'enriches', 'plane', 'xator', '360000', 'pediatrics', 
                                            'imagevideo', 'videographers', 'marketdisrupting', 'headlinedriving', 'vessel',
                                            'societychanging', 'veritas', 'assisted', 'engineersscientists', 'infographics',
                                            'cyberattacks', 'irrespective', 'meade', 'apron', 'streamlit', '1999', 'interchange',
                                            'ead', 'bunch', 'tumblr', 
                                            'gendernonbinary', 'rudeness', 'photographer', 'reused', 'firstplace', 'healthtech',
                                            'potff', 'crisp', 'exploding', 'slate', 'concertai', 'foresight', 'regulartemporary',
                                            '35000', 'underpin', 'morse', 'usb', 'apac', 'doctrine', 'lawrence',
                                            'podcast', 'referenced', 'wwwistockcom', 'industrialization', 'nielseniq',
                                            '275', 'athletic', 'hyperion', 'genre', 'upandcoming', 'interpretive', 'gone', 
                                            'monfri', 'caribbean', 'necessity', 'declaration', 'multibilliondollar', 'compose',
                                            'pinnacle', 'un', 'nba', 'mentally', 'oportuns', 'oportun', 'retire', 'removal',
                                            'meritocracy', 'implementable', 'penchant', 'feedzai', 'kraft', 'clientside', 'heinz',
                                            'regionspecific', 'rpl', 'prevalence', 'lawyer', '2dose', 'showcasing', 'kdp',
                                            'facial', 'louisville', '130000', 'being', 'fullyvaccinated', 'homepage', 'multiagent',
                                            'highland', 'scheduler', 'racing', 'rotating', 'intricacy', 'httpsbenefitsindeedjobs',
                                            'iehp', 'mayvin', 'inconsistent', 'linguistic', 'clearing', 'broadest', 'bomber', 
                                            'moon', 'lims', 'disconnected', 'patience', 'assuming',
                                            'uncertain', 'superiority', 'loyal', 'ofac', 'paste', 'bing', 'rhode', 'smithfield',
                                            'iheartradio', 'vue', 'atypical', 'endlessly', 'rendering', 'llnl', 'whollyowned',
                                            'village', 'evolutionary', 'datafirst', 'psychometric', 'crosscultural', 'daylight',
                                            '14043', 'agronomy', 'refusal', 'cbre', '1974', 'lewis', 'bleeding',
                                            'anderson', 'inspirational', 'beneficiary', 'brighams', 'viasat', 'dao', 'incyte',
                                            '570000', 'blink', 'musical', 'latestage', 'twist', 'modernized', 'chester', 'komodo',
                                            '607415a', 'discovers', 'comp', 'multimedia', 'opportunityaffirmative',
                                            'strauss', 'ribbon', 'outfit', 'uscentcom', 'scientistengineer', 'hyperpersonalized',
                                            'sow', 'rebellion', 'unposting', 'subsequently', 'out', 'c2c', 'mph',
                                            'surrogacy', 'kwx', 'replicate', 'distributes', 'inter', 
                                            'abnormality', 'installed', 'basket', 'agendasetting', 'iowa', 'expertly', 
                                            'wwwguidehousecom', 'briefly', 'persuasion', 'commencing', 'productionalizing',
                                            'licensures', 'budgetary', '119880','patreon', 
                                            'ihub', 'acrobat', 'attrition', 'eventbrite', 'samara', 'sentar', 'infra',
                                            'combination', 'handson', 'high', 'higher', '35', 'military', 'militaryveteran',
                                            'molecular', 'prior', 'track', 'solid', 'track', 'asaservice', 'ad', 
                                            'transform', 'natural', 'next', 'ppas', 'maker', 'sparkcognition', 'flow',
                                            'fast', 'faster','high', 'highly', 'oriented', 'geli', 'sephora', 'serco', 'faire',
                                            'mirati', 'upwork', 'jpl', 'totus', 'quantcast', 'fermentation', 'irap', 'rady',
                                            'bolt', 'dexcom', 'zest', 'terawatt', 'barbara', 'pinner', 'usra', 'ritual',
                                            'jgi', 'moloco', 'airtable', 'moonshot', 'ericsson', 'blackberry', 'tri', 
                                            'sarscov2', 'tinder', 'vida', 'sunnyvale', 'nauto', 'mst', 'shortform', 'bambee',
                                            'thirdlove', 'bric', 'nitro', 'jakarta', 'metabolomics', 'kp', 'atomwise',
                                            '1823', 'goodrx', 'grail', 'franciscos', 'freenome', 'netskope', 'medimpact',
                                            'nvidias', 'viant', 'bsba', 'descript', 'jerry', 'openx', 'bouqs', 'setsail',
                                            'cdph', 'ox', 'iron', 'apd', 'multiplechoice', 'driscolls', 'deferral',
                                            'updater', 'mlds', 'smoke', 'topps', 'ax', 'softbank', 'remodeling',
                                            'annapolis',]))) 
     
    # create stop_words object and toggle on/off additional stop words
    # stop_words = nltk.corpus.stopwords.words('english') + additional_stopwords + ds_cred_terms + ds_prof_skill_terms + ds_soft_skill_terms + ds_tech_skill_terms
    # stop_words = nltk.corpus.stopwords.words('english') + ds_cred_terms + ds_prof_skill_terms + ds_soft_skill_terms + ds_tech_skill_terms
    stop_words = stopwords.words('english')

    # initialize lemmatizer and execute lemmatization; this is the slowest part of the processing
    print('   Lemmatizing...')
    wnl = nltk.stem.WordNetLemmatizer()
    terms_for_nlp = [wnl.lemmatize(word) for word in progressbar(words) if word not in stop_words]
 
    # execute post-lemmatization stopword removal to drop unnecessary lemma
    print('   Post-lemmatization stopword removal...')
    terms_for_nlp = [x for x in progressbar(terms_for_nlp) if x not in additional_stopwords]

    # create a dictionary for term corrections (e.g., misspellings, etc.); values are final form for each term
    term_fixes = {'accreditation': 'accredited',
                  'accurately': 'accurate',
                  'accuracy': 'accurate',
                  'adapt': 'adaptive',
                  'hoc': 'adhoc',
                  'impromptu': 'adhoc',
                  'advance': 'advanced',
                  'aggregate': 'aggregation',
                  'aggregating': 'aggregation',
                  'aggregated': 'aggregation',
                  'agilescrum': 'agile scrum',
                  'scrum': 'agile scrum',
                  'aibased': 'ai',
                  'aidriven': 'ai',
                  'aidata': 'ai data',
                  'aimachine': 'ai machine',
                  'aiml': 'ai machine learning',
                  'aimldata': 'ai machine learning data',
                  'aimlnlp': 'ai machine learning nlp',
                  'ainlp': 'ai nlp',
                  'aipytorch': 'ai pytorch',
                  'algorithmic': 'algorithm',
                  'algorithmsmodels': 'algorithm model',
                  'ambiguous': 'ambiguity',
                  'amis': 'ami',
                  'conda': 'anaconda',
                  'analyticsrelated': 'analytics',
                  'analysis': 'analytics',
                  'analytical': 'analytics',
                  'analytic': 'analytics',
                  'analyticsbased': 'analytics',
                  'analytically': 'analytics',
                  'analyse': 'analytics',
                  'analysing': 'analytics',
                  'analyzes': 'analytics',
                  'analyzed': 'analytics',
                  'analyzing': 'analytics',
                  'analyze': 'analytics',
                  'reportinganalytics': 'analytics',
                  'dataanalyticsinformation': 'analytics data',
                  'analyticsdata': 'analytics data',
                  'analysismodeling': 'analytics model',
                  'processinganalyticsscience': 'analytics science',
                  'apis': 'api',
                  'applying': 'applied',
                  'esri': 'arcgis',
                  'architect': 'architecture',
                  'architecting': 'architecture',
                  'architectural': 'architecture',
                  'articulates': 'articulate',
                  'autoencoders': 'autoencoder',
                  'automated': 'automate',
                  'automates': 'automate',
                  'automation': 'automate',
                  'automationefficiencies': 'automate',
                  'automating': 'automate',
                  'autonomously': 'automate',
                  'fullyautonomous': 'automate',
                  'autonomous': 'automate',
                  'awsazure': 'aws azure',
                  'baccalaureate': 'bachelors',
                  'bachelor': 'bachelors',
                  'bsc': 'bachelors',
                  'bachelorsmasters': 'bachelors masters',
                  'bsms': 'bachelors masters',
                  'bsmsphd': 'bachelors masters phd',
                  'undergraduate': 'bachelors',
                  'master': 'masters',
                  'bayesian': 'bayes',
                  'bestpractice': 'best practice',
                  'bestpractices': 'best practice',
                  'large': 'big',
                  'bigdata': 'big data',
                  'multiterabyte': 'big data',
                  'petabyte': 'big data',
                  'petabytescale': 'big data',
                  'terabyte': 'big data',
                  'teradata': 'big data',
                  'bigquery': 'big query',
                  'bioinformatic': 'bioinformatics',
                  'bioinformatician': 'bioinformatics',
                  'biochemistry': 'bioscience',
                  'bioengineering': 'bioscience',
                  'biology': 'bioscience',
                  'biological': 'bioscience',
                  'biomechanics': 'bioscience',
                  'biophysics': 'bioscience',
                  'epidemiology': 'bioscience',
                  'neuroimaging': 'bioscience',
                  'neuroscience': 'bioscience',
                  'biostatisticians': 'biostatistics',
                  'biostatistician': 'biostatistics', 
                  'biostatistical': 'biostatistics',
                  'biotechnology': 'biotech',
                  'block': 'blocks',
                  'bootcamps': 'bootcamp',
                  'bootstrapping': 'bootstrap',
                  'caffe2': 'caffe',
                  'categorization': 'categorize',
                  'licensecertification': 'certification',
                  'certificate': 'certification',
                  'certified': 'certification',
                  'challenging': 'challenge',
                  'chatbots': 'chatbot',
                  'classifier': 'classification',
                  'classify': 'classification',
                  'clean': 'cleaning',
                  'cleansing': 'cleaning',
                  'cleansed': 'cleaning',
                  'condition': 'cleaning',
                  'conditioned': 'cleaning',
                  'conditioning': 'cleaning',
                  'manipulate': 'cleaning',
                  'manipulation': 'cleaning',
                  'munging': 'cleaning',
                  'wrangle': 'cleaning',
                  'wrangler': 'cleaning',
                  'wranglingcleansing': 'cleaning',
                  'wrangling': 'cleaning',
                  'clearly': 'clear',
                  'commandline': 'cli',
                  'clientfacing': 'client',
                  'clientfocused': 'client',
                  'customer': 'client',
                  'customerfacing': 'client',
                  'customerfocused': 'client',
                  'customerobsessed': 'client',
                  'decisionmaker': 'client',
                  'decisionmaking': 'client',
                  'decisionmakers': 'client',
                  'firmsclients' : 'client',
                  'usercentered' : 'client',
                  'cloudbased': 'cloud',
                  'cloudnative': 'cloud',
                  'cloudready': 'cloud',
                  'cluster': 'clustering',
                  'coding': 'code',
                  'closely': 'collaborate',
                  'collaborative': 'collaborate',
                  'collaboration': 'collaborate',
                  'collaboratively': 'collaborate',
                  'collaborating': 'collaborate',
                  'collaborates': 'collaborate',
                  'collaborator': 'collaborate',
                  'connect': 'collaborate',
                  'cooperation': 'collaborate',
                  'cooperative': 'collaborate',
                  'cooperate': 'collaborate',
                  'coworkers': 'collaborate',
                  'downtoearth': 'collaborate',
                  'inform': 'collaborate',
                  'interact': 'collaborate',
                  'partnering': 'collaborate',
                  'partnership': 'collaborate',
                  'partnered': 'collaborate',
                  'partner': 'collaborate',
                  'personable': 'collaborate',
                  'player': 'collaborate',
                  'team': 'collaborate',
                  'teambased': 'collaborate',
                  'teamdriven': 'collaborate',
                  'teaming': 'collaborate',
                  'teamfirst': 'collaborate',
                  'teamfocused': 'collaborate',
                  'teammate': 'collaborate',
                  'teamwork': 'collaborate',
                  'teamoriented': 'collaborate',
                  'teamplayer': 'collaborate',
                  'together': 'collaborate',
                  'liaise': 'collaborate',
                  'cooperatively': 'collaborate',
                  'collect': 'collection',
                  'collected': 'collection',
                  'collecting': 'collection',
                  'collegeuniversity': 'college',
                  'school': 'college',
                  'university': 'college',
                  'communicated': 'communicate',
                  'communicates': 'communicate',
                  'communicating': 'communicate',
                  'communication': 'communicate',
                  'communicator': 'communicate',
                  'complex': 'complexity',
                  'computation': 'computer',
                  'computational': 'computer',
                  'cs': 'computer science',
                  'cv': 'computer vision',
                  'concisely': 'concise',
                  'succinct': 'concise',
                  'consultant': 'consulting',
                  'containerorchestration' : 'container orchestration',
                  'contributor': 'contribute',
                  'cnn': 'convolutional neural network',
                  'cnns': 'convolutional neural network',
                  'rcnn': 'convolutional neural network',
                  'correlate': 'correlation',
                  'cost': 'costing',
                  'course': 'coursework',
                  'credentialing': 'credential',
                  'crfs': 'crf',
                  'crossdisciplinary': 'crossfunctional',
                  'crossfunctionally': 'crossfunctional',
                  'interdisciplinary': 'crossfunctional',
                  'multidisciplined': 'crossfunctional',
                  'multidisciplinary': 'crossfunctional',
                  'multifunctional': 'crossfunctional',
                  'cross': 'crossfunctional',
                  'css3': 'css',
                  'curated': 'curate',
                  'curation': 'curate',
                  'curiosity': 'curious',
                  'inquisitive': 'curious',
                  'question': 'curious',
                  'selfdevelopment': 'curious',
                  'study': 'curious',
                  'cyber': 'cybersecurity',
                  'd3': 'd3js',
                  'reportsdashboards': 'dashboard',
                  'dashboarding': 'dashboard',
                  'dataai': 'data ai',
                  'databasesdata': 'database data',
                  'businessdata': 'business data',
                  'dataset': 'data',
                  'datasets': 'data',
                  'dataanalysis': 'data analysis',
                  'dataanalytics': 'data analytics',
                  'datacomputing': 'data computing',
                  'datascience': 'data science',
                  'dsml': 'data science machine learning',
                  'datadriven': 'data-driven',
                  'datafocused': 'data-driven',
                  'dataoriented': 'data-driven',
                  'databacked': 'data-driven',
                  'driven': 'data-driven',
                  'databased': 'database',
                  'mining': 'datamining',
                  'mine': 'datamining',
                  'datastores': 'datastore',
                  'store': 'datastore',
                  'deeplearning': 'deep learning',
                  'learningdeep': 'deep learning',
                  'diploma': 'degree',
                  'deliver': 'delivery',
                  'delivering': 'delivery',
                  'deliverable': 'delivery',
                  'delivered': 'delivery',
                  'designed': 'design',
                  'designing': 'design',
                  'designer': 'design',
                  'detect': 'detection',
                  'detecting': 'detection',
                  'develop': 'development',
                  'developed': 'development',
                  'develops': 'development',
                  'developer': 'development',
                  'developing': 'development',
                  'devopsfocused': 'devops',
                  'devsecops': 'devops',
                  'hardworking': 'diligent',
                  'hard': 'diligent',
                  'diligently': 'diligent',
                  'diligence': 'diligent',
                  'deeplearning4j': 'dl4j',
                  'highdimensional': 'dimensional',
                  'highdimensionality': 'dimensional',
                  'multidimensional': 'dimensional',
                  'containerbased': 'container',
                  'containerization': 'container',
                  'containerized': 'container',
                  'containerizing': 'container',
                  'document': 'documenting',
                  'domainspecific': 'domain knowledge',
                  'econometric': 'econometrics',
                  'exploratory': 'eda',
                  'educationexperience': 'education experience',
                  'efficiency': 'efficient',
                  'efficiently': 'efficient',
                  'elastic': 'elasticsearch',
                  'empathy': 'empathetic',
                  'engage': 'engagement',
                  'engages': 'engagement',
                  'engineer': 'engineering',
                  'ensembling': 'ensemble',
                  'willingness': 'enthusiasm',
                  'energetic': 'enthusiastic',
                  'attitude': 'enthusiastic',
                  'highenergy': 'enthusiastic',
                  'energy': 'enthusiastic',
                  'energized': 'enthusiastic',
                  'motivated': 'enthusiastic',
                  'motivating': 'enthusiastic',
                  'motivate': 'enthusiastic',
                  'motivation': 'enthusiastic',
                  'highlymotivated': 'enthusiastic',
                  'passion': 'enthusiastic',
                  'passionate': 'enthusiastic',
                  'passionately': 'enthusiastic',
                  'positive': 'enthusiastic',
                  'enthusiast': 'enthusiastic',
                  'selfdriven': 'enthusiastic',
                  'selfmotivation': 'enthusiastic',
                  'selfmotivated': 'enthusiastic',
                  'selfdriving': 'enthusiastic',
                  'entrepreneur': 'entrepreneurial',
                  'growth': 'entrepreneurial',
                  'estimating': 'estimate',
                  'ethically': 'ethical',
                  'trustworthy': 'ethical',
                  'unbiased': 'ethical',
                  'trust': 'ethical',
                  'trusted': 'ethical',
                  'ethic': 'ethics',
                  'elt': 'etl',
                  'etlelt': 'etl',
                  'etls': 'etl',
                  'extracttransformload': 'etl',
                  'extract': 'etl',
                  'extraction': 'etl',
                  'extracting': 'etl',
                  'excellent': 'excellence',
                  'quality': 'excellence',
                  'strong': 'excellence',
                  'highquality': 'excellence',
                  'mastery': 'excellence',
                  'mastering': 'excellence',
                  'caliber': 'excellence',
                  'great': 'excellence',
                  'greatest': 'excellence',
                  'greater': 'excellence',
                  'greatly': 'excellence',
                  'greatness': 'excellence',
                  'working': 'experience',
                  'cexperience': 'experience',
                  'experienceeducation': 'experience',
                  'experienced': 'experience',
                  'experimental': 'experiment',
                  'experimentation': 'experiment',
                  'experimenting': 'experiment',
                  'expertlevel': 'expert',
                  'expertise': 'expert',
                  'explains': 'explain',
                  'explainable': 'explain',
                  'explainability': 'explain',
                  'explanation': 'explain',
                  'exploration': 'exploratory',
                  'everchanging': 'fast-paced',
                  'fastmoving': 'fast-paced',
                  'fastpaced': 'fast-paced',
                  'quickly': 'fast-paced',
                  'quicker': 'fast-paced',
                  'quick': 'fast-paced',
                  'pace': 'fast-paced',
                  'paced': 'fast-paced',
                  'speed': 'fast-paced',
                  'highspeed': 'fast-paced',
                  'timeliness': 'fast-paced',
                  'rapid': 'fast-paced',
                  'discipline': 'field',
                  'agility': 'flexible',
                  'flex': 'flexible',
                  'flexibility': 'flexible',
                  'adaptive': 'flexible',
                  'dynamic': 'flexible',
                  'dynamically': 'flexible',
                  'nimble': 'flexible',
                  'flexibly': 'flexible',
                  'versatile': 'flexible',
                  'evolving': 'flexible',
                  'evolve': 'flexible',
                  'evolves': 'flexible',
                  'forecasting': 'forecast',
                  'forwardthinking': 'forward-thinking',
                  'forward': 'forward-thinking',
                  'gans': 'gan',
                  'generative': 'gan',
                  'boost': 'gbm',
                  'boosted': 'gbm',
                  'boosting': 'gbm',
                  'google': 'gcp',
                  'generated': 'generation',
                  'genomic': 'genomics',
                  'geography': 'geospatial',
                  'geographic': 'geospatial',
                  'geographical': 'geospatial',
                  'geographically': 'geospatial',
                  'geoint': 'geospatial',
                  'ggplot2': 'ggplot',
                  'ggplot2must': 'ggplot',
                  'github': 'git',
                  'gitgithub': 'git',
                  'gitlab': 'git',
                  'gitlabs': 'git',
                  'glms': 'glm',
                  'glmregression': 'glm regression',
                  'cpugpu': 'gpu',
                  'gpus': 'gpu',
                  'graduatelevel': 'graduate',
                  'graphbased': 'graph',
                  'hadoopdata': 'hadoop data',
                  'hadoopspark': 'hadoop spark',
                  'lakehadoop' : 'data lake hadoop',
                  'help': 'helpful',
                  'helped': 'helpful',
                  'helping': 'helpful',
                  'holistically': 'holistic',
                  'highperformance': 'hpc',
                  'html5': 'html',
                  'htmlcss': 'html css',
                  'hypothesisdriven': 'hypothesis',
                  'identify': 'identification',
                  'identifies': 'identification',
                  'identifying': 'identification',
                  'image': 'imagery',
                  'imaging': 'imagery',
                  'effectively': 'impact',
                  'highimpact': 'impact',
                  'difference': 'impact',
                  'differentiated': 'impact',
                  'success': 'impact',
                  'impact': 'impactful',
                  'improvement': 'impactful',
                  'improves': 'impactful',
                  'improve': 'impactful',
                  'improving': 'impactful',
                  'influence': 'impactful',
                  'value': 'impactful',
                  'outcome': 'impactful',
                  'implementation': 'implement',
                  'independently': 'independent',
                  'individual': 'independent',
                  'selfdirected': 'independent',
                  'selfdirection': 'independent',
                  'selfmanage': 'independent',
                  'selfmanagement': 'independent',
                  'selfsufficient': 'independent',
                  'unique': 'individual',
                  'inferencing': 'inference',
                  'inferencereasoning': 'inference',
                  'ingest': 'ingestion',
                  'ingesting': 'ingestion',
                  'ingested': 'ingestion',
                  'ingests': 'ingestion',
                  'innovate': 'innovative',
                  'innovator': 'innovative',
                  'innovating': 'innovative',
                  'innovatively': 'innovative',
                  'creativity': 'innovative',
                  'creative': 'innovative',
                  'creatively': 'innovative',
                  'outsidethebox': 'innovative',
                  'innovatives': 'innovative',
                  'innovates': 'innovative',
                  'innovation': 'innovative',
                  'insights': 'insight',
                  'savvy': 'intelligence',
                  'intelligenceinsights': 'intelligence insight',
                  'intelligencemachine': 'intelligence machine',
                  'intellectual': 'intelligent',
                  'intellectually': 'intelligent',
                  'intelligently': 'intelligent',
                  'brilliant': 'intelligent',
                  'interpret': 'interpretation',
                  'interpreting': 'interpretation',
                  'invoice': 'invoicing',
                  'iterating': 'iterate',
                  'iteration': 'iterate',
                  'iterative': 'iterate',
                  'iteratively': 'iterate',
                  'javascripttypescript': 'java script',
                  'javac': 'java c',
                  'kera': 'keras',
                  'largescale': 'large',
                  'direct': 'leadership',
                  'directs': 'leadership',
                  'directing': 'leadership',
                  'director': 'leadership',
                  'lead': 'leadership',
                  'leader': 'leadership',
                  'leading': 'leadership',
                  'oversee': 'leadership',
                  'oversees': 'leadership',
                  'overseeing': 'leadership',
                  'owner': 'leadership',
                  'ownership': 'leadership',
                  'track': 'leadership',
                  'learningenabled': 'learning',
                  'learningai': 'learning ai',
                  'learningartificial': 'learning ai',
                  'learningdata': 'learning data',
                  'learningpredictive': 'learning predictive',
                  'learningstatistical': 'learning statistics',
                  'linearlogistic': 'linear logistic',
                  'lm': 'linear model',
                  'linearnonlinear': 'linear nonlinear',
                  'windowslinux': 'linux windows',
                  'linuxbased': 'linux unix',
                  'linuxunix': 'linux unix',
                  'unixlinux': 'linux unix',
                  'listen': 'listening',
                  'listens': 'listening',
                  'logical': 'logic',
                  'logically': 'logic',
                  'lstms': 'lstm',
                  'machinedeep': 'machine deep',
                  'learningenhanced': 'machine learning',
                  'machinelearning': 'machine learning',
                  'ml': 'machine learning',
                  'mldriven': 'machine learning',
                  'mlops': 'machine learning',
                  'mlrelated': 'machine learning',
                  'mlai': 'machine learning ai',
                  'mldl': 'machine learning deep learning',
                  'mlnlp': 'machine learning nlp',
                  'manages': 'management',
                  'managerial': 'management',
                  'supervise': 'management',
                  'supervises': 'management',
                  'supervision': 'management',
                  'supervisory': 'management',
                  'manage': 'management',
                  'managing': 'management',
                  'managed': 'management',
                  'map': 'mapping',
                  'msc': 'masters',
                  'msma': 'masters',
                  'mastersphd': 'masters phd',
                  'msphd': 'masters phd',
                  'math': 'mathematics',
                  'mathematician': 'mathematics',
                  'mathematical': 'mathematics',
                  'mathematicalstatistical': 'mathematics statistics',
                  'mathematicsstatistics': 'mathematics statistics',
                  'mathstatistics': 'mathematics statistics',
                  'pyplot': 'matplotlib',
                  'measurable': 'measure',
                  'measurement': 'measure',
                  'meet': 'meeting',
                  'coach': 'mentor',
                  'coaching': 'mentor',
                  'mentoring': 'mentor',
                  'mentorship': 'mentor',
                  'merging': 'merge',
                  'medidata': 'metadata',
                  'methodology': 'method',
                  'methodological': 'method',
                  'rigor': 'meticulous',
                  'rigorous': 'meticulous',
                  'attention': 'meticulous',
                  'attentiontodetail': 'meticulous',
                  'rigorously': 'meticulous',
                  'detail': 'meticulous',
                  'detailed': 'meticulous',
                  'detailoriented': 'meticulous',
                  'precision': 'meticulous',
                  'precise': 'meticulous',
                  'thorough': 'meticulous',
                  'microservices': 'microservice',
                  'mlib': 'mllib',
                  'modelbased': 'model',
                  'modeldriven': 'model',
                  'modelintent': 'model',
                  'modeler': 'model',
                  'modeling': 'model',
                  'modelling': 'model',
                  'modelsalgorithms': 'model algorithm',
                  'modelingmachine': 'model machine',
                  'modelsmethods': 'model method',
                  'mongo': 'mongodb',
                  'multiarm': 'multiarmed',
                  'multitasking': 'multitask',
                  'tasking': 'multitask',
                  'simultaneously': 'multitask',
                  'simultaneous': 'multitask',
                  'multivariable': 'multivariate',
                  'nn': 'neural network',
                  'nns': 'neural network',
                  'language': 'nlp',
                  'nlpnlu': 'nlp',
                  'nlg': 'nlp',
                  'nlu': 'nlp',
                  'nlpml': 'nlp machine learning',
                  'normalizing': 'normalize',
                  'normalization': 'normalize',
                  'cuttingedge': 'novel',
                  'groundbreaking': 'novel',
                  'latest': 'novel',
                  'leadingedge': 'novel',
                  'modern': 'novel',
                  'generation': 'novel',
                  'nextgen': 'novel',
                  'nextgeneration': 'novel',
                  'stateofart': 'novel',
                  'stateoftheart': 'novel',
                  'cutting': 'novel',
                  'numeric': 'numerical',
                  'numpyscipy': 'numpy scipy',
                  'need': 'objective',
                  'mission': 'objective',
                  'opensource': 'open source',
                  'operation': 'operations',
                  'operating': 'operations',
                  'operational': 'operations',
                  'optimal': 'optimization',
                  'optimally': 'optimization',
                  'optimize': 'optimization',
                  'optimise': 'optimization',
                  'optimized': 'optimization',
                  'optimizes': 'optimization',
                  'optimizing': 'optimization',
                  'orchestrate': 'orchestration',
                  'orchestrates': 'orchestration',
                  'orchestrating': 'orchestration',
                  'organize': 'organized',
                  'organization': 'organized',
                  'wellorganized': 'organized',
                  'structure': 'organized',
                  'timemanagement': 'organized',
                  'panda': 'pandas',
                  'pearsons': 'pearson',
                  'peerreviewed': 'peer-reviewed',
                  'doctorate': 'phd',
                  'doctoral': 'phd',
                  'phdms': 'phd masters',
                  'physic': 'physics',
                  'physicsbased': 'physics',
                  'pipelining': 'pipeline',
                  'platformtensor': 'platform tensor',
                  'postgres': 'postgresql',
                  'postgresdb': 'postgresql',
                  'powerbi': 'power bi',
                  'predicting': 'predictive',
                  'prediction': 'predictive',
                  'predict': 'predictive',
                  'prioritization': 'prioritize',
                  'prioritizing': 'prioritize',
                  'prioritizes': 'prioritize',
                  'prioritized': 'prioritize',
                  'priority': 'prioritize',
                  'prioritise': 'prioritize',
                  'highpriority': 'prioritize',
                  'proactively': 'proactive',
                  'initiative': 'proactive',
                  'selfstarter': 'proactive',
                  'selfstarting': 'proactive',
                  'selfstarters': 'proactive',
                  'initiate': 'proactive',
                  'initiating': 'proactive',
                  'probabilistic': 'probability',
                  'problemsolve': 'problem-solving',
                  'problemsolver': 'problem-solving',
                  'problemsolvers': 'problem-solving',
                  'problemsolving': 'problem-solving',
                  'solve': 'problem-solving',
                  'solver': 'problem-solving',
                  'solving': 'problem-solving',
                  'acumen': 'problem-solving',
                  'think': 'problem-solving',
                  'process': 'processing',
                  'productively': 'productive',
                  'productivity': 'productive',
                  'professionally': 'professional',
                  'professionalism': 'professional',
                  'proficiently': 'proficient',
                  'literate': 'proficient',
                  'proficiency': 'proficient',
                  'profitable': 'profitability',
                  'profit': 'profitability',
                  'profitably': 'profitability',
                  'programmatic': 'program',
                  'programing': 'programming',
                  'programmingscripting': 'programming scripting',
                  'prototype': 'prototyping',
                  'fluency': 'proven',
                  'fluent': 'proven',
                  'fluently': 'proven',
                  'record': 'proven',
                  'python3': 'python',
                  'pythonbased': 'python',
                  'pythonr': 'python r',
                  'pythonrscala': 'python r scala',
                  'qliksense': 'qlik',
                  'qlikview': 'qlik',
                  'qlikviewpower': 'qlik power bi',
                  'quantifying': 'quantitative',
                  'quantify': 'quantitative',
                  'quantitatively': 'quantitative',
                  'quantification': 'quantitative',
                  'quantifiable':'quantitative',
                  'rpython': 'r python',
                  'rrstudio': 'r rstudio',
                  'rshiny': 'r shiny',
                  'rd': 'r&d',
                  'rf': 'random forest',
                  'randomized': 'randomization',
                  'react': 'reactjs',
                  'reason': 'reasoning',
                  'recommenders': 'recommender',
                  'related': 'relevant',
                  'reliability': 'reliable',
                  'repeatability': 'repeatable',
                  'researcher': 'research',
                  'fortitude': 'resilient',
                  'persistent': 'resilient',
                  'persistence': 'resilient',
                  'resilience': 'resilient',
                  'respect': 'respectful',
                  'retraining': 'retrain',
                  'reviewed': 'review',
                  'riskbased': 'risk',
                  'risktakers': 'risk',
                  'risktaking': 'risk',
                  'rnns': 'rnn',
                  'sage': 'sagemaker',
                  'servicessagemaker': 'sagemaker',
                  'sale': 'sales',
                  'sa': 'sas',
                  'sasbase': 'sas',
                  'sasgraph': 'sas',
                  'sasstat': 'sas',
                  'sasmacro': 'sas macro',
                  'sassql': 'sas sql',
                  'scalable': 'scale',
                  'scientist': 'science',
                  'sciencebased': 'science',
                  'scienceanalytics': 'science analytics',
                  'sciencedata': 'science data',
                  'scienceengineering': 'science engineering',
                  'sciencelogic': 'science logic',
                  'sciencemachine': 'science machine',
                  'scientifically': 'scientific',
                  'scrappy': 'scrapy',
                  'script': 'scripting',
                  'scrumagile': 'scrum agile',
                  'sdks': 'sdk',
                  'secretsci': 'secret sci',
                  'segment': 'segmentation',
                  'cando': 'self-confident',
                  'grit': 'self-confident',
                  'gritty': 'self-confident',
                  'tenacious': 'self-confident',
                  'tenacity': 'self-confident',
                  'confidence': 'self-confident',
                  'confident': 'self-confident',
                  'poised': 'self-confident',
                  'semantically': 'semantic',
                  'semantics': 'semantic',
                  'simulate': 'simulation',
                  'scikitimage' : 'skimage',
                  'scikit': 'sklearn',
                  'scikitlearn': 'sklearn',
                  'sparkml': 'spark machine learning',
                  'sparksql': 'spark sql',
                  'speechlanguage': 'speech',
                  'spectroscopy': 'spectrometry',
                  'sqlbased': 'sql',
                  'slq': 'sql',
                  'sqlonhadoop':'sql hadoop',
                  'transactsql': 'sql',
                  'tsql': 'sql',
                  'sqlnosql': 'sql nosql',
                  'sqlpython': 'sql python',
                  'statistician': 'statistics',
                  'stat': 'statistics',
                  'stats': 'statistics',
                  'statistic': 'statistics',
                  'statistically': 'statistics',
                  'statistical': 'statistics',
                  'statisticsbiostatistics': 'statistics biostatistics',
                  'statisticalmachine': 'statistics machine',
                  'statisticalml': 'statistics machine learning',
                  'statisticalmathematical': 'statistics mathematics',
                  'stochasticprocesses': 'stochastic',
                  'present': 'storytelling',
                  'presentation': 'storytelling',
                  'storyboarding': 'storytelling',
                  'storyboards': 'storytelling',
                  'storyteller': 'storytelling',
                  'story': 'storytelling',
                  'presenting': 'storytelling',
                  'presented': 'storytelling',
                  'strategic': 'strategy',
                  'strategically': 'strategy',
                  'strategist': 'strategy',
                  'structuredunstructured': 'structured unstructured',
                  'subjectmatter': 'subject matter',
                  'sme': 'subject matter expert',
                  'smes': 'subject matter expert',
                  'supervisedunsupervised': 'supervised unsupervised',
                  'svms': 'svm',
                  'tableausplunkcvent': 'tableau splunk computer vision',
                  'tech': 'technical',
                  'technically': 'technical',
                  'technological': 'technology',
                  'tensor': 'tensorflow',
                  'tensorflowkeras': 'tensorflow keras',
                  'tensorflowpytorch': 'tensorflow pytorch',
                  'test': 'testing',
                  'theoretical': 'theory',
                  'thought': 'thought-leadership',
                  'thoughtleadership': 'thought-leadership',
                  'timeseries': 'time series',
                  'tooling': 'tool',
                  'topsecret': 'top secret',
                  'tssci': 'top secret sci',
                  'train': 'training',
                  'transformative': 'transformation',
                  'transformer': 'transformers',
                  'translating': 'translation',
                  'treebased': 'tree',
                  'debug': 'troubleshoot',
                  'debugging': 'troubleshoot',
                  'troubleshoots': 'troubleshoot',
                  'troubleshooting': 'troubleshoot',
                  'understands': 'understand',
                  'understood': 'understand',
                  'unixshell': 'unix shell',
                  'use': 'user',
                  'oral': 'verbal',
                  'orally': 'verbal',
                  'verbally': 'verbal',
                  'verbalwritten': 'verbal written',
                  'visionimage': 'vision image',
                  'graphic': 'visualization',
                  'plot': 'visualization',
                  'table': 'visualization',
                  'visualize': 'visualization',
                  'visualisation': 'visualization',
                  'visualizing': 'visualization',
                  'visuals': 'visualization',
                  'visualizingpresenting': 'visualization storytelling',
                  'vlookups': 'vlookup',
                  'warehousing': 'warehouse',
                  'window': 'windows',
                  'windowsunix': 'windows unix',
                  'stream': 'workstreams',
                  'write': 'written',
                  'writing': 'written',
                  'writtenverbal': 'written verbal'}
    
    # alphabetize term_fixes by value, print to console and paste back into code for ease of parsing
    dict(sorted(term_fixes.items(), key=lambda item: item[1]))
        
    # correct misspellings, erroneous concatenations, term ambiguities, etc.; collapse synonyms into single terms
    print('\nCorrecting misspellings, erroneous concatenations, ambiguities, etc...')
    df_term_fixes = pd.DataFrame(terms_for_nlp, columns=['terms'])
    df_term_fixes['terms'].replace(dict(zip(list(term_fixes.keys()), list(term_fixes.values()))), regex=False, inplace=True)
    terms_for_nlp = list(df_term_fixes['terms'])
     
    return terms_for_nlp, additional_stopwords, term_fixes


def clean_listings_for_nlp(series_of_interest, additional_stopwords, term_fixes):
    '''
    Create a dataframe of cleaned Indeed job listings. The dataframe maintains listing integrity by keeping the 
    1:1 relationiship between job listing and dataframe record.

    Parameters
    ----------
    series_of_interest : Series
        A variable set in the main program, series_of_interest contains the targeted job listing data for NLP processing.

    Returns
    -------
    df_jobs : dataframe
        Contains the cleaned and parsed job listings in a 1:1 data structure (i.e., each listing is its own record).

    '''
    print('\nCreating dataframe of tokenized job listings...')
    
    # convert series_of_interest to a dataframe
    df_jobs = pd.DataFrame(series_of_interest)
    
    # remove contractions
    df_jobs['job_description'] = df_jobs['job_description'].apply(lambda x: [contractions.fix(word) for word in x.split()])
    df_jobs['job_description'] = [' '.join(map(str, l)) for l in df_jobs['job_description']]
    
    # # tokenize the new df_jobs dataframe
    # df_jobs['job_description'] = df_jobs['job_description'].apply(word_tokenize)

    # tokenize the new df_jobs dataframe
    print('\nTokenizing the new df_jobs dataframe...')    
    df_jobs['job_description'] = df_jobs['job_description'].progress_apply(word_tokenize)
    
    # convert tokenized text to lowercase
    print('\nConverting tokenized text to lowercase...')
    df_jobs['job_description'] = df_jobs['job_description'].progress_apply(lambda x: [word.lower() for word in x])
    
    # remove punctuation
    print('\nRemoving punctuation...')
    other_punctuation = ['...', '’', '–', '``', '“', '”']
    df_jobs['job_description'] = df_jobs['job_description'].progress_apply(lambda x: [word for word in x if word not in string.punctuation])
    df_jobs['job_description'] = df_jobs['job_description'].progress_apply(lambda x: [word for word in x if word not in other_punctuation])
    
    # remove standard NLTK stopwords
    print('\nRemoving standard NLTK stopwords...')
    stop_words_nltk = set(stopwords.words('english'))
    df_jobs['job_description'] = df_jobs['job_description'].progress_apply(lambda x: [word for word in x if word not in stop_words_nltk])
    
    # remove additional industry-specific NLTK stopwords
    print('\nRemoving additional industry-specific NLTK stopwords...')
    df_jobs['job_description'] = df_jobs['job_description'].progress_apply(lambda x: [word for word in x if word not in additional_stopwords])
    
    # execute term_fixes 
    print('\nExecuting term fixes...')
    df_jobs['job_description'] = df_jobs['job_description'].explode().replace(term_fixes).groupby(level=-1).agg(list)
      
    return df_jobs  


def visualize_indeed_metadata(df):
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
    print('\n***** Visualization *****')
    print('\nVisualizing Indeed data charts...')
    
    # configure plot size, seaborne style and font scale
    plt.figure(figsize=(7, 10))
    sns.set_style('dark')
    sns.set(font_scale = 1.30)
    
    # create countplot for state counts
    ax = sns.countplot(y='state_name', data=df, palette='gist_gray', 
                       order = df['state_name'].value_counts().index) # Blues_d, mako_r, ocean, gist_gray, gist_gray_r, icefire
    ax.set_title('Jobs by State')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    

def visualize_n_grams(n_grams, ds_cred_terms, ds_tech_skill_terms, ds_soft_skill_terms, ds_prof_skill_terms,
                      terms_for_nlp, series_of_interest, additional_stopwords, term_fixes, df):
    '''
    Visualize the n_grams created by the nlp_count_n_grams function.

    Parameters
    ----------
    n_grams : dataframe
        Contains the processed n_grams; sorted by count from highest to lowest; converted to a df in this function.
    
    ds_cred_terms : list
        Contains keywords pertaining to data science credentials (e.g., 'bachelors', 'experience', etc.)
        
    terms_for_nlp : list
        List containing scraped and cleaned terms from the series of interest; created in the clean_for nlp function.

    Returns
    -------
    None. Directly outputs visualizations.

    ''' 
    def visualize_all_monograms(n_grams, unique_titles_viz):
        '''
        print('\nVisualizing all monograms...')
        Visualize all monograms across all skillsets (e.g., crednetials, technical, soft and professional). Can also
        be tuned to produce bigrams only based on the n_gram_count variable set in the main function.

        Parameters
        ----------
        n_grams : dataframe
            Contains the cleaned and processed n_grams; sorted by count from highest to lowest.

        Returns
        -------
        None. Directly outputs visualizations.

        '''
        # bound the count of ngram records to be visualized
        n_grams_sns = n_grams.iloc[:20] # toggle how many records to show in the visualization
        
        # create a horizontal barplot visualizing n_gram counts from greatest to least across all skills, companies and job titles
        plt.figure(figsize=(7, 10))
        sns.set_style('dark')
        sns.set(font_scale = 1.8)
        
        ax = sns.barplot(x='count',
                         y='grams',
                         data=n_grams_sns,
                         orient='h',
                         palette='mako_r') # crest, mako, 'mako_d, Blues_d, mako_r, ocean, gist_gray, gist_gray_r, icefire
        
        ax.set_title('Key Terms for Data Scientist Jobs',
                     fontsize=24,
                     loc='center')
        ax.set(ylabel=None)
        ax.set_xlabel('Count', fontsize=18)

        plt.figtext(0.325, 0.475,
                    '← and this is a red pointer',
                    fontsize=16,
                    color='r',
                    fontweight='demibold')
        
        plt.figtext(0.140, 0.010,
                    textwrap.fill(f'Data: {len(df)} Indeed job listings for {" ".join(str(x) for x in unique_titles_viz)} collected between {min(df.scrape_date)} and {max(df.scrape_date)}',
                                  width=70),
                    bbox=dict(facecolor='none', boxstyle='square', edgecolor='none', pad=0.2),
                    fontsize=14,
                    color='black',
                    fontweight='regular',
                    style='italic',
                    ha='left',
                    in_layout=True,
                    wrap=True)   


    def visualize_credentials(n_grams, ds_cred_terms, terms_for_nlp, series_of_interest, additional_stopwords, 
                              term_fixes, df_jobs_raw, unique_titles_viz):
        '''
        Create visualizations for monograms and bigrams assoicated with the credential skill list.

        Parameters
        ----------
        n_grams : dataframe
            Contains the processed n_grams; sorted by count from highest to lowest.
        ds_cred_terms : list
            Contains keywords pertaining to data science credentials (e.g., 'bachelors', 'experience', etc.).
        terms_for_nlp : list
            List containing scraped and cleaned terms from the series of interest; created in the clean_for nlp function.
        series_of_interest : series
            A variable set in the main program, series_of_interest contains the targeted job listing data for NLP processing.
        additional_stopwords : list
            A list to capture all domain-specific stopwords and 'stop-lemma'.
        term_fixes : dictionary
            A dictionary for correcting misspelled, duplicated or consolidated terms in the series of interest.

        Returns
        -------
        None. Directly outputs visualizations.

        '''  
        print('\nVisualizing Credentials...')
        def monograms_and_bigrams_by_count():
            '''

            Visualize the top n combined list of monograms and bigrams according to how many times they appear
            in the series of interest. Visualizes only the raw counts.

            Returns
            -------
            bigram_match_to_cred_list : list
                A list of bigrams in which each bigram has at least one term matching a term in the ds_cred_terms list.

            '''
            # subset the monograms that appear in the credentials list
            mask_monogram = n_grams.grams.isin(ds_cred_terms)
            monograms_df_sns = n_grams[mask_monogram]
            
            # generate bigrams from the full terms_for_nlp list
            n_gram_count = 2
            n_gram_range_start, n_gram_range_stop  = 0, 100
            bigrams = nlp_count_n_grams(terms_for_nlp, n_gram_count, n_gram_range_start, n_gram_range_stop)
            
            # subset the bigrams for which at least one term appears in the credentials list
            bigram_match_to_cred_list = [x for x in bigrams.grams if any(b in x for b in ds_cred_terms)]
            mask_bigram = bigrams.grams.isin(bigram_match_to_cred_list)
            bigrams_df_sns = bigrams[mask_bigram]
    
            # add the monograms and bigrams
            ngram_combined_sns = pd.concat([monograms_df_sns, bigrams_df_sns], axis=0, ignore_index=True)
    
            # identify noisy, duplicate or unhelpful terms and phrases
            ngrams_to_silence = ['data', 'experience', 'business', 'science', 'year', 'ability',
                                'system', 'experience experience']
            
            # exclude unwanted terms and phrases
            ngram_combined_sns = ngram_combined_sns[~ngram_combined_sns.grams.isin(ngrams_to_silence)].reset_index(drop=True)
    
            # create a horizontal barplot visualizing data science credentials
            plt.figure(figsize=(7, 10))
            sns.set_style('dark')
            sns.set(font_scale = 1.8)
            
            ax = sns.barplot(x='count',
                             y='grams',
                             data=ngram_combined_sns,
                             order=ngram_combined_sns.sort_values('count', ascending = False).grams[:20],
                             orient='h',
                             palette='mako_r') # crest, mako, 'mako_d, Blues_d, mako_r, ocean, gist_gray, gist_gray_r, icefire
                       
            ax.set_title(textwrap.fill('Consider How Intensely Employers Care about Each Credential Focus Area', width=40),
                         fontsize=24,
                         loc='center')   
            ax.set(ylabel=None)
            ax.set_xlabel('Count', fontsize=18)
                      
            plt.figtext(0.140, 0.010,
                        textwrap.fill(f'Data: {len(df)} Indeed job listings for {" ".join(str(x) for x in unique_titles_viz)} collected between {min(df.scrape_date)} and {max(df.scrape_date)}',
                                      width=70),
                        bbox=dict(facecolor='none', boxstyle='square', edgecolor='none', pad=0.2),
                        fontsize=14,
                        color='black',
                        fontweight='regular',
                        style='italic',
                        ha='left',
                        in_layout=True,
                        wrap=True) 
            
            return bigram_match_to_cred_list
               
        
        def monograms_by_percentage(df_jobs_raw):
            '''
            Visualize the credential monograms as a function of percentage of listings in which the monogram appears.
            This function makes first use of dataframes wherein each record is a job listing.

            Parameters
            ----------
            df_jobs_raw : dataframe
                A dataframe wherein each record is a unique listing, and each term in each listing is tokenized. df_jobs_raw is
                created in the visualize_n_grams function so that it can be used in subfunctions.
            
            Returns
            -------
            df_jobs_mono : dataframe
                A dataframe wherein each record is a job listing, and each column is a boolean flag for each
                monogram in the ds_cred_terms list.  The final row and column each contain totals for their 
                respective job listing and credential term, respectively. The job_description field is dropped
                before the summations.

            '''
            # flag job listings if they contain the credential term (from stack question)
            df_jobs_mono = df_jobs_raw.copy()
            df_jobs_mono[ds_cred_terms] = [[any(w==term for w in lst) for term in ds_cred_terms] for lst in df_jobs_mono['job_description']]
            
            # calculate sum of all credential terms for both rows and columns
            df_jobs_mono = df_jobs_mono.drop('job_description', axis=1)
            df_jobs_mono.loc[:, 'total_mono_in_list'] = df_jobs_mono.sum(axis=1) # this does rows; need to plot these to filter out noisy/broken listings; can be used for the unicorn index
            df_jobs_mono.loc['total_mono', :] = df_jobs_mono.sum(axis=0) # this does columns; need to drop the job_description field
                 
            # drop all rows except the total row, transform columns and rows and rename the fields
            df_jobs_mono_sns = df_jobs_mono.drop(df_jobs_mono.index.to_list()[:-1], axis = 0).melt()
            df_jobs_mono_sns.rename(columns={'variable': 'ds_cred_term','value': 'count'}, inplace=True)
            
            # calculate a percentages field; will need to divide by len(df_jobs) * 100
            df_jobs_mono_sns['percentage'] = [round(x / len(df_jobs_raw)*100, 2) for x in df_jobs_mono_sns['count']]
            df_jobs_mono_sns = df_jobs_mono_sns[df_jobs_mono_sns['ds_cred_term'].str.contains('total')==False]      
            
            # create a horizontal barplot visualizing data science credential monograms as a percentage of job listings
            plt.figure(figsize=(7, 10))
            sns.set_style('dark')
            sns.set(font_scale = 1.8)            
           
            ax = sns.barplot(x='percentage',
                             y='ds_cred_term',
                             data=df_jobs_mono_sns,
                             order=df_jobs_mono_sns.sort_values('percentage', ascending = False).ds_cred_term[:25],
                             orient='h',
                             palette='mako_r') # crest, mako, 'mako_d, Blues_d, mako_r, ocean, gist_gray, gist_gray_r, icefire
            
            ax.set_title(textwrap.fill('**For Cred Parsing: Monograms by Percentage', width=40), # original title: Percentage Key Terms for Data Scientist Credentials
                         fontsize=24,
                         loc='center')
            ax.set(ylabel=None)
            ax.set_xlabel('Percentage', fontsize=18)
            
            plt.figtext(0.140, 0.010,
                        textwrap.fill(f'Data: {len(df)} Indeed job listings for {" ".join(str(x) for x in unique_titles_viz)} collected between {min(df.scrape_date)} and {max(df.scrape_date)}',
                                      width=70),
                        bbox=dict(facecolor='none', boxstyle='square', edgecolor='none', pad=0.2),
                        fontsize=14,
                        color='black',
                        fontweight='regular',
                        style='italic',
                        ha='left',
                        in_layout=True,
                        wrap=True)   
            
            return df_jobs_mono
        
        
        def bigrams_by_percentage(df_jobs_raw, bigram_match_to_cred_list): 
            '''
            Visualize the credential bigrams as a function of percentage of listings in which the monogram appears.

            Parameters
            ----------
            df_jobs_raw : dataframe
                A dataframe wherein each record is a unique listing, and each term in each listing is tokenized. df_jobs_raw is
                created in the visualize_n_grams function so that it can be used in subfunctions.
            bigram_match_to_cred_list : list
                A list of bigrams in which each bigram has at least one term matching a term in the ds_cred_terms list.

            Returns
            -------
            df_jobs_bigrams : dataframe
                A dataframe wherein each record is a job listing, and each column is a boolean flag for each
                bigram in the bigram_match_to_cred_list list.  The final row and column each contain totals for their 
                respective job listing and credential bigram, respectively. The job_description field is dropped
                before the summations.

            '''               
            # create df_jobs_bigrams from a copy of df_jobs_raw
            df_jobs_bigrams = df_jobs_raw.copy()
            
            # flag job listings if they contain the credential term (from stack question)
            def find_bigram_match_to_cred_list(data):
                output = np.zeros((data.shape[0], len(bigram_match_to_cred_list)), dtype=bool)
                for i, d in enumerate(data):
                    possible_bigrams = [' '.join(x) for x in list(nltk.bigrams(d)) + list(nltk.bigrams(d[::-1]))]
                    indices = np.where(np.isin(bigram_match_to_cred_list, list(set(bigram_match_to_cred_list).intersection(set(possible_bigrams)))))
                    output[i, indices] = True
                return list(output.T)
    
            output = find_bigram_match_to_cred_list(df_jobs_bigrams['job_description'].to_numpy())
            df_jobs_bigrams = df_jobs_bigrams.assign(**dict(zip(bigram_match_to_cred_list, output)))
            
            # identify and silence noisy, duplicate or unhelpful bigrams 
            bigrams_to_silence = ['analytics data', 'experience experience', 'collaborate data', 'ability work',
                                  'experience knowledge', 'statistics analytics', 'development data', 'management data',
                                  'mathematics statistics', 'experience collaborate', 'data data']
            df_jobs_bigrams = df_jobs_bigrams.drop(columns=bigrams_to_silence, axis=0, errors='ignore')
            
            # calculate sum of all credential terms for both rows and columns
            df_jobs_bigrams = df_jobs_bigrams.drop('job_description', axis=1)
            df_jobs_bigrams.loc[:, 'total_bigram_in_list'] = df_jobs_bigrams.sum(axis=1) # this does rows; need to plot these to filter out noisy/broken listings; can be used for the unicorn index
            df_jobs_bigrams.loc['total_bigram', :] = df_jobs_bigrams.sum(axis=0) # this does columns; need to drop the job_description field
            
            # drop all rows except the total row, transform columns and rows and rename the fields
            df_jobs_bigrams_sns = df_jobs_bigrams.drop(df_jobs_bigrams.index.to_list()[:-1], axis = 0).melt()
            df_jobs_bigrams_sns.rename(columns={'variable': 'ds_cred_term','value': 'count'}, inplace=True)
            
            # calculate a percentages field; will need to divide by len(df_jobs) * 100
            df_jobs_bigrams_sns['percentage'] = [round(x / len(df_jobs_raw)*100, 2) for x in df_jobs_bigrams_sns['count']]
            df_jobs_bigrams_sns = df_jobs_bigrams_sns[df_jobs_bigrams_sns['ds_cred_term'].str.contains('total')==False]
            
            # create a horizontal barplot visualizing data science credentials as a percentage of job listings
            plt.figure(figsize=(7, 10))
            sns.set_style('dark')
            sns.set(font_scale = 1.8)               
       
            ax = sns.barplot(x='percentage',
                             y='ds_cred_term',
                             data=df_jobs_bigrams_sns,
                             order=df_jobs_bigrams_sns.sort_values('percentage', ascending = False).ds_cred_term[:25],
                             orient='h',
                             palette='mako_r') # crest, mako, 'mako_d, Blues_d, mako_r, ocean, gist_gray, gist_gray_r, icefire
            
            ax.set_title(textwrap.fill('**For Cred Parsing: Bigrams by Percentage', width=40), # original title: Percentage Key Bigrams for Data Scientist Credentials
                         fontsize=24,
                         loc='center')
            ax.set(ylabel=None)
            ax.set_xlabel('Percentage', fontsize=18)
            
            plt.figtext(0.140, 0.010,
                        textwrap.fill(f'Data: {len(df)} Indeed job listings for {" ".join(str(x) for x in unique_titles_viz)} collected between {min(df.scrape_date)} and {max(df.scrape_date)}',
                                      width=70),
                        bbox=dict(facecolor='none', boxstyle='square', edgecolor='none', pad=0.2),
                        fontsize=14,
                        color='black',
                        fontweight='regular',
                        style='italic',
                        ha='left',
                        in_layout=True,
                        wrap=True)  
            
            return df_jobs_bigrams


        def monograms_and_bigrams_by_percentage(df_jobs_mono, df_jobs_bigrams):
            '''
            Visualize the combined credential monograms and bigrams as a function of percentage of listings in which
            either the monogram or bigram appears.

            Parameters
            ----------
            df_jobs_mono : dataframe
                A dataframe wherein each record is a job listing, and each column is a boolean flag for each
                monogram in the ds_cred_terms list.  The final row and column each contain totals for their 
                respective job listing and credential term, respectively. The job_description field is dropped
                before the summations.
            df_jobs_bigrams : dataframe
                A dataframe wherein each record is a job listing, and each column is a boolean flag for each
                bigram in the bigram_match_to_cred_list list.  The final row and column each contain totals for their 
                respective job listing and credential bigram, respectively. The job_description field is dropped
                before the summations.

            Returns
            -------
            None. Directly outputs visualizations.

            '''
            # combine monograms and bigrams into a single dataframe
            df_jobs_combined = pd.concat([df_jobs_mono, df_jobs_bigrams], axis=1)
            
            # melt the dataframe, drop nan rows, rename the fields and drop the two 'total' rows
            df_jobs_combined_sns = df_jobs_combined.drop(df_jobs_combined.index.to_list()[:-2], axis = 0).melt()
            df_jobs_combined_sns = df_jobs_combined_sns[df_jobs_combined_sns['value'].notna()]
            df_jobs_combined_sns.rename(columns={'variable': 'ds_cred_term_phrase','value': 'count'}, inplace=True)
            df_jobs_combined_sns = df_jobs_combined_sns[~df_jobs_combined_sns.ds_cred_term_phrase.isin(['total_mono_in_list', 'total_bigram_in_list'])]
    
            # calculate a percentages field
            df_jobs_combined_sns['percentage'] = [round(x / len(df_jobs_raw)*100, 2) for x in df_jobs_combined_sns['count']]
      
            # visualize combined mongrams and bigrams
            plt.figure(figsize=(7, 10))
            sns.set_style('dark')
            sns.set(font_scale = 1.8)  
       
            ax = sns.barplot(x='percentage',
                             y='ds_cred_term_phrase',
                             data=df_jobs_combined_sns,
                             order=df_jobs_combined_sns.sort_values('percentage', ascending = False).ds_cred_term_phrase[:20],
                             orient='h',
                             palette='mako_r') # crest, mako, 'mako_d, Blues_d, mako_r, ocean, gist_gray, gist_gray_r, icefire
            
            ax.set_title(textwrap.fill('Focus Your Credentialing on High-Priority Areas', width=30), # original title: Percentage Key Bigrams for Data Scientist Credentials
                         fontsize=24,
                         loc='center')
            ax.set(ylabel=None)
            ax.set_xlabel('Percentage', fontsize=18)
            
            plt.figtext(0.140, 0.010,
                        textwrap.fill(f'Data: {len(df)} Indeed job listings for {" ".join(str(x) for x in unique_titles_viz)} collected between {min(df.scrape_date)} and {max(df.scrape_date)}',
                                      width=70),
                        bbox=dict(facecolor='none', boxstyle='square', edgecolor='none', pad=0.2),
                        fontsize=14,
                        color='black',
                        fontweight='regular',
                        style='italic',
                        ha='left',
                        in_layout=True,
                        wrap=True) 

        
        # visualize credentials by count
        bigram_match_to_cred_list = monograms_and_bigrams_by_count()
                
        # visualize credentials by percentage
        df_jobs_mono = monograms_by_percentage(df_jobs_raw)
        df_jobs_bigrams = bigrams_by_percentage(df_jobs_raw, bigram_match_to_cred_list)
        monograms_and_bigrams_by_percentage(df_jobs_mono, df_jobs_bigrams)
        

    def visualize_technicals(n_grams, ds_tech_skill_terms, terms_for_nlp, series_of_interest, additional_stopwords,
                             term_fixes, df_jobs_raw, unique_titles_viz):
        '''
        Create visualizations for monograms and bigrams assoicated with the technical skill list.

        Parameters
        ----------
        n_grams : dataframe
            Contains the processed n_grams; sorted by count from highest to lowest.
        ds_tech_skill_terms : list
            Contains keywords pertaining to data science technical skills (e.g., 'python', 'tableau', etc.).
        terms_for_nlp : list
            List containing scraped and cleaned terms from the series of interest; created in the clean_for nlp function.
        series_of_interest : series
            A variable set in the main program, series_of_interest contains the targeted job listing data for NLP processing.
        additional_stopwords : list
            A list to capture all domain-specific stopwords and 'stop-lemma'.
        term_fixes : dictionary
            A dictionary for correcting misspelled, duplicated or consolidated terms in the series of interest.

        Returns
        -------
        None. Directly outputs visualizations.

        '''
        print('\nVisualizing Technical Skills...')
        
        def monograms_and_bigrams_by_count():
            '''
            Visualize the top n combined list of monograms and bigrams according to how many times they appear
            in the series of interest. Visualizes only the raw counts.

            Returns
            -------
            bigram_match_to_tech_list : list
                A list of bigrams in which each bigram has at least one term matching a term in the ds_tech_skill_terms list.

            '''          
            # subset the monograms that appear in the technical skills list
            mask_monogram = n_grams.grams.isin(ds_tech_skill_terms)
            monograms_df_sns = n_grams[mask_monogram]
            
            # generate bigrams from the full terms_for_nlp list
            n_gram_count = 2
            n_gram_range_start, n_gram_range_stop  = 0, 100
            bigrams = nlp_count_n_grams(terms_for_nlp, n_gram_count, n_gram_range_start, n_gram_range_stop)
            
            # subset the bigrams for which at least one term appears in the technical skills list
            bigram_match_to_tech_list = [x for x in bigrams.grams if any(b in x for b in ds_tech_skill_terms)]
            mask_bigram = bigrams.grams.isin(bigram_match_to_tech_list)
            bigrams_df_sns = bigrams[mask_bigram]
    
            # add the monograms and bigrams
            ngram_combined_sns = pd.concat([monograms_df_sns, bigrams_df_sns], axis=0, ignore_index=True)
    
            # identify noisy, duplicate or unhelpful terms and phrases
            # ngrams_to_silence = ['data', 'experience', 'business', 'science', 'year', 'ability', 'system'] # these from cred
            ngrams_to_silence = ['system'] 
            
            # exclude unwanted terms and phrases
            ngram_combined_sns = ngram_combined_sns[~ngram_combined_sns.grams.isin(ngrams_to_silence)].reset_index(drop=True)
    
            # create a horizontal barplot visualizing data science technical skills
            plt.figure(figsize=(7, 10))
            sns.set_style('dark')
            sns.set(font_scale = 1.8) 
                
            ax = sns.barplot(x='count',
                             y='grams',
                             data=ngram_combined_sns,
                             order=ngram_combined_sns.sort_values('count', ascending = False).grams[:25],
                             orient='h',
                             palette='mako_r') # crest, mako, 'mako_d, Blues_d, mako_r, ocean, gist_gray, gist_gray_r, icefire
            
            ax.set_title(textwrap.fill('Consider How Intensely Employers Care about Each Technical Skill', width=40),
                         fontsize=24,
                         loc='center')   
            ax.set(ylabel=None)
            ax.set_xlabel('Count', fontsize=18)
            
            plt.figtext(0.140, 0.010,
                        textwrap.fill(f'Data: {len(df)} Indeed job listings for {" ".join(str(x) for x in unique_titles_viz)} collected between {min(df.scrape_date)} and {max(df.scrape_date)}',
                                      width=70),
                        bbox=dict(facecolor='none', boxstyle='square', edgecolor='none', pad=0.2),
                        fontsize=14,
                        color='black',
                        fontweight='regular',
                        style='italic',
                        ha='left',
                        in_layout=True,
                        wrap=True)  
                       
            return bigram_match_to_tech_list
        
    
        def monograms_by_percentage(df_jobs_raw): 
            '''
            Visualize the technical skill monograms as a function of percentage of listings in which the monogram appears.
            This function makes first use of dataframes wherein each record is a job listing.

            Parameters
            ----------
            df_jobs_raw : dataframe
                A dataframe wherein each record is a unique listing, and each term in each listing is tokenized. df_jobs_raw is
                created in the visualize_n_grams function so that it can be used in subfunctions.

            Returns
            -------
            df_jobs_mono : dataframe
                A dataframe wherein each record is a job listing, and each column is a boolean flag for each
                monogram in the ds_tech_skill_terms list.  The final row and column each contain totals for their 
                respective job listing and technical skill term, respectively. The job_description field is dropped
                before the summations.

            '''
            # flag job listings if they contain the technical skill term (from stack question)
            df_jobs_mono = df_jobs_raw.copy()
            df_jobs_mono[ds_tech_skill_terms] = [[any(w==term for w in lst) for term in ds_tech_skill_terms] for lst in df_jobs_mono['job_description']]
            
            # calculate sum of all technical skill terms for both rows and columns
            df_jobs_mono = df_jobs_mono.drop('job_description', axis=1)
            df_jobs_mono.loc[:, 'total_mono_in_list'] = df_jobs_mono.sum(axis=1) # this does rows; need to plot these to filter out noisy/broken listings; can be used for the unicorn index
            df_jobs_mono.loc['total_mono', :] = df_jobs_mono.sum(axis=0) # this does columns; need to drop the job_description field
                 
            # drop all rows except the total row, transform columns and rows and rename the fields
            df_jobs_mono_sns = df_jobs_mono.drop(df_jobs_mono.index.to_list()[:-1], axis = 0).melt()
            df_jobs_mono_sns.rename(columns={'variable': 'ds_tech_skill_term','value': 'count'}, inplace=True)
            
            # calculate a percentages field
            df_jobs_mono_sns['percentage'] = [round(x / len(df_jobs_raw)*100, 2) for x in df_jobs_mono_sns['count']]
            df_jobs_mono_sns = df_jobs_mono_sns[df_jobs_mono_sns['ds_tech_skill_term'].str.contains('total')==False]
            
            # create a horizontal barplot visualizing data science technical skill monograms as a percentage of job listings
            plt.figure(figsize=(7, 10))
            sns.set_style('dark')
            sns.set(font_scale = 1.8) 
      
            ax = sns.barplot(x='percentage',
                             y='ds_tech_skill_term',
                             data=df_jobs_mono_sns,
                             order=df_jobs_mono_sns.sort_values('percentage', ascending = False).ds_tech_skill_term[:25],
                             orient='h',
                             palette='mako_r') # crest, mako, 'mako_d, Blues_d, mako_r, ocean, gist_gray, gist_gray_r, icefire
            
            
            ax.set_title(textwrap.fill('**For Tech Parsing: Monograms by Percentage', width=40), # original title: Percentage Key Terms for Data Scientist Credentials
                         fontsize=24,
                         loc='center')
            ax.set(ylabel=None)
            ax.set_xlabel('Percentage', fontsize=18)
            
            plt.figtext(0.140, 0.010,
                        textwrap.fill(f'Data: {len(df)} Indeed job listings for {" ".join(str(x) for x in unique_titles_viz)} collected between {min(df.scrape_date)} and {max(df.scrape_date)}',
                                      width=70),
                        bbox=dict(facecolor='none', boxstyle='square', edgecolor='none', pad=0.2),
                        fontsize=14,
                        color='black',
                        fontweight='regular',
                        style='italic',
                        ha='left',
                        in_layout=True,
                        wrap=True)  
            
            return df_jobs_mono
    

        def bigrams_by_percentage(df_jobs_raw, bigram_match_to_tech_list): 
            '''
            Visualize the technical skills bigrams as a function of percentage of listings in which the monogram appears.

            Parameters
            ----------
            df_jobs_raw : dataframe
                A dataframe wherein each record is a unique listing, and each term in each listing is tokenized. df_jobs_raw is
                created in the visualize_n_grams function so that it can be used in subfunctions.
            bigram_match_to_tech_list : list
                A list of bigrams in which each bigram has at least one term matching a term in the ds_tech_skill_terms list.

            Returns
            -------
            df_jobs_bigrams : dataframe
                A dataframe wherein each record is a job listing, and each column is a boolean flag for each
                bigram in the bigram_match_to_tech_list list.  The final row and column each contain totals for their 
                respective job listing and technical skill bigram, respectively. The job_description field is dropped
                before the summations.

            '''               
            # create df_jobs_bigrams from a copy of df_jobs_raw
            df_jobs_bigrams = df_jobs_raw.copy()
            
            # flag job listings if they contain the technical skill term (from stack question)
            def find_bigram_match_to_tech_list(data):
                output = np.zeros((data.shape[0], len(bigram_match_to_tech_list)), dtype=bool)
                for i, d in enumerate(data):
                    possible_bigrams = [' '.join(x) for x in list(nltk.bigrams(d)) + list(nltk.bigrams(d[::-1]))]
                    indices = np.where(np.isin(bigram_match_to_tech_list, list(set(bigram_match_to_tech_list).intersection(set(possible_bigrams)))))
                    output[i, indices] = True
                return list(output.T)
    
            output = find_bigram_match_to_tech_list(df_jobs_bigrams['job_description'].to_numpy())
            df_jobs_bigrams = df_jobs_bigrams.assign(**dict(zip(bigram_match_to_tech_list, output)))
            
            # identify and silence noisy, duplicate or unhelpful bigrams 
            bigrams_to_silence = ['analytics data', 'experience experience', 'collaborate data', 'ability work',
                                  'experience knowledge', 'statistics analytics', 'development data', 'management data',
                                  'mathematics statistics', 'experience collaborate', 'data data']
            df_jobs_bigrams = df_jobs_bigrams.drop(columns=bigrams_to_silence, axis=0, errors='ignore')
            
            # calculate sum of all technical skill terms for both rows and columns
            df_jobs_bigrams = df_jobs_bigrams.drop('job_description', axis=1)
            df_jobs_bigrams.loc[:, 'total_bigram_in_list'] = df_jobs_bigrams.sum(axis=1) # this does rows; need to plot these to filter out noisy/broken listings; can be used for the unicorn index
            df_jobs_bigrams.loc['total_bigram', :] = df_jobs_bigrams.sum(axis=0) # this does columns; need to drop the job_description field
            
            # drop all rows except the total row, transform columns and rows and rename the fields
            df_jobs_bigrams_sns = df_jobs_bigrams.drop(df_jobs_bigrams.index.to_list()[:-1], axis = 0).melt()
            df_jobs_bigrams_sns.rename(columns={'variable': 'ds_tech_term','value': 'count'}, inplace=True)
            
            # calculate a percentages field
            df_jobs_bigrams_sns['percentage'] = [round(x / len(df_jobs_raw)*100, 2) for x in df_jobs_bigrams_sns['count']]
            df_jobs_bigrams_sns = df_jobs_bigrams_sns[df_jobs_bigrams_sns['ds_tech_term'].str.contains('total')==False]
            
            # create a horizontal barplot visualizing data science credentials as a percentage of job listings
            plt.figure(figsize=(7, 10))
            sns.set_style('dark')
            sns.set(font_scale = 1.8)               
       
            ax = sns.barplot(x='percentage',
                             y='ds_tech_term',
                             data=df_jobs_bigrams_sns,
                             order=df_jobs_bigrams_sns.sort_values('percentage', ascending = False).ds_tech_term[:25],
                             orient='h',
                             palette='mako_r') # crest, mako, 'mako_d, Blues_d, mako_r, ocean, gist_gray, gist_gray_r, icefire
            
            ax.set_title(textwrap.fill('**For Tech Parsing: Bigrams by Percentage', width=40), # original title: Percentage Key Bigrams for Data Scientist Credentials
                         fontsize=24,
                         loc='center')
            ax.set(ylabel=None)
            ax.set_xlabel('Percentage', fontsize=18)
            
            plt.figtext(0.140, 0.010,
                        textwrap.fill(f'Data: {len(df)} Indeed job listings for {" ".join(str(x) for x in unique_titles_viz)} collected between {min(df.scrape_date)} and {max(df.scrape_date)}',
                                      width=70),
                        bbox=dict(facecolor='none', boxstyle='square', edgecolor='none', pad=0.2),
                        fontsize=14,
                        color='black',
                        fontweight='regular',
                        style='italic',
                        ha='left',
                        in_layout=True,
                        wrap=True)  
            
            return df_jobs_bigrams    


        def monograms_and_bigrams_by_percentage(df_jobs_mono, df_jobs_bigrams):
            '''
            Visualize the combined technical skill monograms and bigrams as a function of percentage of listings in which
            either the monogram or bigram appears.
    
            Parameters
            ----------
            df_jobs_mono : dataframe
                A dataframe wherein each record is a job listing, and each column is a boolean flag for each
                monogram in the ds_tech_skill_terms list.  The final row and column each contain totals for their 
                respective job listing and technical skill term, respectively. The job_description field is dropped
                before the summations.
            df_jobs_bigrams : dataframe
                A dataframe wherein each record is a job listing, and each column is a boolean flag for each
                bigram in the bigram_match_to_tech_list list.  The final row and column each contain totals for their 
                respective job listing and technical skill bigram, respectively. The job_description field is dropped
                before the summations.
    
            Returns
            -------
            None. Directly outputs visualizations.
    
            '''
            # combine monograms and bigrams into a single dataframe
            df_jobs_combined = pd.concat([df_jobs_mono, df_jobs_bigrams], axis=1)
            
            # melt the dataframe, drop nan rows, rename the fields and drop the two 'total' rows
            df_jobs_combined_sns = df_jobs_combined.drop(df_jobs_combined.index.to_list()[:-2], axis = 0).melt()
            df_jobs_combined_sns = df_jobs_combined_sns[df_jobs_combined_sns['value'].notna()]
            df_jobs_combined_sns.rename(columns={'variable': 'ds_tech_term_phrase','value': 'count'}, inplace=True)
            df_jobs_combined_sns = df_jobs_combined_sns[~df_jobs_combined_sns.ds_tech_term_phrase.isin(['total_mono_in_list', 'total_bigram_in_list'])]
    
            # calculate a percentages field
            df_jobs_combined_sns['percentage'] = [round(x / len(df_jobs_raw)*100, 2) for x in df_jobs_combined_sns['count']]
      
            # visualize combined mongrams and bigrams
            plt.figure(figsize=(7, 10))
            sns.set_style('dark')
            sns.set(font_scale = 1.8)  
       
            ax = sns.barplot(x='percentage',
                             y='ds_tech_term_phrase',
                             data=df_jobs_combined_sns,
                             order=df_jobs_combined_sns.sort_values('percentage', ascending = False).ds_tech_term_phrase[:20],
                             orient='h',
                             palette='mako_r') # crest, mako, 'mako_d, Blues_d, mako_r, ocean, gist_gray, gist_gray_r, icefire
            
            ax.set_title(textwrap.fill('Focus Your Learning Time on High-Priority Technical Skills', width=30), # original title: Percentage Key Bigrams for Data Scientist Credentials
                         fontsize=24,
                         loc='center')
            ax.set(ylabel=None)
            ax.set_xlabel('Percentage', fontsize=18)
            
            plt.figtext(0.140, 0.010,
                        textwrap.fill(f'Data: {len(df)} Indeed job listings for {" ".join(str(x) for x in unique_titles_viz)} collected between {min(df.scrape_date)} and {max(df.scrape_date)}',
                                      width=70),
                        bbox=dict(facecolor='none', boxstyle='square', edgecolor='none', pad=0.2),
                        fontsize=14,
                        color='black',
                        fontweight='regular',
                        style='italic',
                        ha='left',
                        in_layout=True,
                        wrap=True) 

    
        # visualize technical skills by count
        bigram_match_to_tech_list = monograms_and_bigrams_by_count()
        
        # visualize technical skills by percentage
        df_jobs_mono = monograms_by_percentage(df_jobs_raw)
        df_jobs_bigrams = bigrams_by_percentage(df_jobs_raw, bigram_match_to_tech_list)
        monograms_and_bigrams_by_percentage(df_jobs_mono, df_jobs_bigrams)


    def visualize_soft(n_grams, ds_soft_skill_terms, terms_for_nlp, series_of_interest, additional_stopwords,
                       term_fixes, df_jobs_raw, unique_titles_viz):
        '''
        Create visualizations for monograms and bigrams assoicated with the soft skill list.

        Parameters
        ----------
        n_grams : dataframe
            Contains the processed n_grams; sorted by count from highest to lowest.
        ds_soft_skill_terms : list
            Contains keywords pertaining to data science soft skills (e.g., 'collaborate', 'enthusiastic', etc.).
        terms_for_nlp : list
            List containing scraped and cleaned terms from the series of interest; created in the clean_for nlp function.
        series_of_interest : series
            A variable set in the main program, series_of_interest contains the targeted job listing data for NLP processing.
        additional_stopwords : list
            A list to capture all domain-specific stopwords and 'stop-lemma'.
        term_fixes : dictionary
            A dictionary for correcting misspelled, duplicated or consolidated terms in the series of interest.

        Returns
        -------
        None. Directly outputs visualizations.

        '''
        print('\nVisualizing Soft Skills...')
        
        def monograms_and_bigrams_by_count():
            '''
            Visualize the top n combined list of monograms and bigrams according to how many times they appear
            in the series of interest. Visualizes only the raw counts.

            Returns
            -------
            bigram_match_to_soft_list : list
                A list of bigrams in which each bigram has at least one term matching a term in the ds_soft_skill_terms list.

            '''          
            # subset the monograms that appear in the technical skills list
            mask_monogram = n_grams.grams.isin(ds_soft_skill_terms)
            monograms_df_sns = n_grams[mask_monogram]
            
            # generate bigrams from the full terms_for_nlp list
            n_gram_count = 2
            n_gram_range_start, n_gram_range_stop  = 0, 100
            bigrams = nlp_count_n_grams(terms_for_nlp, n_gram_count, n_gram_range_start, n_gram_range_stop)
            
            # subset the bigrams for which at least one term appears in the technical skills list
            bigram_match_to_soft_list = [x for x in bigrams.grams if any(b in x for b in ds_soft_skill_terms)]
            mask_bigram = bigrams.grams.isin(bigram_match_to_soft_list)
            bigrams_df_sns = bigrams[mask_bigram]
    
            # add the monograms and bigrams
            ngram_combined_sns = pd.concat([monograms_df_sns, bigrams_df_sns], axis=0, ignore_index=True)
    
            # identify noisy, duplicate or unhelpful terms and phrases
            # ngrams_to_silence = ['data', 'experience', 'business', 'science', 'year', 'ability', 'system'] # these from cred
            ngrams_to_silence = ['system'] 
            
            # exclude unwanted terms and phrases
            ngram_combined_sns = ngram_combined_sns[~ngram_combined_sns.grams.isin(ngrams_to_silence)].reset_index(drop=True)
    
            # create a horizontal barplot visualizing data science technical skills
            plt.figure(figsize=(7, 10))
            sns.set_style('dark')
            sns.set(font_scale = 1.8) 
                
            ax = sns.barplot(x='count',
                             y='grams',
                             data=ngram_combined_sns,
                             order=ngram_combined_sns.sort_values('count', ascending = False).grams[:25],
                             orient='h',
                             palette='mako_r') # crest, mako, 'mako_d, Blues_d, mako_r, ocean, gist_gray, gist_gray_r, icefire
            
            ax.set_title(textwrap.fill('Consider How Intensely Employers Care about Each Soft Skill', width=40),
                         fontsize=24,
                         loc='center')   
            ax.set(ylabel=None)
            ax.set_xlabel('Count', fontsize=18)
            
            plt.figtext(0.140, 0.010,
                        textwrap.fill(f'Data: {len(df)} Indeed job listings for {" ".join(str(x) for x in unique_titles_viz)} collected between {min(df.scrape_date)} and {max(df.scrape_date)}',
                                      width=70),
                        bbox=dict(facecolor='none', boxstyle='square', edgecolor='none', pad=0.2),
                        fontsize=14,
                        color='black',
                        fontweight='regular',
                        style='italic',
                        ha='left',
                        in_layout=True,
                        wrap=True) 
                       
            return bigram_match_to_soft_list


        def monograms_by_percentage(df_jobs_raw): 
            '''
            Visualize the soft skill monograms as a function of percentage of listings in which the monogram appears.
            This function makes first use of dataframes wherein each record is a job listing.

            Parameters
            ----------
            df_jobs_raw : dataframe
                A dataframe wherein each record is a unique listing, and each term in each listing is tokenized. df_jobs_raw is
                created in the visualize_n_grams function so that it can be used in subfunctions.

            Returns
            -------
            df_jobs_mono : dataframe
                A dataframe wherein each record is a job listing, and each column is a boolean flag for each
                monogram in the ds_soft_skill_terms list.  The final row and column each contain totals for their 
                respective job listing and soft skill term, respectively. The job_description field is dropped
                before the summations.

            '''
            # flag job listings if they contain the technical skill term (from stack question)
            df_jobs_mono = df_jobs_raw.copy()
            df_jobs_mono[ds_soft_skill_terms] = [[any(w==term for w in lst) for term in ds_soft_skill_terms] for lst in df_jobs_mono['job_description']]
            
            # calculate sum of all technical skill terms for both rows and columns
            df_jobs_mono = df_jobs_mono.drop('job_description', axis=1)
            df_jobs_mono.loc[:, 'total_mono_in_list'] = df_jobs_mono.sum(axis=1) # this does rows; need to plot these to filter out noisy/broken listings; can be used for the unicorn index
            df_jobs_mono.loc['total_mono', :] = df_jobs_mono.sum(axis=0) # this does columns; need to drop the job_description field
                 
            # drop all rows except the total row, transform columns and rows and rename the fields
            df_jobs_mono_sns = df_jobs_mono.drop(df_jobs_mono.index.to_list()[:-1], axis = 0).melt()
            df_jobs_mono_sns.rename(columns={'variable': 'ds_soft_skill_term','value': 'count'}, inplace=True)
            
            # calculate a percentages field
            df_jobs_mono_sns['percentage'] = [round(x / len(df_jobs_raw)*100, 2) for x in df_jobs_mono_sns['count']]
            df_jobs_mono_sns = df_jobs_mono_sns[df_jobs_mono_sns['ds_soft_skill_term'].str.contains('total')==False]
            
            # create a horizontal barplot visualizing data science technical skill monograms as a percentage of job listings
            plt.figure(figsize=(7, 10))
            sns.set_style('dark')
            sns.set(font_scale = 1.8) 
      
            ax = sns.barplot(x='percentage',
                             y='ds_soft_skill_term',
                             data=df_jobs_mono_sns,
                             order=df_jobs_mono_sns.sort_values('percentage', ascending = False).ds_soft_skill_term[:25],
                             orient='h',
                             palette='mako_r') # crest, mako, 'mako_d, Blues_d, mako_r, ocean, gist_gray, gist_gray_r, icefire
            
            
            ax.set_title(textwrap.fill('**For Soft Parsing: Monograms by Percentage', width=40), # original title: Percentage Key Terms for Data Scientist Credentials
                         fontsize=24,
                         loc='center')
            ax.set(ylabel=None)
            ax.set_xlabel('Percentage', fontsize=18)
            
            plt.figtext(0.140, 0.010,
                        textwrap.fill(f'Data: {len(df)} Indeed job listings for {" ".join(str(x) for x in unique_titles_viz)} collected between {min(df.scrape_date)} and {max(df.scrape_date)}',
                                      width=70),
                        bbox=dict(facecolor='none', boxstyle='square', edgecolor='none', pad=0.2),
                        fontsize=14,
                        color='black',
                        fontweight='regular',
                        style='italic',
                        ha='left',
                        in_layout=True,
                        wrap=True)   
            
            return df_jobs_mono


        def bigrams_by_percentage(df_jobs_raw, bigram_match_to_soft_list): 
            '''
            Visualize the soft skills bigrams as a function of percentage of listings in which the monogram appears.

            Parameters
            ----------
            df_jobs_raw : dataframe
                A dataframe wherein each record is a unique listing, and each term in each listing is tokenized. df_jobs_raw is
                created in the visualize_n_grams function so that it can be used in subfunctions.
            bigram_match_to_soft_list : list
                A list of bigrams in which each bigram has at least one term matching a term in the ds_tech_skill_terms list.

            Returns
            -------
            df_jobs_bigrams : dataframe
                A dataframe wherein each record is a job listing, and each column is a boolean flag for each
                bigram in the bigram_match_to_soft_list list.  The final row and column each contain totals for their 
                respective job listing and soft skill bigram, respectively. The job_description field is dropped
                before the summations.

            '''               
            # create df_jobs_bigrams from a copy of df_jobs_raw
            df_jobs_bigrams = df_jobs_raw.copy()
            
            # flag job listings if they contain the technical skill term (from stack question)
            def find_bigram_match_to_soft_list(data):
                output = np.zeros((data.shape[0], len(bigram_match_to_soft_list)), dtype=bool)
                for i, d in enumerate(data):
                    possible_bigrams = [' '.join(x) for x in list(nltk.bigrams(d)) + list(nltk.bigrams(d[::-1]))]
                    indices = np.where(np.isin(bigram_match_to_soft_list, list(set(bigram_match_to_soft_list).intersection(set(possible_bigrams)))))
                    output[i, indices] = True
                return list(output.T)
    
            output = find_bigram_match_to_soft_list(df_jobs_bigrams['job_description'].to_numpy())
            df_jobs_bigrams = df_jobs_bigrams.assign(**dict(zip(bigram_match_to_soft_list, output)))
            
            # identify and silence noisy, duplicate or unhelpful bigrams 
            bigrams_to_silence = ['machine learning', 'experience experience', 'collaborate collaborate']
            df_jobs_bigrams = df_jobs_bigrams.drop(columns=bigrams_to_silence, errors='ignore')
            
            # calculate sum of all technical skill terms for both rows and columns
            df_jobs_bigrams = df_jobs_bigrams.drop('job_description', axis=1)
            df_jobs_bigrams.loc[:, 'total_bigram_in_list'] = df_jobs_bigrams.sum(axis=1) # this does rows; need to plot these to filter out noisy/broken listings; can be used for the unicorn index
            df_jobs_bigrams.loc['total_bigram', :] = df_jobs_bigrams.sum(axis=0) # this does columns; need to drop the job_description field
            
            # drop all rows except the total row, transform columns and rows and rename the fields
            df_jobs_bigrams_sns = df_jobs_bigrams.drop(df_jobs_bigrams.index.to_list()[:-1], axis = 0).melt()
            df_jobs_bigrams_sns.rename(columns={'variable': 'ds_soft_term','value': 'count'}, inplace=True)
            
            # calculate a percentages field
            df_jobs_bigrams_sns['percentage'] = [round(x / len(df_jobs_raw)*100, 2) for x in df_jobs_bigrams_sns['count']]
            df_jobs_bigrams_sns = df_jobs_bigrams_sns[df_jobs_bigrams_sns['ds_soft_term'].str.contains('total')==False]
            
            # create a horizontal barplot visualizing data science credentials as a percentage of job listings
            plt.figure(figsize=(7, 10))
            sns.set_style('dark')
            sns.set(font_scale = 1.8)               
       
            ax = sns.barplot(x='percentage',
                             y='ds_soft_term',
                             data=df_jobs_bigrams_sns,
                             order=df_jobs_bigrams_sns.sort_values('percentage', ascending = False).ds_soft_term[:25],
                             orient='h',
                             palette='mako_r') # crest, mako, 'mako_d, Blues_d, mako_r, ocean, gist_gray, gist_gray_r, icefire
            
            ax.set_title(textwrap.fill('**For Soft Parsing: Bigrams by Percentage', width=40), # original title: Percentage Key Bigrams for Data Scientist Credentials
                         fontsize=24,
                         loc='center')
            ax.set(ylabel=None)
            ax.set_xlabel('Percentage', fontsize=18)
            
            plt.figtext(0.140, 0.010,
                        textwrap.fill(f'Data: {len(df)} Indeed job listings for {" ".join(str(x) for x in unique_titles_viz)} collected between {min(df.scrape_date)} and {max(df.scrape_date)}',
                                      width=70),
                        bbox=dict(facecolor='none', boxstyle='square', edgecolor='none', pad=0.2),
                        fontsize=14,
                        color='black',
                        fontweight='regular',
                        style='italic',
                        ha='left',
                        in_layout=True,
                        wrap=True)  
            
            return df_jobs_bigrams


        def monograms_and_bigrams_by_percentage(df_jobs_mono, df_jobs_bigrams):
            '''
            Visualize the combined soft skill monograms and bigrams as a function of percentage of listings in which
            either the monogram or bigram appears.
    
            Parameters
            ----------
            df_jobs_mono : dataframe
                A dataframe wherein each record is a job listing, and each column is a boolean flag for each
                monogram in the ds_soft_skill_terms list.  The final row and column each contain totals for their 
                respective job listing and soft skill term, respectively. The job_description field is dropped
                before the summations.
            df_jobs_bigrams : dataframe
                A dataframe wherein each record is a job listing, and each column is a boolean flag for each
                bigram in the bigram_match_to_soft_list list.  The final row and column each contain totals for their 
                respective job listing and soft skill bigram, respectively. The job_description field is dropped
                before the summations.
    
            Returns
            -------
            None. Directly outputs visualizations.
    
            '''
            # combine monograms and bigrams into a single dataframe
            df_jobs_combined = pd.concat([df_jobs_mono, df_jobs_bigrams], axis=1)
            
            # melt the dataframe, drop nan rows, rename the fields and drop the two 'total' rows
            df_jobs_combined_sns = df_jobs_combined.drop(df_jobs_combined.index.to_list()[:-2], axis = 0).melt()
            df_jobs_combined_sns = df_jobs_combined_sns[df_jobs_combined_sns['value'].notna()]
            df_jobs_combined_sns.rename(columns={'variable': 'ds_soft_term_phrase','value': 'count'}, inplace=True)
            df_jobs_combined_sns = df_jobs_combined_sns[~df_jobs_combined_sns.ds_soft_term_phrase.isin(['total_mono_in_list', 'total_bigram_in_list'])]
    
            # calculate a percentages field
            df_jobs_combined_sns['percentage'] = [round(x / len(df_jobs_raw)*100, 2) for x in df_jobs_combined_sns['count']]
      
            # visualize combined mongrams and bigrams
            plt.figure(figsize=(7, 10))
            sns.set_style('dark')
            sns.set(font_scale = 1.8)  
       
            ax = sns.barplot(x='percentage',
                             y='ds_soft_term_phrase',
                             data=df_jobs_combined_sns,
                             order=df_jobs_combined_sns.sort_values('percentage', ascending = False).ds_soft_term_phrase[:20],
                             orient='h',
                             palette='mako_r') # crest, mako, 'mako_d, Blues_d, mako_r, ocean, gist_gray, gist_gray_r, icefire
            
            ax.set_title(textwrap.fill('Focus Your Learning Time on High-Priority Soft Skills', width=30), # original title: Percentage Key Bigrams for Data Scientist Credentials
                         fontsize=24,
                         loc='center')
            ax.set(ylabel=None)
            ax.set_xlabel('Percentage', fontsize=18)
            
            plt.figtext(0.140, 0.010,
                        textwrap.fill(f'Data: {len(df)} Indeed job listings for {" ".join(str(x) for x in unique_titles_viz)} collected between {min(df.scrape_date)} and {max(df.scrape_date)}',
                                      width=70),
                        bbox=dict(facecolor='none', boxstyle='square', edgecolor='none', pad=0.2),
                        fontsize=14,
                        color='black',
                        fontweight='regular',
                        style='italic',
                        ha='left',
                        in_layout=True,
                        wrap=True)         
              
        # visualize soft skills by count
        bigram_match_to_soft_list = monograms_and_bigrams_by_count()
        
        # visualize soft skills by percentage
        df_jobs_mono = monograms_by_percentage(df_jobs_raw)
        df_jobs_bigrams = bigrams_by_percentage(df_jobs_raw, bigram_match_to_soft_list)
        monograms_and_bigrams_by_percentage(df_jobs_mono, df_jobs_bigrams)


    def visualize_professional(n_grams, ds_prof_skill_terms, terms_for_nlp, series_of_interest, 
                               additional_stopwords, term_fixes, df_jobs_raw, unique_titles_viz):
        '''
        Create visualizations for monograms and bigrams assoicated with the professional skill list.

        Parameters
        ----------
        n_grams : dataframe
            Contains the processed n_grams; sorted by count from highest to lowest.
        ds_prof_skill_terms : list
            Contains keywords pertaining to data science professional skills (e.g., 'management', 'thought-leadership', etc.).
        terms_for_nlp : list
            List containing scraped and cleaned terms from the series of interest; created in the clean_for nlp function.
        series_of_interest : series
            A variable set in the main program, series_of_interest contains the targeted job listing data for NLP processing.
        additional_stopwords : list
            A list to capture all domain-specific stopwords and 'stop-lemma'.
        term_fixes : dictionary
            A dictionary for correcting misspelled, duplicated or consolidated terms in the series of interest.

        Returns
        -------
        None. Directly outputs visualizations.

        '''
        print('\nVisualizing Professional Skills...')

        def monograms_and_bigrams_by_count():
            '''
            Visualize the top n combined list of monograms and bigrams according to how many times they appear
            in the series of interest. Visualizes only the raw counts.

            Returns
            -------
            bigram_match_to_prof_list : list
                A list of bigrams in which each bigram has at least one term matching a term in the ds_prof_skill_terms list.

            '''          
            # subset the monograms that appear in the technical skills list
            mask_monogram = n_grams.grams.isin(ds_prof_skill_terms)
            monograms_df_sns = n_grams[mask_monogram]
            
            # generate bigrams from the full terms_for_nlp list
            n_gram_count = 2
            n_gram_range_start, n_gram_range_stop  = 0, 100
            bigrams = nlp_count_n_grams(terms_for_nlp, n_gram_count, n_gram_range_start, n_gram_range_stop)
            
            # subset the bigrams for which at least one term appears in the technical skills list
            bigram_match_to_prof_list = [x for x in bigrams.grams if any(b in x for b in ds_prof_skill_terms)]
            mask_bigram = bigrams.grams.isin(bigram_match_to_prof_list)
            bigrams_df_sns = bigrams[mask_bigram]
    
            # add the monograms and bigrams
            ngram_combined_sns = pd.concat([monograms_df_sns, bigrams_df_sns], axis=0, ignore_index=True)
    
            # identify noisy, duplicate or unhelpful terms and phrases
            # ngrams_to_silence = ['data', 'experience', 'business', 'science', 'year', 'ability', 'system'] # these from cred
            ngrams_to_silence = ['system'] 
            
            # exclude unwanted terms and phrases
            ngram_combined_sns = ngram_combined_sns[~ngram_combined_sns.grams.isin(ngrams_to_silence)].reset_index(drop=True)
    
            # create a horizontal barplot visualizing data science technical skills
            plt.figure(figsize=(7, 10))
            sns.set_style('dark')
            sns.set(font_scale = 1.8) 
                
            ax = sns.barplot(x='count',
                             y='grams',
                             data=ngram_combined_sns,
                             order=ngram_combined_sns.sort_values('count', ascending = False).grams[:25],
                             orient='h',
                             palette='mako_r') # crest, mako, 'mako_d, Blues_d, mako_r, ocean, gist_gray, gist_gray_r, icefire
            
            ax.set_title(textwrap.fill('Consider How Intensely Employers Care about Each Professional Skill', width=40),
                         fontsize=24,
                         loc='center')   
            ax.set(ylabel=None)
            ax.set_xlabel('Count', fontsize=18)
            
            plt.figtext(0.140, 0.010,
                        textwrap.fill(f'Data: {len(df)} Indeed job listings for {" ".join(str(x) for x in unique_titles_viz)} collected between {min(df.scrape_date)} and {max(df.scrape_date)}',
                                      width=70),
                        bbox=dict(facecolor='none', boxstyle='square', edgecolor='none', pad=0.2),
                        fontsize=14,
                        color='black',
                        fontweight='regular',
                        style='italic',
                        ha='left',
                        in_layout=True,
                        wrap=True)  
                       
            return bigram_match_to_prof_list


        def monograms_by_percentage(df_jobs_raw): 
            '''
            Visualize the professional skill monograms as a function of percentage of listings in which the monogram appears.
            This function makes first use of dataframes wherein each record is a job listing.

            Parameters
            ----------
            df_jobs_raw : dataframe
                A dataframe wherein each record is a unique listing, and each term in each listing is tokenized. df_jobs_raw is
                created in the visualize_n_grams function so that it can be used in subfunctions.

            Returns
            -------
            df_jobs_mono : dataframe
                A dataframe wherein each record is a job listing, and each column is a boolean flag for each
                monogram in the ds_prof_skill_terms list.  The final row and column each contain totals for their 
                respective job listing and professional skill term, respectively. The job_description field is dropped
                before the summations.

            '''
            # flag job listings if they contain the technical skill term (from stack question)
            df_jobs_mono = df_jobs_raw.copy()
            df_jobs_mono[ds_prof_skill_terms] = [[any(w==term for w in lst) for term in ds_prof_skill_terms] for lst in df_jobs_mono['job_description']]
            
            # calculate sum of all technical skill terms for both rows and columns
            df_jobs_mono = df_jobs_mono.drop('job_description', axis=1)
            df_jobs_mono.loc[:, 'total_mono_in_list'] = df_jobs_mono.sum(axis=1) # this does rows; need to plot these to filter out noisy/broken listings; can be used for the unicorn index
            df_jobs_mono.loc['total_mono', :] = df_jobs_mono.sum(axis=0) # this does columns; need to drop the job_description field
                 
            # drop all rows except the total row, transform columns and rows and rename the fields
            df_jobs_mono_sns = df_jobs_mono.drop(df_jobs_mono.index.to_list()[:-1], axis = 0).melt()
            df_jobs_mono_sns.rename(columns={'variable': 'ds_prof_skill_term','value': 'count'}, inplace=True)
            
            # calculate a percentages field
            df_jobs_mono_sns['percentage'] = [round(x / len(df_jobs_raw)*100, 2) for x in df_jobs_mono_sns['count']]
            df_jobs_mono_sns = df_jobs_mono_sns[df_jobs_mono_sns['ds_prof_skill_term'].str.contains('total')==False]
            
            # create a horizontal barplot visualizing data science technical skill monograms as a percentage of job listings
            plt.figure(figsize=(7, 10))
            sns.set_style('dark')
            sns.set(font_scale = 1.8) 
      
            ax = sns.barplot(x='percentage',
                             y='ds_prof_skill_term',
                             data=df_jobs_mono_sns,
                             order=df_jobs_mono_sns.sort_values('percentage', ascending = False).ds_prof_skill_term[:25],
                             orient='h',
                             palette='mako_r') # crest, mako, 'mako_d, Blues_d, mako_r, ocean, gist_gray, gist_gray_r, icefire
                        
            ax.set_title(textwrap.fill('**For Prof Parsing: Monograms by Percentage', width=40), # original title: Percentage Key Terms for Data Scientist Credentials
                         fontsize=24,
                         loc='center')
            ax.set(ylabel=None)
            ax.set_xlabel('Percentage', fontsize=18)
            
            plt.figtext(0.140, 0.010,
                        textwrap.fill(f'Data: {len(df)} Indeed job listings for {" ".join(str(x) for x in unique_titles_viz)} collected between {min(df.scrape_date)} and {max(df.scrape_date)}',
                                      width=70),
                        bbox=dict(facecolor='none', boxstyle='square', edgecolor='none', pad=0.2),
                        fontsize=14,
                        color='black',
                        fontweight='regular',
                        style='italic',
                        ha='left',
                        in_layout=True,
                        wrap=True)   
            
            return df_jobs_mono


        def bigrams_by_percentage(df_jobs_raw, bigram_match_to_prof_list): 
            '''
            Visualize the professional skills bigrams as a function of percentage of listings in which the monogram appears.

            Parameters
            ----------
            df_jobs_raw : dataframe
                A dataframe wherein each record is a unique listing, and each term in each listing is tokenized. df_jobs_raw is
                created in the visualize_n_grams function so that it can be used in subfunctions.
            bigram_match_to_prof_list : list
                A list of bigrams in which each bigram has at least one term matching a term in the ds_prof_skill_terms list.

            Returns
            -------
            df_jobs_bigrams : dataframe
                A dataframe wherein each record is a job listing, and each column is a boolean flag for each
                bigram in the bigram_match_to_prof_list list.  The final row and column each contain totals for their 
                respective job listing and professional skill bigram, respectively. The job_description field is dropped
                before the summations.

            '''               
            # create df_jobs_bigrams from a copy of df_jobs_raw
            df_jobs_bigrams = df_jobs_raw.copy()
            
            # flag job listings if they contain the technical skill term (from stack question)
            def find_bigram_match_to_prof_list(data):
                output = np.zeros((data.shape[0], len(bigram_match_to_prof_list)), dtype=bool)
                for i, d in enumerate(data):
                    possible_bigrams = [' '.join(x) for x in list(nltk.bigrams(d)) + list(nltk.bigrams(d[::-1]))]
                    indices = np.where(np.isin(bigram_match_to_prof_list, list(set(bigram_match_to_prof_list).intersection(set(possible_bigrams)))))
                    output[i, indices] = True
                return list(output.T)
    
            output = find_bigram_match_to_prof_list(df_jobs_bigrams['job_description'].to_numpy())
            df_jobs_bigrams = df_jobs_bigrams.assign(**dict(zip(bigram_match_to_prof_list, output)))
            
            # identify and silence noisy, duplicate or unhelpful bigrams 
            bigrams_to_silence = ['machine learning', 'experience experience', 'collaborate collaborate']
            df_jobs_bigrams = df_jobs_bigrams.drop(columns=bigrams_to_silence, errors='ignore')
            
            # calculate sum of all technical skill terms for both rows and columns
            df_jobs_bigrams = df_jobs_bigrams.drop('job_description', axis=1)
            df_jobs_bigrams.loc[:, 'total_bigram_in_list'] = df_jobs_bigrams.sum(axis=1) # this does rows; need to plot these to filter out noisy/broken listings; can be used for the unicorn index
            df_jobs_bigrams.loc['total_bigram', :] = df_jobs_bigrams.sum(axis=0) # this does columns; need to drop the job_description field
            
            # drop all rows except the total row, transform columns and rows and rename the fields
            df_jobs_bigrams_sns = df_jobs_bigrams.drop(df_jobs_bigrams.index.to_list()[:-1], axis = 0).melt()
            df_jobs_bigrams_sns.rename(columns={'variable': 'ds_prof_term','value': 'count'}, inplace=True)
            
            # calculate a percentages field
            df_jobs_bigrams_sns['percentage'] = [round(x / len(df_jobs_raw)*100, 2) for x in df_jobs_bigrams_sns['count']]
            df_jobs_bigrams_sns = df_jobs_bigrams_sns[df_jobs_bigrams_sns['ds_prof_term'].str.contains('total')==False]
            
            # create a horizontal barplot visualizing data science credentials as a percentage of job listings
            plt.figure(figsize=(7, 10))
            sns.set_style('dark')
            sns.set(font_scale = 1.8)               
       
            ax = sns.barplot(x='percentage',
                             y='ds_prof_term',
                             data=df_jobs_bigrams_sns,
                             order=df_jobs_bigrams_sns.sort_values('percentage', ascending = False).ds_prof_term[:25],
                             orient='h',
                             palette='mako_r') # crest, mako, 'mako_d, Blues_d, mako_r, ocean, gist_gray, gist_gray_r, icefire
            
            ax.set_title(textwrap.fill('**For Prof Parsing: Bigrams by Percentage', width=40), # original title: Percentage Key Bigrams for Data Scientist Credentials
                         fontsize=24,
                         loc='center')
            ax.set(ylabel=None)
            ax.set_xlabel('Percentage', fontsize=18)
            
            plt.figtext(0.140, 0.010,
                        textwrap.fill(f'Data: {len(df)} Indeed job listings for {" ".join(str(x) for x in unique_titles_viz)} collected between {min(df.scrape_date)} and {max(df.scrape_date)}',
                                      width=70),
                        bbox=dict(facecolor='none', boxstyle='square', edgecolor='none', pad=0.2),
                        fontsize=14,
                        color='black',
                        fontweight='regular',
                        style='italic',
                        ha='left',
                        in_layout=True,
                        wrap=True)  
            
            return df_jobs_bigrams


        def monograms_and_bigrams_by_percentage(df_jobs_mono, df_jobs_bigrams):
            '''
            Visualize the combined professional skill monograms and bigrams as a function of percentage of listings in which
            either the monogram or bigram appears.
    
            Parameters
            ----------
            df_jobs_mono : dataframe
                A dataframe wherein each record is a job listing, and each column is a boolean flag for each
                monogram in the ds_tech_skill_terms list.  The final row and column each contain totals for their 
                respective job listing and professional skill term, respectively. The job_description field is dropped
                before the summations.
            df_jobs_bigrams : dataframe
                A dataframe wherein each record is a job listing, and each column is a boolean flag for each
                bigram in the bigram_match_to_prof_list list.  The final row and column each contain totals for their 
                respective job listing and professional skill bigram, respectively. The job_description field is dropped
                before the summations.
    
            Returns
            -------
            None. Directly outputs visualizations.
    
            '''
            # combine monograms and bigrams into a single dataframe
            df_jobs_combined = pd.concat([df_jobs_mono, df_jobs_bigrams], axis=1)
            
            # melt the dataframe, drop nan rows, rename the fields and drop the two 'total' rows
            df_jobs_combined_sns = df_jobs_combined.drop(df_jobs_combined.index.to_list()[:-2], axis = 0).melt()
            df_jobs_combined_sns = df_jobs_combined_sns[df_jobs_combined_sns['value'].notna()]
            df_jobs_combined_sns.rename(columns={'variable': 'ds_prof_term_phrase','value': 'count'}, inplace=True)
            df_jobs_combined_sns = df_jobs_combined_sns[~df_jobs_combined_sns.ds_prof_term_phrase.isin(['total_mono_in_list', 'total_bigram_in_list'])]
    
            # calculate a percentages field
            df_jobs_combined_sns['percentage'] = [round(x / len(df_jobs_raw)*100, 2) for x in df_jobs_combined_sns['count']]
      
            # visualize combined mongrams and bigrams
            plt.figure(figsize=(7, 10))
            sns.set_style('dark')
            sns.set(font_scale = 1.8)  
       
            ax = sns.barplot(x='percentage',
                             y='ds_prof_term_phrase',
                             data=df_jobs_combined_sns,
                             order=df_jobs_combined_sns.sort_values('percentage', ascending = False).ds_prof_term_phrase[:20],
                             orient='h',
                             palette='mako_r') # crest, mako, 'mako_d, Blues_d, mako_r, ocean, gist_gray, gist_gray_r, icefire
            
            ax.set_title(textwrap.fill('Focus Your Learning Time on High-Priority Professional Skills', width=33), # original title: Percentage Key Bigrams for Data Scientist Credentials
                         fontsize=24,
                         loc='center')
            ax.set(ylabel=None)
            ax.set_xlabel('Percentage', fontsize=18)
            
            plt.figtext(0.140, 0.010,
                        textwrap.fill(f'Data: {len(df)} Indeed job listings for {" ".join(str(x) for x in unique_titles_viz)} collected between {min(df.scrape_date)} and {max(df.scrape_date)}',
                                      width=70),
                        bbox=dict(facecolor='none', boxstyle='square', edgecolor='none', pad=0.2),
                        fontsize=14,
                        color='black',
                        fontweight='regular',
                        style='italic',
                        ha='left',
                        in_layout=True,
                        wrap=True)  

        # visualize professional skills by count
        bigram_match_to_prof_list = monograms_and_bigrams_by_count()
        
        # visualize professional skills by percentage
        df_jobs_mono = monograms_by_percentage(df_jobs_raw)
        df_jobs_bigrams = bigrams_by_percentage(df_jobs_raw, bigram_match_to_prof_list)
        monograms_and_bigrams_by_percentage(df_jobs_mono, df_jobs_bigrams)

    
    def prepare_job_titles_for_viz(df):
        '''
        Identify the unique job titles inherent in the raw Indeed dataframe, df. Convert the unique abbreviations
        to their full names (e.g., 'ds' to 'data scientist') and prepare the list of full names for automatic
        insertion into visualizations.

        Parameters
        ----------
        df : dataframe
            Contains the raw concatenated Indeed csvs. Created in the clean_raw_csv function.

        Returns
        -------
        unique_titles_viz : list
            Fully conditioned list of unique job titles with inserted quotation marks and conjunctions (e.g., 'and').

        '''
        # create a dictionary for job title and abbreviation mapping
        job_title_map = {'ds': 'data scientist',
                         'ml': 'machine learning',
                         'da': 'data analyst',
                         'ai': 'artificial intelligence',
                         'dd': 'data science director',
                         'de': 'data engineer',
                         'ce': 'cloud engineer',
                         'se': 'software engineer',
                         'ba': 'business analyst'}

        # capture the unique job title abbreviations from df
        unique_titles_raw = list(df.scrape_job_title.unique()) # maybe intersperse a comma and and and in between titles
             
        # convert abbreviations to full job titles
        unique_titles = [job_title_map.get(title, title) for title in unique_titles_raw]
        
        # append quotation marks to every element
        unique_titles = ['"' + title for title in unique_titles]
        append_quotes = '"'
        unique_titles = (pd.Series(unique_titles) + append_quotes).tolist()
        
        # insert 'and' in between job titles
        unique_titles_viz = unique_titles[:]
        unique_titles_viz.insert(1, 'and')
    
        # print test
        # print(f'This is a test of the job titles, which are {" ".join(str(x) for x in unique_titles_viz)}')
        
        return unique_titles_viz


    unique_titles_viz = prepare_job_titles_for_viz(df)
    
    # create a clean dataframe where each record is a unique listing, and each term is tokenized
    df_jobs_raw = clean_listings_for_nlp(series_of_interest, additional_stopwords, term_fixes) 
    
    # visualize all mongrams regardless of skillset
    visualize_all_monograms(n_grams, unique_titles_viz)
    
    # visualize credentials, technical skills, soft skills and professional skills
    visualize_credentials(n_grams, ds_cred_terms, terms_for_nlp, series_of_interest, additional_stopwords,
                          term_fixes, df_jobs_raw, unique_titles_viz) 
    
    visualize_technicals(n_grams, ds_tech_skill_terms, terms_for_nlp, series_of_interest, additional_stopwords,
                         term_fixes, df_jobs_raw, unique_titles_viz)
    
    visualize_soft(n_grams, ds_soft_skill_terms, terms_for_nlp, series_of_interest, additional_stopwords,
                   term_fixes, df_jobs_raw, unique_titles_viz)
    
    visualize_professional(n_grams, ds_prof_skill_terms, terms_for_nlp, series_of_interest, additional_stopwords,
                           term_fixes, df_jobs_raw, unique_titles_viz)
    
    return df_jobs_raw, unique_titles_viz


def visualize_word_clouds(terms_for_nlp, series_of_interest):
    '''
    Generate masked and unmasked word clouds from the processed terms extracted from the 
    series of interest (e.g., job_description, company, etc.)

    Parameters
    ----------
    terms_for_nlp : list
        List containing scraped and cleaned terms from the series of interest; created
        in the clean_for nlp function.

    Returns
    -------
    None. Directly outputs and saves visualizations as pngs.

    '''
    print('\nCreating word clouds...')
    
    # convert the terms_for_nlp list into a string, which is what WordCloud expects
    word_cloud_terms = ' '.join(terms_for_nlp)
       
    # create a WordCloud object and optimize the terms for display; tune the Dunning collocation_threshold
    # to increase or decrease bigram frequency (low threshold=more bigrams)
    
    word_cloud = WordCloud(max_font_size=50,
                           max_words=100,
                           background_color='lightgray',      # whitesmoke, gainsboro, lightgray, silver
                           colormap='crest',                  # mako, crest
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
                                   colormap='crest',                  # mako, crest
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


def visualize_subtopic(df, df_jobs_raw, terms_for_nlp, subtopic_list, unique_titles_viz, viz_title):
    '''
    Visualize counts and percentages of monograms for subtopics of interest.

    Parameters
    ----------
    subtopic_list : list
        List containing monograms of interest for the subtopic (e.g., 'pandas' amd 'numpy' for python libraries).
    viz_title : string
        String containing the title for the outputted visualization.

    Returns
    -------
    None. Directly outputs visualizations.

    ''' 
    # generate monograms from the full terms_for_nlp list
    n_gram_count = 1
    n_gram_range_start, n_gram_range_stop  = 0, int((len(terms_for_nlp) / 2))
    monograms = nlp_count_n_grams(terms_for_nlp, n_gram_count, n_gram_range_start, n_gram_range_stop)    

    # subset the monograms that appear in the subtopic list
    mask_monogram = monograms.grams.isin(subtopic_list)
    monograms_df_sns = monograms[mask_monogram]  

    # create a horizontal barplot visualizing data science skills in the subtopic list - by count
    plt.figure(figsize=(7, 10))
    sns.set_style('dark')
    sns.set(font_scale = 1.8) 
        
    ax = sns.barplot(x='count',
                     y='grams',
                     data=monograms_df_sns,
                     order=monograms_df_sns.sort_values('count', ascending = False).grams[:25],
                     orient='h',
                     palette='mako_r') # crest, mako, 'mako_d, Blues_d, mako_r, ocean, gist_gray, gist_gray_r, icefire
    
    ax.set_title(textwrap.fill(viz_title, width=40),
                 fontsize=24,
                 loc='center')   
    ax.set(ylabel=None)
    ax.set_xlabel('Count', fontsize=18)
    
    plt.figtext(0.140, 0.010,
                textwrap.fill(f'Data: {len(df)} Indeed job listings for {" ".join(str(x) for x in unique_titles_viz)} collected between {min(df.scrape_date)} and {max(df.scrape_date)}',
                              width=70),
                bbox=dict(facecolor='none', boxstyle='square', edgecolor='none', pad=0.2),
                fontsize=14,
                color='black',
                fontweight='regular',
                style='italic',
                ha='left',
                in_layout=True,
                wrap=True) 
     
    # create a horizontal barplot visualizing data science skills in the subtopic list - by percentage
    df_jobs_mono = df_jobs_raw.copy()
    df_jobs_mono[subtopic_list] = [[any(w==term for w in lst) for term in subtopic_list] for lst in df_jobs_mono['job_description']]
    
    # calculate sum of all credential terms for both rows and columns
    df_jobs_mono = df_jobs_mono.drop('job_description', axis=1)
    df_jobs_mono.loc[:, 'total_mono_in_list'] = df_jobs_mono.sum(axis=1) # this does rows; need to plot these to filter out noisy/broken listings; can be used for the unicorn index
    df_jobs_mono.loc['total_mono', :] = df_jobs_mono.sum(axis=0) # this does columns; need to drop the job_description field
         
    # drop all rows except the total row, transform columns and rows and rename the fields
    df_jobs_mono_sns = df_jobs_mono.drop(df_jobs_mono.index.to_list()[:-1], axis = 0).melt()
    df_jobs_mono_sns.rename(columns={'variable': 'subtopic_term','value': 'count'}, inplace=True)
    
    # calculate a percentages field; will need to divide by len(df_jobs) * 100
    df_jobs_mono_sns['percentage'] = [round(x / len(df_jobs_raw)*100, 2) for x in df_jobs_mono_sns['count']]
    df_jobs_mono_sns = df_jobs_mono_sns[df_jobs_mono_sns['subtopic_term'].str.contains('total')==False]      
    
    # create a horizontal barplot visualizing data science credential monograms as a percentage of job listings
    plt.figure(figsize=(7, 10))
    sns.set_style('dark')
    sns.set(font_scale = 1.8)            
   
    ax = sns.barplot(x='percentage',
                     y='subtopic_term',
                     data=df_jobs_mono_sns,
                     order=df_jobs_mono_sns.sort_values('percentage', ascending = False).subtopic_term[:25],
                     orient='h',
                     palette='mako_r') # crest, mako, 'mako_d, Blues_d, mako_r, ocean, gist_gray, gist_gray_r, icefire
    
    ax.set_title(textwrap.fill(viz_title, width=40), 
                 fontsize=24,
                 loc='center')
    ax.set(ylabel=None)
    ax.set_xlabel('Percentage', fontsize=18)
    
    plt.figtext(0.140, 0.010,
                textwrap.fill(f'Data: {len(df)} Indeed job listings for {" ".join(str(x) for x in unique_titles_viz)} collected between {min(df.scrape_date)} and {max(df.scrape_date)}',
                              width=70),
                bbox=dict(facecolor='none', boxstyle='square', edgecolor='none', pad=0.2),
                fontsize=14,
                color='black',
                fontweight='regular',
                style='italic',
                ha='left',
                in_layout=True,
                wrap=True)    


def nlp_skill_lists(additional_stopwords):
    '''   
    Generate lists of keywords for each job title's: credentials, technical skills, soft skills and professional skills.
    These lists are used by other functions to filter the job_description field, visualize key terms, etc.

    Parameters
    ----------
    additional_stopwords : list
        A list to capture all domain-specific stopwords and 'stop-lemma'. Created in the clean_terms_for_nlp function.

    Returns
    -------
    ds_cred_terms : list
        Contains keywords pertaining to data science credentials (e.g., 'bachelors', 'experience', etc.).
    ds_tech_skill_terms : list
        Contains keywords pertaining to data science technical skills (e.g., 'python', 'tableau', etc.).
    ds_soft_skill_terms : list
        Contains keywords pertaining to data science soft skills (e.g., 'collaboration', 'self-motivation', etc.).
    ds_prof_skill_terms : list
        Contains keywords pertaining to data science processional skills (e.g., 'client interaction', 'leadership', etc.).
    ds_skills_combined : list
        Contains combination of all keywords from ds_cred_terms, ds_tech_skill_terms, ds_soft_skill_terms, 
        and ds_prof_skill_terms.
    subtopic_python : list
        Contains keywords pertaining only to python-specific tech (e.g., 'pandas', 'numpy', 'anaconda', etc.)
    '''
    
    # establish credential and skill lists for nlp filtering
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
                     '20',
                     'advanced',
                     'administration',
                     'accredited',
                     'aerospace',
                     'analyst',
                     'analytics',
                     'associate',
                     'astronomy',
                     'bachelors',
                     'bioinformatics',
                     'bioscience',
                     'biostatistics',
                     'biotech',
                     'bootcamp',
                     'business',
                     'capability',
                     'certification',
                     'chemistry',
                     'cism',
                     'clearance',
                     'college',
                     'computer',
                     'coursework',
                     'credential',
                     'cryptography',
                     'ctap',
                     'data',
                     'degree',
                     'economics',
                     'econometrics',
                     'education',
                     'electrical',
                     'engineering',
                     'equivalent',
                     'experience',
                     'expert',
                     'field',
                     'ged',
                     'genomics',
                     'geophysical',
                     'geospatial',
                     'gpa',
                     'graduate',
                     'industry',
                     'informatics',
                     'knowledge',
                     'linkedin',
                     'major',
                     'masters',
                     'mathematics',
                     'mba',
                     'mpp',
                     'operations', 
                     'peer-reviewed', 
                     'phd',
                     'physics',
                     'physical',
                     'polygraph',
                     'portfolio',
                     'practical',
                     'professional',
                     'proficient',
                     'proven',
                     'publication',
                     'quantitative',
                     'relevant',
                     'research',
                     'sci',
                     'science',
                     'security',
                     'software',
                     'spectrometry',
                     'ssbi',
                     'statistics',
                     'stem',
                     'system', 
                     'technical',
                     'top secret',
                     'training',
                     'understand',
                     'year']  

    subtopic_agile = ['agile', 'backlog', 'kanban', 'mvp', 'roadmap', 'scrum', 'sprint', 'waterfall']

    subtopic_aws = ['aws', 'amazon', 'athena', 'aurora', 'cloudhsm', 'cloudtrail', 'codebuild', 'codedeploy', 'codepipeline',
                    'codestar', 'dms', 'dynamodb', 'ebs', 'ec2', 'elasticache', 'elasticsearch', 'emr',
                    'glacier', 'glue', 'inspector', 'kinesis', 'kms', 'lambda', 'lex', 'macie', 'mapreduce', 'polly', 
                    'presto', 'quicksight', 'rds', 'redshift', 'rekognition', 's3', 'sagemaker', 'sdk', 'snowball',
                    'sqs', 'vpc', 'xray']
    
    subtopic_big_data = ['apache', 'flume', 'hadoop', 'hdfs', 'hive', 'hiveql', 'oozie', 'mapreduce', 'pig', 'presto',
                         'sqoop', 'spark', 'yarn',]   
    
    subtopic_cloud = ['amazon', 'alibaba', 'aurora', 'aws', 'azure', 'cloud', 'cloudera', 'cosmos', 'databricks',
                      'digitalocean', 'gcp', 'google', 'h2o', 'informatica', 'nifi', 'oracle',
                      'redhat', 'sap', 'splunk', 'vertica', 'vmware', 'zeromq'] ### might need to deal with 'google cloud' bigram
    
    subtopic_containers = ['artifactory', 'buildah', 'buildkit', 'container', 'containerd', 'crun', 'dive',
                           'docker', 'hyperv', 'kaniko', 'kubeflow', 'kubernetes', 'lxc', 'lxd', 'mesos',
                           'openshift', 'openvz', 'orchestration', 'podman', 'rancher', 'rkt', 'runc', 'skopeo',
                           'vagrant', 'virtualbox', 'windock', 'zerovm'] ### 'dive' might have to be in context of cloud tech

    subtopic_databases = ['access', 'accumulo', 'cassandra', 'cosmos', 'couchbase', 'db2', 'dynamodb',
                          'elasticsearch', 'flockdb', 'hbase', 'hibari', 'mldb', 'mongodb', 'mssql',
                          'mysql', 'neo4j', 'nosql', 'orientdb', 'postgresql', 'rabbitmq', 'rdbms',
                          'redis', 'relational', 'riak', 'solr', 'terrstore'] 

    subtopic_datatypes = ['avro', 'continuous', 'csv', 'discrete', 'excel', 'hdf5', 'html', 'jpeg',
                          'json', 'matlab', 'netcdf', 'onnx', 'orc', 'parquet', 'pb', 'pdf', 'petastorm',
                          'pickle', 'pmml', 'png', 'qualitative', 'quantitative',
                          'sas', 'sql', 'stata', 'structured', 'text', 'tfrecords', 'time series',
                          'trajectory', 'txt', 'unstructured', 'xlsx', 'xml', 'yaml', 'zip'] 

    subtopic_dl_algorithms = ['algorithm', 'autoencoder', 'boltzmann', 'convolutional', 'dbm', 'dbn', 'gan',
                              'hopfield', 'lvq', 'lstm', 'mlp', 'neural network', 'perceptron',
                              'rbfn', 'rbm', 'rbn', 'rnn', 'som']
     
    subtopic_dl_frameworks = ['blocks', 'caffe', 'chainer', 'cntk ', 'dgl', 'dl4j', 'flux', 'gluon', 'h2o',
                              'keras', 'lasagne', 'mxnet', 'paddlepaddle', 'pytorch', 'singa', 'sonnet',
                              'spark ml', 'tensorflow', 'tfx', 'theano']

    subtopic_dl_supporting = ['automl', 'ensemble', 'inference', 'lightgbm', 'neuron', 'nvidia', 'onnx', 'pipeline',
                              'pla', 'tensorrt'] 
    
    subtopic_excel = ['excel', 'pivot', 'spreadsheet', 'vlookup',]
    
    subtopic_geospatial = ['arcgis', 'cad', 'qgis',]

    subtopic_ide = ['anaconda', 'atom', 'dataspell', 'eclipse', 'emacs', 'gedit', 'jupyter', 'notepad',
                    'nteract', 'pycharm', 'pydev', 'rstudio', 'rodeo', 'spyder', 'sublime', 'thonny', 
                    'vim', 'visual studio']

    subtopic_it_and_web = ['ansible', 'aspnet', 'executable', 'flask', 'fortify', 'jenkins',
                           'nginx', 'reactjs', 'rest', 'twistlock']
    
    subtopic_javascript = ['ajax', 'd3js', 'jquery', 'reactjs',] 

    subtopic_languages = ['assembly', 'awk', 'bash', 'c', 'css', 'dax', 'fortran', 'go', 'golang', 'graphql', 'groovy',
                         'hiveql', 'hpcml', 'html', 'java', 'javascript', 'julia', 'kotlin', 'lisp',
                         'matlab', 'nodejs', 'opencl', 'perl', 'php', 'pig', 'plsql', 'python', 'r', 'ruby', 'rust',
                         'sas', 'scala', 'shell', 'sparql', 'sql', 'swift', 'torch', 'typescript',
                         'wolfram', 'vba', 'xml',]

    subtopic_linux = ['bash', 'centos', 'cli', 'cuda', 'debian', 'fedora', 'linux', 'mint', 'nvidia', 'openshift', 'redhat',
                      'shell', 'ubuntu', 'unix',] # rapids, but have to find

    subtopic_mathematics = ['algebra',  'bayes', 'calculus', 'differential', 'discrete math', 'geometry', 'graph theory', 
                            'information theory', 'linear algebra', 'mathematics', 'multivariate', 'probability',
                            'statistics']  #### !!!! DECONFLICT WITH MATH LIST RIGHT BELOW

    subtopic_math = ['algebra', 'calculus', 'discrete math', 'dsp', 'geometry', 'graph theory', 'information theory',
                     'linear algebra', 'mathematics', 'multivariate', 'probability', 'statistics',]
    
########## SUPERVISED ##########     
    subtopic_ml_classification = ['bayes', 'classification', 'decision tree', 'decision stump ',
                                  'discriminant analysis', 'gradient boosting', 'knn', 'lda', 'logistic regression',
                                  'naive bayes', 'neural network', 'nusvc', 'oner', 'pla', 'qda', 'random forest',
                                  'sgd', 'svc', 'svm', 'zeror']
    
    subtopic_ml_regression = ['arima', 'decision tree', 'ensemble', 'glm', 'gpr', 'gradint boosting', 'linear regression',
                              'loess', 'mars', 'neural network', 'ols', 'random forest', 'regression', 'svr']

    subtopic_ml_supervised = ['supervised', 'classification', 'regression'] + subtopic_ml_classification + subtopic_ml_regression

########## SEMI-SUPERVISED ########## 
    subtopic_ml_semisupervised = ['gan']


########## UNSUPERVISED ########## 
    subtopic_ml_unsupervised = ['apriori', 'clustering',  'cmeans', 'gmm', 'kmeans', 'markov',
                                'neural network', 'pca', 'unsupervised'] 
    
    subtopic_ml_association = ['apriori', 'eclat']

    subtopic_ml_clustering = ['clustering',  'cmeans', 'dbscan', 'expectation maximization',
                              'gmm', 'hierarchical', 'kmeans', 'kmedians', 'kmodes',
                              'mean shift', 'markov', 'neural network', 'spectral clustering']  # recommender systems
    
    subtopic_ml_dimen_reduct = ['autoencoder', 'backward feature', 'dimensionality reduction', 'factor analysis',
                                'forward feature', 'ica', 'isomap', 'lda', 'lle', 'mds', 'pca', 'plsr', 'rda', 'sammon',
                                'subset selction', 'svd', 'tsne']
    
    subtopic_ml_unsupervised = ['unsupervised'] + subtopic_ml_association + subtopic_ml_clustering + subtopic_ml_dimen_reduct




    subtopic_nlp = ['allennlp', 'asr', 'corenlp', 'corpus', 'crf', 'flair', 'gensim', 'gpt', 'kaldi',
                    'lda', 'nlp', 'nltk', 'pattern', 'polyglot', 'pynlpl', 'rasa', 'sentiment',
                    'sklearn', 'spacy', 'speechtotext', 'spss', 'textblob', 'transformers', 'translation', 'word2vec']
    
    subtopic_other = ['ivr', 'jira', 'peoplesoft', 'powershell', 'windows']
    
    subtopic_platforms = ['alteryx', 'cplex', 'dataminr', 'datarobot', 'delfi', 'fivetran', 'ide', 'knime',
                          'mathematica', 'mlflow', 'nvivo', 'snaplogic', 'snowflake', 'sorcero', 'weka',]
    
    subtopic_problem_types = ['anomaly detection', 'classification', 'clustering', 'identification', 'localization',
                              'imputation', 'modeling',  'optimization', 'pattern', 'regression', 'regularization',
                              'segmentation', 'transformation',]  

    subtopic_python = ['allennlp', 'anaconda', 'beautifulsoup', 'bokeh', 'caffe', 'corenlp', 'dash', 'dask', 'dgl', 
                       'django', 'fastapi', 'flask', 'gensim', 'ipython', 'jupyter', 'keras', 'luigi', 'mahotas',
                       'matplotlib', 'mlpack', 'mxnet', 'nltk', 'numpy', 'opencv', 'optimus', 'pandas', 'petastorm', 'pillow',
                       'plotly', 'polyglot', 'pycaret', 'pycharm', 'pydot', 'pynlpl', 'pyspark', 'pytest',
                       'pytorch', 'pyunit', 'rasa', 'requests', 'scipy', 'scrapy', 'sdk',
                       'seaborn', 'selenium', 'simpleitk', 'skimage', 'sklearn', 'sonnet', 'spacy', 
                       'sqlalchemy', 'statsmodels', 'tensorflow', 'textblob', 'theano', 'word2vec',
                       'xgboost', 'zookeeper'] 
    
    subtopic_r = ['bioconductor', 'caret', 'dataexplorer', 'datatable', 'dplyr', 'e1071', 'esquisse' , 'ggplot',
                  'janitor', 'kernlab', 'knitr', 'lattice', 'lubridate', 'mboost', 'mlr3', 'plotly', 'purr',
                  'quanteda', 'rcrawler', 'readr', 'readxl', 'rio', 'rmarkdown', 'shiny', 'stringr', 'superml',
                  'tidyquant', 'tidyr', 'tidyverse', 'tidyxl', 'vroom', 'xgboost',] 
    
    subtopic_sql = ['mssql', 'mysql', 'nosql', 'postgresql', 'presto', 'sql', 'sqlite', 'ssis',] 

    subtopic_viz = ['bokeh', 'chartblocks', 'cognos', 'd3js', 'dashboard',  'datawrapper',  'domo',
                    'dundas', 'echarts', 'excel', 'finereport', 'fusioncharts', 'ggplot', 'grafana',
                    'highcharts', 'infogram', 'interactive', 'kibana', 'kizan', 'klipfolio', 'leaflet',
                    'looker', 'matplotlib', 'palantir', 'periscope', 'plotly', 'polymaps', 'power bi', 'pydot',
                    'qlik', 'seaborn', 'sigmajs', 'sisense', 'spotfire', 'tableau', 'tibco', 'vega',
                    'visio', 'visualization', 'watson', 'zoho'] 
   
    subtopic_version_control = ['bazaar', 'bitbucket', 'cvs', 'delta lake', 'dolt', 'dvc', 'git', 'lakefs', 'mercurial', 
                                'monotone', 'neptune', 'pachyderm', 'svn', 'tfs', 'vsts' ]  ### need to add in other git elements like GitHub, Git LFS, GitLab, etc.



########## REINFORCEMENT LEARNING ########## 
    subtopic_ml_reinforcement = ['multiarmed bandit', 'reinforcement',]
    

########## ANOMALY DETECTION ########## 
    subtopic_ml_anomaly = ['anomaly detection', 'isolation forest', 'lof', 'mcd', 'pca', 'svm']


########## I AM NOT SURE YET; COULD BE OVERLAPS ##########
    subtopic_ml_ensemble = ['adaboost', 'bagging', 'bootstrap', 'catboost', 'gbm', 'gbrt',
                            'gradient boosting', 'random forest', 'voting classifier', 'xgboost']
    
    subtopic_ml_regularization = ['elastic net', 'lasso', 'lars', 'regularization', 'ridge']
    
    subtopic_ml_recommendation = ['recommender', 'recommendation',]

    subtopic_ml_stats = ['anova', 'correlation', 'gaussian', 'gradient descent', 'inferential', 'linear',
                         'loss', 'maximum likelihood', 'monte carlo', 'nonlinear', 'nonparametric',
                         'normalize', 'outlier', 'parametric', 'pearson', 'randomization', 'rms',
                         'sampling', 'simulation', 'skewness', 'stochastic', 'univariate',]
    
    subtopic_ml_optimization= ['optimization', 'pso']

  

####### !!!!!!!! WORKING HERE: organize the subtopic lists and create aggregation lists of key subtopics   
    subtopic_holding_tank = ['logic', 'matrix', 'nearest', 'neighbor', 'ontology', 'reasoning', 'validation', 'vector',]    
    
      
    ds_tech_skill_terms = ['ab',
                           'access',
                           'accumulo',
                           'accurate', 
                           'activation',
                           'adaboost',
                           'adobe', 
                           'adopter',
                           'adversarial',
                           'aggregation',
                           'ai',
                           'airflow',
                           'advanced',
                           'algebra',
                           'algorithm',
                           'alibaba', 
                           'allennlp',
                           'alteryx',
                           'amazon',
                           'anaconda',
                           'analytics',
                         'angularjs',
                         'anomaly', 
                         'anova',
                         'ansible',
                         'ansys', 
                         'apache', 
                         'api',
                         'application',
                         'applied', 
                         'approach',
                         'apriori',
                         'arcgis',
                         'architecture',
                         'arima',
                         'artifactory',
                         'artificial',
                         'aspnet',
                         'assembly',
                         'assumption',
                         'asr',
                         'athena',
                         'atom',
                         'augmentation',
                         'aurora',
                         'autoencoder', 
                         'automate',
                         'auto',
                         'automl',
                         'avro',
                         'awk', 
                         'aws', 
                         'azure',
                         'bandit',
                         'bash',
                         'batch',
                         'bayes',
                         'bazaar',
                         'beautifulsoup',
                         'behavioral',
                         'bi', 
                         'big',
                         'bioconductor',
                         'bitbucket',
                         'blockchain',
                         'blocks',
                         'bokeh',
                         'bootstrap',
                         'breakthrough',
                         'build',
                         'buildah',
                         'buildkit',
                         'c',
                         'cad',
                         'caffe',
                         'calculus',
                         'caret',
                         'carlo',
                         'cassandra',
                         'categorize',
                         'causal', 
                         'center',
                         'centos',
                         'cicd',
                         'chain',
                         'chainer',
                         'chart',
                         'chartblocks',
                         'chatbot',
                         'classification',
                         'cleaning',
                         'cli',
                         'cloud',
                         'cloudera',
                         'cloudhsm',
                         'clustering',
                         'cloudtrail',
                         'cntk ',
                         'code', 
                         'codebuild',
                         'codedeploy',
                         'codepipeline',
                         'codestar',
                         'collection',
                         'cognitive',
                         'cognos',
                         'compression',
                         'compile',
                         'complexity',
                         'compute',
                         'computer',
                         'computing',
                         'concept',
                         'conceptual',
                         'confluence',
                         'constraint',
                         'container',
                         'containerd',
                         'continuous',
                         'control',
                         'convex',
                         'convolutional',
                         'corenlp',
                         'corpus',
                         'correlation',
                         'cosmos',
                         'costing',
                         'couchbase',
                         'cplex',
                         'crf',
                         'crm',
                         'crowdstrike',
                         'crun',
                         'css',
                         'csv',
                         'cuda',
                         'curate',
                         'cutting',
                         'cvs',
                         'cybersecurity',
                         'd3js',
                         'dash',
                         'dashboard',
                         'dask',
                         'data',
                         'databricks',
                         'dataexplorer',
                         'database',
                         'datamining',
                         'dataminr',
                         'datarobot',
                         'dataspell',
                         'datatable',
                         'datawrapper',
                         'data-driven',
                         'dax', 
                         'db2',
                         'dbm',
                         'dbn',
                         'decomposition',  
                         'deep',
                         'delfi',
                         'deploy',
                         'design',
                         'detection',
                         'development',
                         'devops',
                         'dgl',
                         'dictionary',
                         'differential',
                         'dimensional',
                         'dimensionality', 
                         'discrete',
                         'disparate',
                         'distributed',
                         'dive',
                         'dl4j',
                         'dlp',
                         'dms',
                         'docker',
                         'dolt',
                         'domo',
                         'django',
                         'dplyr',
                         'dsp',
                         'dundas',
                         'dvc',
                         'dynamodb',
                         'e1071',
                         'early', 
                         'ebs',
                         'ec2',
                         'echarts',
                         'eclipse',
                         'ecosystem',
                         'eda',
                         'edge',
                         'elasticache',
                         'elasticsearch',
                         'emacs',
                         'emerging',
                         'empirical',
                         'emr',
                         'endtoend',
                         'engine',
                         'engineering', 
                         'ensemble',
                         'enterprise',
                         'environment',
                         'equation',
                         'error',
                         'esquisse',
                         'estimate',
                         'estimation',
                         'etl',
                         'evaluate',
                         'evaluation',
                         'evidence',
                         'excel',
                         'executable',
                         'experiment',
                         'fastapi',
                         'feature',
                         'filter', 
                         'filtering',
                         'finereport',
                         'fivetran',
                         'flair',
                         'flask',
                         'flockdb',
                         'flume',
                         'flux',
                         'forecast',
                         'forest',
                         'formulation',
                         'fortify',
                         'fortran',
                         'fpga',
                         'framework',
                         'fraud', 
                         'full', 
                         'fullstack',
                         'function',
                         'fundamental',
                         'fusion',
                         'fusioncharts',
                         'gan',
                         'gaussian',
                         'gbm',
                         'gedit',
                         'gensim',
                         'geometry',
                         'geospatial',
                         'gcp', 
                         'ggplot',
                         'git',
                         'glacier',
                         'glm',
                         'glue',
                         'gluon',
                         'gmm',
                         'go',
                         'golang',
                         'google',
                         'gpt',
                         'gpu',
                         'gradient',
                         'grafana',
                         'graph',
                         'graphql',
                         'groovy',
                         'grpc',
                         'gtm',
                         'h2o',
                         'hadoop',
                         'hana',
                         'hardware',
                         'hbase',
                         'hcatalog',
                         'hcm',
                         'hdf5',
                         'hdfs',
                         'hibari',
                         'hidden',
                         'hierarchical',
                         'highcharts',
                         'hive',
                         'hiveql',
                         'hpc',
                         'hpcml',
                         'hris',
                         'html',
                         'hux',
                         'hyperparameter',
                         'hyperv',
                         'hypothesis',
                         'ide',
                         'identification',
                         'imagery',
                         'implement',
                         'imputation',
                         'indexing', 
                         'inference',
                         'inferential',
                         'infogram',
                         'informatica',
                         'information',
                         'infrastructure',
                         'ingestion',
                         'insight',
                         'inspector',
                         'integration',
                         'integrity',
                         'intelligence',
                         'interactive',
                         'interface',
                         'interpretation',
                         'invoicing',
                         'ipython',
                         'iterate', 
                         'itsm',
                         'ivr',
                         'kaniko',
                         'kinesis',
                         'janitor',
                         'java',
                         'javascript',
                         'jenkins', 
                         'jira',
                         'jpeg',
                         'jquery',
                         'json',
                         'julia',
                         'jupyter',
                         'kaldi',
                         'keras',
                         'kernel',
                         'kernlab',
                         'kibana',
                         'kizan',
                         'klipfolio',
                         'kmeans',
                         'kms',
                         'knime',
                         'knitr',
                         'knn',
                         'kotlin',
                         'kubeflow',
                         'kubernetes',
                         'label',
                         'lake',
                         'lakefs',
                         'lambda',
                         'lasagne',
                         'lattice',
                         'lda', 
                         'ldap',
                         'leaflet',
                         'learning',
                         'lex',
                         'library',
                         'lidar',
                         'lifecycle',
                         'lightgbm',
                         'likelihood',
                         'linear',
                         'linux',
                         'lisp',
                         'literacy', 
                         'load',
                         'localization',
                         'logic',
                         'logistic',
                         'looker',
                         'loss',
                         'lstm',
                         'ltv',
                         'lubridate',
                         'luigi',
                         'lvq',
                         'lxc',
                         'lxd',
                         'machine',
                         'macie',
                         'mahotas',
                         'maintain',
                         'mapping',
                         'mapreduce',
                         'markov',
                         'mathematica',
                         'mathematics',
                         'matlab',
                         'matplotlib',
                         'matrix',
                         'maven',
                         'maximum',
                         'mboost',
                         'measure',
                         'mercurial',
                         'merge',
                         'mesos',
                         'metadata',
                         'method',
                         'metric',
                         'microservice',
                         'microsoft',
                         'mldb',
                         'mlflow',
                         'mllib',
                         'mlp',
                         'mlpack',
                         'mlr3',
                         'model',
                         'mongodb',
                         'monotone',
                         'monte',
                         'mpi',
                         'multiarmed',
                         'multivariate',
                         'municipal',
                         'mxnet',
                         'mssql',
                         'mysql',
                         'naive',
                         'nearest',
                         'neighbor',
                         'neo4j',
                         'neptune',
                         'netcdf',
                         'network',
                         'neural',
                         'neuron',
                         'nginx',
                         'nifi',
                         'nlp',
                         'nltk',
                         'nodejs',
                         'nonlinear',
                         'nonparametric',
                         'normalize',
                         'nosql',
                         'notebook',
                         'notepad',
                         'nteract',
                         'numerical',
                         'numpy',
                         'nvidia',
                         'nvivo',
                         'object',
                         'objectoriented',
                         'office',
                         'onnx',
                         'ontology',
                         'oozie',
                         'open', 
                         'opencl',
                         'opencv', 
                         'openshift',
                         'openvz',
                         'optical',
                         'optimization',
                         'optimus',
                         'oracle',
                         'orc',
                         'orchestration',
                         'orientdb',
                         'outlier',
                         'paas',
                         'pachyderm',
                         'paddlepaddle',
                         'palantir',
                         'pandas',
                         'parallel',
                         'parametric',
                         'parquet',
                         'parsing',
                         'pattern',
                         'pb',
                         'pca',
                         'pdf',
                         'pearson', 
                         'peoplesoft',
                         'periscope',
                         'perl',
                         'petastorm', 
                         'php',
                         'pickle',
                         'pig',
                         'pillow',
                         'pipeline',
                         'pivot',
                         'pla',
                         'platform',
                         'plc',
                         'plotly',
                         'plsql',
                         'pmml',
                         'png',
                         'podman',
                         'polly',
                         'polyglot',
                         'polymaps',
                         'postgresql',
                         'power',
                         'powershell',
                         'predictive',
                         'preprocessing',
                         'prescriptive',
                         'presto',
                         'principle',
                         'probability',
                         'problem',
                         'procedure',
                         'processing',
                         'proficient',
                         'programming',
                         'pso',
                         'purr',
                         'pycaret',
                         'pycharm',
                         'pydev',
                         'pydot',
                         'pynlpl',
                         'pyspark',
                         'python',
                         'pytest',
                         'pytorch',
                         'pyunit',
                         'qgis',
                         'qlik',
                         'qualitative',
                         'quantitative',
                         'quanteda',
                         'quantum',
                         'query', 
                         'quicksight',
                         'relational', 
                         'r',
                         'radar',
                         'r&d',  
                         'rabbitmq',
                         'rancher',
                         'random',
                         'randomization',
                         'rasa',
                         'raw',
                         'rcrawler',
                         'rdbms',
                         'rbfn',
                         'rbm',
                         'rbn',
                         'rds',
                         'readr',
                         'readxl',
                         'reasoning', 
                         'reactjs',
                         'recognition',
                         'recommendation', 
                         'recommender',
                         'redhat',
                         'redis',
                         'redshift',
                         'reduction',
                         'regression',
                         'regularization',
                         'reinforcement',
                         'rekognition',
                         'relational',
                         'relationship',
                         'remote', 
                         'repeatable',
                         'repository', 
                         'reproducible',
                         'requests',
                         'research', 
                         'rest', 
                         'retrain',
                         'review',
                         'riak',
                         'rio',
                         'rkt',
                         'rmarkdown',
                         'rms',
                         'rnn',
                         'robotics',
                         'rodeo',
                         'rpa',
                         'rstudio',
                         'ruby',
                         'runc',
                         'rust', 
                         's3',
                         'sas',
                         'saas',
                         'sagemaker',
                         'sampling',
                         'sap',
                         'satellite',
                         'scada',
                         'scala',
                         'scale',
                         'scenario',
                         'schema',
                         'science',
                         'scientific',
                         'scrapy',
                         'scripting',
                         'skimage',
                         'scipy',
                         'scraping',
                         'sdk',
                         'seaborn',
                         'search',
                         'segmentation', 
                         'selenium',
                         'semantic',
                         'semistructured',
                         'semisupervised',
                         'sensing',
                         'sensor',
                         'sensitivity',
                         'sentiment',
                         'seo',
                         'series',
                         'server',
                         'service',
                         'set',
                         'shell',
                         'shiny',
                         'sigint',
                         'sigmajs',
                         'signal',
                         'simpleitk',
                         'simulation',
                         'singa',
                         'sisense',
                         'skewness',
                         'skill', 
                         'sklearn',
                         'skopeo',
                         'snaplogic',
                         'snowball',
                         'snowflake',
                         'soa',
                         'software',
                         'solr',
                         'som',
                         'sonnet',
                         'sorcero',
                         'source',
                         'sp',                      # what the hell is this
                         'spacy',
                         'spark',
                         'sparql',
                         'spatial',
                         'speech', 
                         'speechtotext',
                         'splunk',
                         'spotfire',
                         'spreadsheet',
                         'spss',
                         'spyder',
                         'sql',
                         'sqlite',
                         'sqoop',
                         'sqs',
                         'sre',
                         'ssis',
                         'stack',
                         'statsmodels',
                         'statistics',
                         'stata',
                         'stochastic',
                         'stringr',
                         'structured',
                         'superml',
                         'supervised',
                         'svm',
                         'svn',
                         'swift',
                         'synthetic',
                         'system',
                         'sublime',
                         'tableau',
                         'technical',
                         'technique',
                         'technology',
                         'temporal',
                         'tensorflow',
                         'tensorrt',
                         'terrstore',
                         'testing',
                         'text',
                         'textblob',
                         'tfrecords',
                         'tfs',
                         'tfx',
                         'theano',
                         'theory',
                         'thonny',
                         'tibco',
                         'tidyquant',
                         'tidyr',
                         'tidyverse',
                         'tidyxl',
                         'time', 
                         'timely',
                         'tool',
                         'toolkit',
                         'torch',
                         'training',
                         'trajectory',
                         'transformation',
                         'transformers',
                         'translation',
                         'troubleshoot',
                         'tree',
                         'trend',
                         'txt',
                         'typescript',
                         'ubuntu',
                         'uncover', 
                         'unix',
                         'univariate',
                         'unstructured',
                         'user',
                         'unsupervised',
                         'ux',
                         'vagrant',
                         'validation',
                         'variable',
                         'vba',
                         'vector',
                         'vega',
                         'version',
                         'vertica',
                         'vim',
                         'virtualbox',
                         'virtualization',
                         'visio',
                         'vision',
                         'visualization',
                         'vlookup',
                         'vmware',
                         'volume',
                         'vpc',
                         'vroom',
                         'vsts',
                         'warehouse',
                         'warfare',
                         'watson', 
                         'web',
                         'weka',
                         'windock',
                         'windows',
                         'word2vec',
                         'xgboost',
                         'xlsx',
                         'xml',
                         'xray',
                         'yaml',
                         'yarn',
                         'zeromq',
                         'zerovm',
                         'zip',
                         'zoho',
                         'zookeeper']    
    
    ds_soft_skill_terms = ['adhoc',
                           'ability',
                           'agile',
                           'ambiguity',
                           'articulate',
                           'assumption',
                           'audience',
                           'authenticity',
                           'best',
                           'boundary',
                           'business',
                           'cause',
                           'challenge', 
                           'clear',
                           'cognitive',
                           'collaborate',
                         'communicate',
                         'complexity',
                         'concise',
                         'conclusion',
                         'consulting',
                         'contribute',
                         'innovative',
                         'credibility',
                         'critical',
                         'crossfunctional',
                         'curious',
                         'deadline',
                         'documenting',
                         'draw',              
                         'efficient',
                         'empathetic',
                         'enthusiastic',
                         'entrepreneurship',
                         'environment',
                         'ethics',
                         'excellence',
                         'exercise', 
                         'experience',
                         'explain',
                         'fast-paced',
                         'finding',
                         'flexible',
                         'focus',
                         'forward-thinking',
                         'guidance',
                         'helpful',
                         'diligent',
                         'holistic',
                         'idea',
                         'impactful',
                         'independent',
                         'innovative',
                         'intelligent',
                         'interpersonal',
                         'interpret',
                         'judgment',
                         'learn',
                         'level',
                         'listening', 
                         'literature',
                         'making',
                         'management',
                         'meeting',
                         'member',
                         'meticulous',
                         'minimal',
                         'multitask',
                         'nontechnical', 
                         'novel',
                         'openminded',
                         'optimus',
                         'organized',
                         'people',
                         'perspective',
                         'powerpoint',
                         'practice',
                         'pragmatic',
                         'prioritize',
                         'proactive',
                         'problem-solving',
                         'productive',
                         'prototyping',
                         'push',
                         'rapport',
                         'read',
                         'reliable',
                         'resilient',
                         'respectful',
                         'risk',
                         'root', 
                         'self-confident',
                         'storytelling',
                         'ethical',
                         'user',
                         'verbal',
                         'written',
                         'word',
                         'work',
                         'workflow']  
    
    ds_prof_skill_terms = ['ability',                         # business building, leadership, client engagement
                           'actionable',                         # team building, mentorship make as MECE as possible WRT soft skills
                           'advisor',
                           'agile',
                           'assumption',
                           'backlog',
                           'build',
                           'business',
                           'career', 
                           'case',
                           'change', 
                           'chosen',
                           'client',
                           'collaborate',
                           'communicate',
                           'complexity', 
                           'crossfunctional',
                           'data',
                           'decision',
                           'delivery',
                           'development',
                           'digital',
                           'domain',
                           'data-driven',
                           'earn', 
                           'efficient',
                           'end', 
                           'engagement',
                           'entrepreneurial',
                           'executive',
                           'experience',
                           'expert',
                           'fast-paced',
                           'flexible',
                           'feedback',
                           'gartner',
                           'generate',
                           'functional',
                           'gdpr',
                           'governance',
                           'impactful',
                           'innovative',
                           'insight',
                           'jaic',
                           'kaggle',
                           'kanban',
                           'knowledge',
                           'leadership',
                           'lean',
                           'line',
                           'management',
                           'market',
                           'marketing',
                           'matter',
                           'meeting',
                           'member',
                           'mentor',
                           'mvp',
                           'objective',
                           'operations',
                           'paper',
                           'peer', 
                           'pm',
                           'pmp',
                           'problem',
                           'process',
                           'product',
                           'professional',
                           'profitability',
                           'program',
                           'project', 
                           'proposal',
                           'prototype',
                           'proven',
                           'punctual',
                           'recommendation',
                           'requirement',
                           'review',
                           'risk',
                           'roadmap',
                           'sales',
                           'science',
                           'scrum',
                           'service',
                           'sigma',
                           'six', 
                           'skill',
                           'solution',
                           'sprint',
                           'problem-solving',
                           'stakeholder',
                           'storytelling',
                           'strategy',
                           'subject',
                           'technical',
                           'tell', 
                           'thought-leadership',
                           'transformation',
                           'ethical', 
                           'understanding',
                           'user',
                           'impactful',
                           'warfighters',
                           'waterfall',
                           'white', 
                           'win',
                           'work',
                           'workstreams'] 
    
    ds_skills_combined = ds_cred_terms + ds_tech_skill_terms + ds_soft_skill_terms + ds_prof_skill_terms
    subtopics_combined = (subtopic_aws + subtopic_cloud + subtopic_agile + subtopic_languages + subtopic_big_data +
                         subtopic_nlp + subtopic_viz + subtopic_r + subtopic_dl_frameworks +
                         subtopic_containers + subtopic_datatypes + subtopic_ide + subtopic_databases +
                         subtopic_version_control + subtopic_mathematics + subtopic_sql + subtopic_dl_algorithms +
                         subtopic_dl_supporting + subtopic_linux + subtopic_python +
                         subtopic_platforms + subtopic_it_and_web + subtopic_geospatial + subtopic_other + 
                         subtopic_javascript + subtopic_excel + subtopic_math) #subtopic_tooling +

    # confirm exclusivity of each list with the additional_stopwords list in the clean_terms_for_nlp function
    print('\n***** Stopword and Skill List Testing ***** \n')
    print(f'Test for Stopword pollution in skill lists: {not set(ds_skills_combined).isdisjoint(additional_stopwords)}\n')
    stopword_pollutants = list(set(additional_stopwords).intersection(ds_skills_combined))
    print(f'Stopword pollutants from primary skill lists: {stopword_pollutants}\n')
    
    # confirm exclusivity of the subtopic lists with the additional_stopwords list in the clean_terms_for_nlp function
    print(f'Test for Stopword pollution in subtopic lists: {not set(subtopics_combined).isdisjoint(additional_stopwords)}\n')
    stopword_pollutants = list(set(additional_stopwords).intersection(subtopics_combined))
    print(f'Stopword pollutants from primary skill lists: {stopword_pollutants}\n')  
    
    # confirm all terms in subtopics are reflected in at least on skill list
    print(f'Test for subtopic terms missing in skill lists: {not set(subtopics_combined).isdisjoint(ds_skills_combined)}\n')
    subtopic_missing_terms = sorted(list(set(subtopics_combined) - set(ds_skills_combined)))
    print(f'Subtopic terms missing from primary skill lists: {subtopic_missing_terms}\n')  
    
    return ds_cred_terms, ds_tech_skill_terms, ds_soft_skill_terms, ds_prof_skill_terms, ds_skills_combined, subtopic_python


def nlp_count_n_grams(terms_for_nlp, n_gram_count, n_gram_range_start, n_gram_range_stop, print_flag=False):
    '''
    Count the volume of n_grams present within the job_description field of the Indeed data.

    Parameters
    ----------
    terms_for_nlp : list
        List containing scraped and cleaned terms from the series of interest; created in the clean_for nlp function.
    n_gram_count : integer
        A parameter representing the dimensionality of the n_grams of interest (e.g., 2 = bigram, 3 = trigram, etc.).
    n_gram_range_start : integer
        A parameter indicating the uoper bound (i.e., the n_gram with the highest count) of the n_grams Series.
    n_gram_range_stop : integer
        A parameter indicating the lower bound (i.e., the n_gram with the lowest count) of the n_grams Series.

    Returns
    -------
    n_grams : Dataframe
        Contains the cleaned and processed n_grams; sorted by count from highest to lowest.

    '''
    # indicate processing status in the console
    print('\n***** Natural Language Processing *****')
    print('Identifying n_grams...\n')

    # count n grams in the field of interest, bounding the count according to n_gram_range_start and n_gram_range_stop
    n_grams_raw = (pd.Series(nltk.ngrams(terms_for_nlp, n_gram_count)).value_counts())[n_gram_range_start:n_gram_range_stop]
    
    # convert n_grams series into a dataframe, clean the terms and reset the index
    n_grams_cols = ['count']
    n_grams = pd.DataFrame(n_grams_raw, columns=n_grams_cols)
    
    # pull the n_grams out of the index, reset the index and extract only the ngrams
    n_grams['grams'] = n_grams.index.astype('string')
    n_grams.reset_index(inplace=True, drop=True)
    n_grams['grams'] = [" ".join(re.findall("[a-zA-Z0-9]+", x)) for x in n_grams['grams']]
    
    if print_flag == True:
        print(f'Count of ngrams for new data parsing:\n{n_grams}\n')

    return n_grams


def nlp_filter_terms():
    '''
    (in development and not currently used) Filter datafames for specific terms of interest

    Returns
    -------
    None.

    '''
    # get only rows without the key terms - not sure if this works
    # inverse_boolean_series = ~pd.Series(terms_for_nlp).isin(value_list)
    # inverse_filtered_df = pd.Series(terms_for_nlp)[inverse_boolean_series]
    
    value_list = ['python']#['data', 'science', 'python']
    boolean_series = pd.Series(terms_for_nlp).isin(value_list)
    filtered_series = pd.Series(terms_for_nlp)[boolean_series]
    print(len(filtered_series))
    print(filtered_series.value_counts())
    print(filtered_series[:10])

 
def parse_new_data(terms_for_nlp, ds_skills_combined, term_fixes):
    '''
    Parse new Indeed data for key terms, additional stopwords and term fixes. Not yet called, so currently
    just a holding tank for code to run when bringing in new data.

    Parameters
    ----------
    terms_for_nlp : list
        List containing scraped and cleaned terms from the series of interest; created
        in the clean_for nlp function.
    ds_skills_combined : list
        Combination of the credentials, technical, soft and professional skills lists.
    term_fixes : dictionary
        Mapping of misspellings, concatentions and term rollups used as a final
        repair of n_grams.

    Returns
    -------
    None.

    '''    
    # Step 1: create the skill lists
    (ds_cred_terms, ds_tech_skill_terms, ds_soft_skill_terms, ds_prof_skill_terms, ds_skills_combined,
     subtopic_python) = nlp_skill_lists(additional_stopwords)
     
    # Step 2: redact ds_skills_combined from terms_for_nlp
    new_terms_for_nlp = [x for x in terms_for_nlp if x not in ds_skills_combined] 
    
    # Step 3: redact term_fixes values (i.e., the values from the term_fixes dictionary)
    new_terms_for_nlp = [x for x in new_terms_for_nlp if x not in list(set((term_fixes.values())))]
    
    # Step 4: count the volume of n-grams from the job_description field and the given range
    n_gram_count = 1
    n_gram_range_start, n_gram_range_stop  = 0, 200
    _ = nlp_count_n_grams(new_terms_for_nlp, n_gram_count, n_gram_range_start, n_gram_range_stop, print_flag=True)
    
    # (Optional) Step 5: Print in convenient form for addition to additional_stopword list
    # n_grams = [x[2:-3] for x in list(n_grams.index)] # This can be swapped for the regex code; this stopped working on 2022-03-10
    # n_grams = [x for x in list(n_grams.grams)]
    # print(f'List of ngrams for new data parsing:\n{n_grams}\n')


####### !!!!!!!! WORKING HERE: create function for searching job listings for a single term of interest
def utilities():
    pass
    # probably need to bring in df_jobs_raw, or a later processed version that is used in the percentage charts
    # df_jobs_raw is a df, where each record is a list of tokenized strings
    # NEED THE PROCESSED df_jobs DATAFRAME!! Might need to call clean_listings_for_nlp, of get it passed in (optimal)
    # maybe
    # 1) recombine each line into a single string
    # 2) flag each line for a single term 


###### MAIN EXECUTION BELOW ######
# define universal variables and data paths
csv_path = r'C:\Users\ca007843\Documents\100_mine\nlp\data_ds'

def main_program(csv_path):
    # load and concatenate the raw csvs collected from Indeed and stored in a data_xx directory
    df_raw = load_and_concat_csvs(csv_path)
    
    # execute basic cleaning of the df_raw dataframe (e.g., deal with NaNs, drop duplicates, etc.)
    df = clean_raw_csv(df_raw)

    # this begins the NLP component
    # select the series of interest for natural language processing (e.g., company, job_title, job_description, etc.)
    series_of_interest = df['job_description']
    
    # execute all NLP data conditioning (e.g., lowercasing, lemmatization, etc.)
    terms_for_nlp, additional_stopwords, term_fixes = clean_terms_for_nlp(series_of_interest)
    
    # create lists for key terms related to credentialing and key skill sets, and a combined list for all terms of interest
    (ds_cred_terms, ds_tech_skill_terms, ds_soft_skill_terms,
     ds_prof_skill_terms, ds_skills_combined, subtopic_python) = nlp_skill_lists(additional_stopwords)
    
    # count all n_grams 
    n_gram_count = 1
    n_gram_range_start, n_gram_range_stop  = 0, 100 # 3900, 4000 # NEXT - advance the range
    n_grams = nlp_count_n_grams(terms_for_nlp, n_gram_count, n_gram_range_start, n_gram_range_stop)
    
    # visualize results of NLP
    visualize_indeed_metadata(df)
    visualize_word_clouds(terms_for_nlp, series_of_interest)
    
    # visualize n_grams and skill lists as horizontal bar plots
    df_jobs_raw, unique_titles_viz = visualize_n_grams(n_grams, ds_cred_terms, ds_tech_skill_terms, ds_soft_skill_terms, ds_prof_skill_terms,
                                    terms_for_nlp, series_of_interest, additional_stopwords, term_fixes, df)
    
    visualize_subtopic(df, df_jobs_raw, terms_for_nlp, subtopic_python, unique_titles_viz, viz_title='Python Subtopic')

    return df, series_of_interest, terms_for_nlp, additional_stopwords, term_fixes, n_grams, ds_cred_terms


# execute main program
df, series_of_interest, terms_for_nlp, additional_stopwords, term_fixes, n_grams, ds_cred_terms = main_program(csv_path)

# close time calculation
end_time = time.time()
print(f'\nTotal Processing Time: {(time.time() - start_time) / 60:.2f} minutes')

# clean up intermediate dataframes and variables
del start_time, end_time









#######  ARCHIVE ######
# original -> df_term_fixes['terms'].replace(dict(zip(list(term_fixes.values()), list(term_fixes.keys()))), regex=False, inplace=True)
# code for inverting the dictionary and sorting it alphabetically by key; then switch them
# inv_map = {v: k for k, v in term_fixes.items()}
# inv_map
# dict_items = sorted(inv_map.items())
# dict_items
# dict(dict_items)

# dict_alpha = sorted(term_fixes.items())

# hold = df[df['job_description'].str.contains('attention to detail')]
# hold = df[df['job_description'].str.contains('|'.join(['passion','collaborate','teamwork','team work', 'interpersonal','flexibility','flexible','listening','listener','listen','empathy','empathetic']))]

# def utilities(terms_for_nlp):
#     # working on extracting hex colors from seaborn palletes
#     pal = sns.color_palette('mako')
#     print(pal.as_hex())

# ngram_combined_sns = [x for x in ngram_combined_sns.grams if x not in ngrams_to_redact]
# ngram_combined_sns['grams'] = ngram_combined_sns.grams.apply(lambda x: [i for i in x if i != ngram_combined_sns])

# import nltk

# w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
# lemmatizer = nltk.stem.WordNetLemmatizer()

# def lemmatize_text(text):
#     return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

# df_test = pd.DataFrame(series_of_interest.str.lower())
# df_test['job_description'] = df_test.job_description.apply(lemmatize_text)

# failed attempt at term_fixes find/replace
# # GETTING CLOSER!! This replaces, but gets partials, like if 'rf' is part of a word it gets replaced with 'random forest'
# df_jobs['test'] = df_jobs['job_description'].replace(term_fixes, regex=True)

# # this did not work at all; didn't replace anything
# df_jobs['test'] = df_jobs['job_description'].replace(term_fixes, regex=False) 

# # trying this - threw an error
# df_jobs['test'] = df_jobs['job_description'].str.replace(term_fixes, regex=False)

# # trying this - didn't work
# df_jobs['test'] = df_jobs['job_description'].str.replace(str(term_fixes.keys()), str(term_fixes.values()), regex=True)


# # w = "Where are we one today two twos them"                                # YES, each record in df_jobs['job_description']
# # lookup_dict = {"one":"1", "two":"2", "three":"3"}                         # YES, the term_fixes dictionary
# # pattern = re.compile(r'\b(' + '|'.join(lookup_dict.keys()) + r')\b')      # YES, just point at term_fixes
# # output = pattern.sub(lambda x: lookup_dict[x.group()], w)                 # Not yet, need to make this loop

# # # try #1
# # w = "Where are we one today two twos them"                                
# # lookup_dict = term_fixes                         
# # pattern = re.compile(r'\b(' + '|'.join(lookup_dict.keys()) + r')\b')     
# # output = pattern.sub(lambda x: lookup_dict[x.group()], w)                 

# # # boooooooooooooooooo
# # df_jobs['test'] = df_jobs['job_description'].apply(lambda x: pattern.sub(lambda x: lookup_dict[x.group()], word) for word in df_jobs['job_description'])

# # setting up for stackoverflow
# df = pd.DataFrame(data={'job_description': ['knowledge of algorithm like rf',
#                                             'must have a mastersphd',
#                                             'must be trustworthy and possess curiosity',
#                                             'we realise performance is critical']})

# df = pd.DataFrame(data={'job_description': [['knowledge', 'of', 'algorithm', 'like', 'rf'],
#                                             ['must', 'have', 'a', 'mastersphd'],
#                                             ['must', 'be', 'trustworthy', 'and', 'possess', 'curiosity'],
#                                             ['we', 'realise', 'performance', 'is', 'critical']]})



# df_jobs['test'] = df_jobs['job_description'].apply(lambda x: [word.replace(term_fixes, regex=True) for word in x])
# df_jobs['test'] = df_jobs['job_description'].apply(lambda x: [word.replace('data', 'test_success') for word in x])



# df_jobs['job_description'] = df_jobs['job_description'].apply(lambda x: [word.replace('data', 'test_success') for word in x])

# df_jobs['job_description'] = df_jobs['job_description'].apply(lambda x: [word.replace(str(term_fixes.keys()), str(term_fixes.values())) for word in x])

# # The originals
# df_term_fixes['terms'].replace(dict(zip(list(term_fixes.keys()), list(term_fixes.values()))), regex=False, inplace=True)
# terms_for_nlp = list(df_term_fixes['terms'])

# # Failures
# df_jobs['job_description'].replace(list(term_fixes.keys()), list(term_fixes.values()), regex=False, inplace=True)
# df_jobs['job_description'].replace(dict(zip(list(term_fixes.keys()), list(term_fixes.values()))), regex=False, inplace=True)
# df_jobs['test'] = df_jobs['job_description'].replace(term_fixes, regex=True)
# df_jobs['test'] = df_jobs['job_description'].replace(term_fixes, regex=False)
# df_jobs['test'] = [x.replace(term_fixes, regex=False) for x in df_jobs['job_description']] # failed when df records are lists
# df_jobs['test'] = [x.replace(term_fixes, regex=False) for x in df_jobs['job_description']] # still fails


# # seeking help
# # data = {'Name':[["'Tom' 'is' 'qualified'"], 'nick', 'krish', 'jack'],
# # 'Age':[20, 21, 19, 18]}
# # df = pd.DataFrame(data)
       
        
# trying from stackoverflow
# for k in term_fixes:
#     df_jobs['test'] = (df_jobs['job_description'].str.replace(r'(^|(?<= )){}((?= )|$)'.format(k), term_fixes[k]))

# DON'T THINK I NEED THIS STUFF BELOW    
# apply parts of speech tags; nltk.download('averaged_perceptron_tagger')
# df_jobs['job_description'] = df_jobs['job_description'].apply(nltk.tag.pos_tag)


# # tag parts of speech
# df_jobs['job_description'] = df_jobs['job_description'].apply(lambda x: [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in x])

# # lemmatize
# wnl = WordNetLemmatizer()
# df_jobs['job_description'] = df_jobs['job_description'].apply(lambda x: [wnl.lemmatize(word, tag) for word, tag in x])

# second stackoverflow question
# df = pd.DataFrame(data={'job_description': [['innovative', 'data', 'science'],
#                                             ['scientist', 'have', 'a', 'masters'],
#                                             ['database', 'rf', 'innovative'],
#                                             ['sciencebased', 'data', 'performance']],
#                         'innovative': [True, False, True, False],
#                         'data': [True, False, False, True],
#                         'rf': [False, False, True, False]})

# alternate solutions to the skill list flagging operation
# start_time = time.time()
# e = df_jobs['job_description'].explode()
# df_jobs[ds_cred_terms] = pd.concat([e.eq(t).groupby(level=0).any().rename(t) for t in ds_cred_terms], axis=1)
# print(time.time() - start_time)

# start_time = time.time()
# e = df_jobs['job_description'].explode()
# df_jobs[ds_cred_terms] = pd.concat([e.eq(t).rename(t) for t in ds_cred_terms], axis=1).groupby(level=0).any()
# print(time.time() - start_time)


# # MY IMPLEMENTATION AND YES THIS WORKS!!
# def find_bigram_match_to_cred_list(data):
#     output = np.zeros((data.shape[0], len(bigram_match_to_cred_list)), dtype=bool)
#     for i, d in enumerate(data):
#         possible_bigrams = [' '.join(x) for x in list(nltk.bigrams(d)) + list(nltk.bigrams(d[::-1]))]
#         indices = np.where(np.isin(bigram_match_to_cred_list, list(set(bigram_match_to_cred_list).intersection(set(possible_bigrams)))))
#         output[i, indices] = True
#     return list(output.T)

# output = find_bigram_match_to_cred_list(df_jobs['job_description'].to_numpy())
# df_jobs = df_jobs.assign(**dict(zip(bigram_match_to_cred_list, output)))

# Alternate Title Brainstorm for Skill Bar Charts
# Where to Focus Your Credentials
# Focus Your Credentialing on These Key Areas ##############
# Skills, Terms, Areas, Subjects, Qualifications, Advantages, 
# Credential Intensity: A Measure of How Deeply Employers Care
# Consider How Intensely Employers Care about Each Credential Focus Area

# shell command to count total lines of code:
# pygount --format=summary .

# for movie in movies:
#   	# If actor is not found between character 37 and 41 inclusive
#     # Print word not found
#     if movie.find("actor", 37, 42) == -1:
#         print("Word not found")
#     # Count occurrences and replace two with one
#     elif movie.count("actor") == 2:  
#         print(movie.replace("actor actor", "actor"))
#     else:
#         # Replace three occurrences with one
#         print(movie.replace("actor actor actor", "actor"))

# for movie in movies:
#   try:
#     # Find the first occurrence of word
#   	print(movie.index('money', 12, 51))
#   except ValueError:
#     print("substring not found")
