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
    additional_stopwords = sorted(list(set(['14042', '3rd', '401k', '50', '500', 'a16z', 'able', 'accepted',
                                            'access', 'accommodation', 'accomodation', 'account', 'across',
                                            'action', 'additional', 'adhering', 'affiliation', 'affirmative',
                                            'age', 'allen', 'also', 'amazecon', 'america', 'amount', 'ancestry',
                                            'andor', 'angeles', 'another', 'applicable', 'applicant', 'application',
                                            'apply', 'applyaccommodationschangehealthcarecom', 'area', 'around',
                                            'arrest', 'assigned', 'assistance', 'assurance', 'authentic', 'authorization',
                                            'authorized', 'background', 'balance', 'base', 'based', 'basic', 'basis',
                                            'belief', 'belonging', 'benefit', 'best', 'beyond', 'billion', 'bonus',
                                            'booz', 'broad', 'california', 'call', 'candidate', 'cannot', 'capital',
                                            'card', 'care', 'chain', 'chance', 'characteristic', 'chase', 'check',
                                            'chicago', 'childbirth', 'citizen', 'citizenship', 'city', 'civil',
                                            'classified', 'click', 'clinical', 'closely', 'color', 'colorado', 'come',
                                            'comfortable', 'commitment', 'committed', 'commuter', 'company', 'compensation',
                                            'competitive', 'complaint', 'compliance', 'comprehensive', 'confidential',
                                            'consider', 'consideration', 'considered', 'consistent', 'contact', 'contractor',
                                            'conversation', 'conversational', 'conviction', 'core', 'cover', 'covid',
                                            'covid19', 'creating', 'credit', 'creed', 'criminal', 'culture', 'current',
                                            'currently', 'cv', 'date', 'day', 'dc', 'december', 'dedicated', 'defense',
                                            'demand', 'dental', 'deploying', 'description', 'disability', 'disclose',
                                            'disclosed', 'disclosure', 'discriminate', 'discrimination', 'discussed',
                                            'disruption', 'distributed', 'diverse', 'diversity', 'domestic', 'drive',
                                            'drugfree', 'drugtesting', 'due', 'duty', 'eeo', 'eg', 'eligibility',
                                            'eligible', 'email', 'embracing', 'employee', 'employeeled', 'employer',
                                            'employment', 'encouraged', 'enjoy', 'ensure', 'equal', 'equity', 'essential',
                                            'estate', 'etc', 'every', 'everyone', 'executive', 'existing', 'expression',
                                            'extensive', 'external', 'fair', 'family', 'fargo', 'federal', 'feel', 'following',
                                            'fortune', 'francisco', 'friday', 'full', 'fulltime', 'fully', 'furnish',
                                            'furtherance', 'gender', 'genetic', 'genetics', 'getty', 'globe', 'go',
                                            'good', 'great', 'group', 'growing', 'hand', 'harassment', 'health', 'healthcare',
                                            'hearing', 'high', 'highly', 'hire', 'hiring', 'history', 'holiday', 'home',
                                            'host', 'hour', 'httpswwwamazonjobsendisabilityus',
                                            'httpswwwdolgovofccpregscomplianceposterspdfofccp_eeo_supplement_final_jrf_qa_508cpdf',
                                            'httpswwweeocgovemployerseeolawposter', 'human', 'ibm', 'id', 'identity',
                                            'il', 'include', 'including', 'inclusion', 'inclusive', 'indepth', 'industry',
                                            'inquired', 'inside', 'insurance', 'internal', 'job', 'johnson', 'join',
                                            'jpmorgan', 'kept', 'key', 'kpmg', 'largest', 'law', 'laws', 'least', 'leave',
                                            'legally', 'letter', 'level', 'leverage', 'life', 'lightspeed', 'like', 'limited',
                                            'local', 'location', 'lockheed', 'long', 'looking', 'los', 'love', 'm', 'made',
                                            'maintaining', 'make', 'mandate', 'manner', 'marital', 'martin', 'match',
                                            'matching', 'mature', 'may', 'medical', 'medium', 'mental', 'million', 'minimum',
                                            'monday', 'motivated', 'multiple', 'must', 'national', 'navigate', 'need',
                                            'new', 'nondiscrimination', 'nonessential', 'north', 'notice', 'offer', 'one',
                                            'opportunity', 'opportunityaffirmative', 'order', 'ordinance', 'orientation',
                                            'origin', 'outside', 'overview', 'package', 'paid', 'pandemic', 'parental',
                                            'part', 'participate', 'partnership', 'party', 'pay', 'per', 'perform', 'performed',
                                            'perk', 'personal', 'phone', 'physical', 'place', 'plan', 'please', 'plus',
                                            'point', 'policy', 'political', 'position', 'posse', 'poster', 'preemployment',
                                            'preferred', 'pregnancy', 'premier', 'prescribe', 'previous', 'primary', 'prior',
                                            'privacy', 'privilege', 'proceeding', 'process', 'proof', 'protected', 'proud',
                                            'provide', 'providing', 'public', 'puerto', 'purchase', 'qualification', 'qualified',
                                            'quality', 'race', 'range', 'rapidly', 'real', 'reasonable', 'receive', 'recruiter',
                                            'recruiting', 'recruitment', 'referral', 'regard', 'regarding', 'regardless',
                                            'regulation', 'regulatory', 'reimbursement', 'relic', 'religion', 'religious',
                                            'relocation', 'remote', 'remotely', 'reporting', 'req', 'request', 'required',
                                            'requirement', 'requires', 'resource', 'responsibility', 'responsible', 'resume',
                                            'retirement', 'reward', 'rico', 'role', 'safety', 'salary', 'salesforcecom',
                                            'salesforceorg', 'san', 'saving', 'schedule', 'scratch', 'secret', 'seeking',
                                            'self', 'sending', 'senior', 'sense', 'sequoia', 'set', 'sex', 'sexual', 'shape',
                                            'shift', 'show', 'sincerely', 'small', 'social', 'someone', 'sound', 'spending',
                                            'sponsorship', 'sr', 'standard', 'start', 'state', 'statement', 'status', 'stay',
                                            'stock', 'strong', 'suite', 'summary', 'supplemental', 'supply', 'support', 'sure',
                                            'suspended', 'talented', 'teladoc', 'tenure', 'term', 'therapeutic', 'third',
                                            'total', 'toughest', 'transgender', 'translate', 'transparency', 'travel', 'trial',
                                            'trove', 'tuition', 'type', 'u', 'union', 'unit', 'united', 'unitedhealth', 'vaccine',
                                            'unsolicited', 'upon', 'using', 'vaccinated', 'vaccination', 'variety', 'vast',
                                            'veteran', 'visa', 'vision', 'visit', 'washington', 'way', 'wed', 'well', 'live',
                                            'wellbeing', 'wellness', 'whats', 'wide', 'within', 'without', 'workforce',
                                            'worklife', 'workplace', 'world', 'would', 'york', 'youll', 'zone', 'view', 'note',
                                            'achieve', 'goal', 'organization', 'future', 'sourcing', 'offering', 'throughout',
                                            'choice', 'let', 'know', 'strategic', 'immigration', 'available', 'important',
                                            'government', 'agency', 'financial', 'institution', 'resolve', 'issue', 'active',
                                            'leveraging', 'drug', 'free', 'monitor', 'successful', 'completion', 'community', 'serve',
                                            'hired', 'accenture', 'chief', 'officer', 'investigation', 'otherwise', 'unless',
                                            'right', 'thing', 'better', 'function', 'response', 'formal', 'charge', 'b', '2021',
                                            'conducted', 'legal', 'placing', 'manager', 'talent', 'firm', '100', 'ongoing',
                                            'ethnicity', 'conference', 'resident', 'submitting', 'acknowledge', 'mix', 'building',
                                            'celebrates', 'httpswwwdolgovofccppdfpaytransp_20english_formattedesqa508cpdf',
                                            'vacation', 'sick', 'january', '2022', 'tiger', 'global', 'get', 'done', 'via',
                                            'top', 'internally', 'externally', 'performance', 'indicator', 'thrive',
                                            'continue', 'grow', 'faculty', 'staff', 'bring', 'closer', 'result', 'space',
                                            'virtual', 'assistant', 'approved', 'condition', 'save', 'money', 'create',
                                            'understand', 'various', 'production', 'activity', 'take', 'department', 'provides',
                                            'familiarity', 'others', 'assist', 'needed', 'enable', 'believe', 'effective',
                                            'different', 'planning', 'task', 'want', 'supporting', 'appropriate', 'consumer',
                                            'effort', 'define', 'document', 'conduct', 'potential', 'used', 'patient', 'inc',
                                            'find', 'documentation', 'finance', 'similar', 'first', 'specific', 'share',
                                            'deployment', 'includes', 'require', 'focused', 'act', 'implementing', 'desired',
                                            'organizational', 'person', 'many', 'brand', 'search', 'content', 'address',
                                            'directly', 'driving', 'execution', 'colleague', 'general', 'online', 'addition',
                                            'asset', 'commercial', 'meaningful', 'purpose', 'ideal', 'today', 'investment',
                                            'monitoring', 'provider', 'interest', 'event', 'seek', 'assessment', 'necessary',
                                            'country', 'option', 'revenue', 'execute', 'corporate', 'ensuring', 'direction',
                                            'youre', 'performing', 'enhance', 'component', 'significant', 'possible', 'give',
                                            'complete', 'cost', 'site', 'form', 'guide', 'student', 'common', 'contract',
                                            'number', 'changing', 'two', 'week', 'embrace', 'furthering', 'submitted', 'force',
                                            'box', 'annual', 'reinforced', 'maintains', 'accordance', 'protection', 'requesting',
                                            '190', 'chapter', 'coordinated', 'increasingly', 'chapter', 'globally', 'affinity',
                                            'reaching', 'banking', 'junior', 'moving', 'forward', 'undue', 'hardship', 'assign',
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
                                            'greatest', 'bringing', 'hybrid', 'relying', 'bringing', 'relying', 'statementibm',
                                            'personalized', 'ibms', 'ibmers', 'restless', 'depth', 'empower', 'depth', 'select',
                                            'facilitate', 'guarantee', 'continued', 'counseling', 'consisting', 'bonding',
                                            'giving', 'obligated', 'sealed', 'expunged', 'rich', 'discriminated', 'asked',
                                            'observance', 'toll', 'speak', 'employed', 'affiliated', 'send', 'employed',
                                            'sealed', 'shall', 'mayo', 'clinic', 'lottery', 'except', 'casual', 'dress',
                                            'lawfacebook', 'accommodationsextfbcom', 'goldman', 'sachs', 'jp', 'morgan',
                                            'private', 'sector', 'mutual', 'mount', 'sinai', 'exhaustive', 'list', 'provided',
                                            'interview', 'unlimited', 'pto', 'keep', 'facing', 'silicon', 'valley', 'adequate',
                                            'protocol', 'imperative', 'sustain', 'pwc', 'pwcsponsored', 'signon', 'restricted',
                                            'mayo', 'clinic', 'h1b', 'lottery', 'canada', 'saics', 'saic', 'ie', 'collateral',
                                            'autonomous', 'vehicle', 'unwavering', 'aim', 'discharge', 'additionally', 'audit',
                                            '8899009', '877', 'accept', 'headhunter', 'insider', 'threat', 'mobile', 'app',
                                            'described', 'representative', 'weve', 'got', 'st', 'louis', 'skillsexperience',
                                            'helpful', 'leaf', 'absence', 'crucial', 'happiness', 'participates', 'everify',
                                            'striking', 'healthy', 'verizon', '41', 'cfr', 'lot', 'fun', 'profit', 'sharing',
                                            'recognize', 'strength', 'constantly', 'strength', 'linked', 'vacancy', 'apps',
                                            'announcement', 'salesforce', 'einstein', 'landscape', 'builder', 'constantly',
                                            'expand', 'whether', 'array', 'celebrate', 'stronger', 'twitter', 'instagram',
                                            'kind', 'connects', 'arlington', 'va', 'prepared', 'ready', 'voice', 'carves',
                                            'factor', 'starting', 'influenced', 'path', 'us', 'discover', 'possibility',
                                            'unleash', 'ready', 'express', 'craving', 'prepared', 'actual', 'influenced',
                                            'carves', 'ibmare', 'possibility', 'voice', 'cocreate', 'individuality', 'add',
                                            'alter', 'fabric', 'truly', 'personally', 'hundred', 'thousand', 'six', 'month',
                                            'layoff', 'recall', 'ethnic', 'sensory', 'past', 'operate', 'there', 'always',
                                            'edward', 'jones', 'nice', 'have', 'determine', 'exceed', 'expectation', 'harmony',
                                            'predisposition', 'carrier', 'dealing', 'listing', 'figure', 'backwards',
                                            'vigorously', 'hiv', 'past', 'organizationleaders', 'customerobsessed', 'thirteen',
                                            '85000', 'property', 'casualty', 'mass', 'brigham', 'driver', 'license',
                                            'second', 'dose', 'investigator', 'pittsburgh', 'pa', 'attract', 'retain',
                                            'catered', 'lunch', 'booster', 'shot', 'maternity', 'prohibited', 'behavior',
                                            'exceptional', 'energy',
                                             'aspect',
                                             'expected',
                                             'discovery',
                                             'setting',
                                             'material',
                                             'produce',
                                             'see',
                                             'improving',
                                             'academic',
                                             'startup',
                                             'exciting',
                                             'recognized',
                                             'contribution',
                                             'answer',
                                             'creation',
                                             'play',
                                             'robust',
                                             'payment',
                                             'integrate',
                                             'class',
                                             'explore',
                                             'overall',
                                             'establish',
                                             'move',
                                             'integrated',
                                             'three',
                                             'desire',
                                             'obtain',
                                             'pricing',
                                             'regular',
                                             'maintenance',
                                             'interested',
                                             'along',
                                             'increase',
                                             'utilize',
                                             'manufacturing',
                                             'built',
                                             'acquisition',
                                             'engage',
                                             'thats',
                                             'submit',
                                             'roadmap',
                                             'coordinate',
                                             'prepare',
                                             'utilizing',
                                             'foundation',
                                             'encourage',
                                             'cycle',
                                             'course',
                                             'vendor',
                                             'become',
                                             'device',
                                             'smart',
                                             'ass',
                                             'effectiveness',
                                             'mean',
                                             'adoption',
                                             'located',
                                             'preferably',
                                             'division',
                                             'mindset',
                                             'scope',
                                             'exposure',
                                             'progress',
                                             'transforming',
                                             'lab',
                                             'enabling',
                                             'interaction',
                                             'look',
                                             'promote',
                                             'example',
                                             'foster',
                                             'specification',
                                             'associated',
                                             'daily',
                                             'realtime',
                                             'campaign',
                                             'independent',
                                             'storage',
                                             'competency',
                                             'website',
                                             'medicine',
                                             'incentive',
                                             'follow',
                                             'facility',
                                             'coverage',
                                             'validate',
                                             'continuously',
                                             'defining',
                                             'bank',
                                             'serving',
                                             'claim',
                                             'creativity',
                                             'entire',
                                             'outstanding',
                                             'successfully',
                                             'excited',
                                             'conducting',
                                             'actively',
                                             'input',
                                             'realworld',
                                             'gain',
                                             'principal',
                                             'onsite',
                                             'towards',
                                             'selection',
                                             'accelerate',
                                             'among',
                                             'presenting',
                                             'output',
                                             'worldwide',
                                             'generous',
                                             'channel',
                                             'video',
                                             'special',
                                             'taking',
                                             'advertising',
                                             'prediction',
                                             'selected',
                                             'custom',
                                             'posting',
                                             'title',
                                             'accountability',
                                             'corporation',
                                             'especially',
                                             'tracking',
                                             'target',
                                             'industrial',
                                             'advantage',
                                             'secure',
                                             'performs',
                                             'game',
                                             'transportation',
                                             'five',
                                             'founded',
                                             'art',
                                             'survey',
                                             'format',
                                             'equipment',
                                             'even',
                                             'specialist',
                                             'creates',
                                             'situation',
                                             'simple',
                                             'nation',
                                             'safe',
                                             'shared',
                                             'delivers',
                                             'step',
                                             'ecommerce',
                                             'visual',
                                             'intelligent',
                                             'applies',
                                             'sustainable',
                                             'dod',
                                             'topic',
                                             'personnel',
                                             'dont',
                                             'assignment',
                                             'welcome',
                                             'willing',
                                             'phase',
                                             'preference',
                                             'combine',
                                             'accessible',
                                             'defined',
                                             'back',
                                             'specifically',
                                             'budget',
                                             'evaluating',
                                             'either']))) + ds_skills_combined
    
    stop_words = nltk.corpus.stopwords.words('english') + additional_stopwords + ds_cred_terms + ds_prof_skill_terms + ds_soft_skill_terms + ds_tech_skill_terms
    
    # normalize, split and lowercase the parsed text
    text = (unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore').lower())
    words = re.sub(r'[^\w\s]', '', text).split()
    
    # initialize lemmatizer and execute lemmatization
    wnl = nltk.stem.WordNetLemmatizer()
    terms_for_nlp = [wnl.lemmatize(word) for word in words if word not in stop_words]
    
    # execute post-lemmatization stopword removal to drop unnecessary lemma
    terms_for_nlp = [x for x in terms_for_nlp if x not in additional_stopwords]
    
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
    value_list = ['data', 'science', 'python']
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
# NEED TO DECONFLICT SKILL LISTS AND STOPWORD LISTS
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
#     'collaborate' and 'collaboratively' and 'work together' and 'work closely'; all numbers (1 vs one);
#     'cloud' and 'cloudbased'; 'gcp' into google cloud platform;  'sa' back into 'sas'; scrub the alphabetical lists;
#     'speech recognition' to 'nlp'; 'ai' 'ml' and 'aiml'; 'latest', 'newest', 'cutting edge'; 'recommender' and 'recommendation
#     in the systems;'making decison' bigram to 'decisionmaking'; 'approach' and 'method' and all cousins; 'principle', 'method, apprach, etc.
#     'experience' and 'experienced'; ETL with extract transform and load; 
# consider a swarm plot for ....something, with job title or skill along the x_axis and some count/value along the y-axis,
  # maybe count of ds skills FOR THE UNICORN INDEX; yes, count the number of skills cited in each job listing, parsed by job title (which
  # has been collapsed and simplified)
# will need to make an index of key skills based on the n_gram results
# create a functionality to count how many jobs cite a specifc term I searched; probably just search the series with lov=c
# really need to grind out all stop words that aren't relevant
# can break skill lists into why/how/what subsets later
# think about a graphic showing a tree from a key word, like 'experience' linking to the highest bigrams on the right

def utilities():
    
    # script for cleaning up the n_grams series and converting it to a list
    strip_end = [x[:-3] for x in n_grams.index]
    clean_ngram_list = [x[2:] for x in strip_end]


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
                 '20',
                 '35',
                 'advance',
                 'advanced',
                 'administration',
                 'accredited',
                 'analyst',
                 'analytics',
                 'associate',
                 'bachelor',
                 'biology',
                 'biostatistics',
                 'business',
                 'capability',
                 'certification',
                 'clearance',
                 'college', 
                 'collegeuniversity',
                 'combination',
                 'computer',
                 'computational', 
                 'data',
                 'degree',
                 'demonstrated', 
                 'demonstrates',
                 'discipline',
                 'economics',
                 'education',
                 'electrical',
                 'engineer',
                 'engineering',
                 'equivalent',
                 'experience',
                 'experienced',
                 'expert',
                 'field',
                 'graduate',
                 'handson',
                 'high', 
                 'higher',
                 'industry',
                 'knowledge',
                 'linkedin',
                 'major',
                 'master',
                 'mathematics', 
                 'military', 
                 'operation', 
                 'peerreviewed', 
                 'phd',
                 'physic',
                 'physical',
                 'portfolio',
                 'practical',
                 'prior',
                 'professional',
                 'professionally',
                 'proficiency',
                 'proven',
                 'publication',
                 'quantitative', 
                 'record',
                 'related',
                 'relevant',
                 'research',
                 'researcher',
                 'school',
                 'science',
                 'scientist',
                 'security', 
                 'service',
                 'social',
                 'software',
                 'solid', 
                 'statistic',
                 'statistics',
                 'statistician',
                 'stem',
                 'system', 
                 'technical',
                 'track',
                 'training',
                 'tssci',
                 'understand',
                 'understanding',
                 'understood',
                 'university',
                 'work',
                 'working',
                 'year']  

ds_tech_skill_terms = ['ab',
                       'access',
                       'accurate', 
                       'accurately',
                       'accuracy',
                       'adopter',
                       'agile',
                       'ai',
                       'aidriven',
                       'aiml',
                       'airflow',
                       'advanced',
                       'algebra',
                       'algorithm',
                       'amazon',
                     'analysis',
                     'analytical',
                     'analytic',
                     'analytics',
                     'analyze',
                     'analyzing',
                     'anomaly', 
                     'apache', 
                     'apis',
                     'application',
                     'applied', 
                     'applying',
                     'approach',
                     'architect',
                     'architecting',
                     'architecture',
                     'artificial',
                     'asaservice',
                     'automated',
                     'automation', 
                     'automate',
                     'aws', 
                     'azure',
                     'bayes',
                     'bi',
                     'big',
                     'blockchain',
                     'boosting',
                     'build', 
                     'c',
                     'causal', 
                     'center',
                     'classification',
                     'cleaning',
                     'cleansing',
                     'cloud', 
                     'cloudbased',
                     'cluster',
                     'clustering',
                     'code', 
                     'coding',
                     'collect',
                     'collection',
                     'cognitive',
                     'complex',
                     'computer',
                     'computing',
                     'concept',
                     'conceptual',
                     'confidence',
                     'confluence',
                     'continuous',
                     'control',
                     'cs',
                     'cutting', 
                     'cuttingedge',
                     'dashboard',
                     'dask',
                     'data',
                     'datadriven',
                     'dataset',
                     'datasets',
                     'database',
                     'decision', 
                     'deep',
                     'demonstrate',
                     'deploy',
                     'design', 
                     'designed',
                     'designing',
                     'detection',
                     'develop',
                     'developed',
                     'develops',
                     'developer',
                     'developing',
                     'development',
                     'dimensionality', 
                     'distributed', 
                     'docker',
                     'django',
                     'early', 
                     'ec2',
                     'ecosystem',
                     'edge',
                     'emerging',
                     'endtoend',
                     'engine',
                     'engineer',
                     'engineering',
                     'enterprise',
                     'environment',
                     'etl',
                     'evaluate',
                     'evaluation',
                     'excel',
                     'experiment',
                     'experimental', 
                     'experimentation',
                     'exploration',
                     'exploratory',
                     'extract', 
                     'extraction',
                     'fastapi',
                     'feature', 
                     'flask',
                     'flow',
                     'forecasting',
                     'forest',
                     'framework',
                     'fraud', 
                     'full', 
                     'fundamental',
                     'generation',
                     'gcp',
                     'git',
                     'google',
                     'gradient', 
                     'graph',
                     'hadoop',
                     'hardware',
                     'hidden',
                     'hive',
                     'html', 
                     'hypothesis', 
                     'image',
                     'imagery',
                     'implement',
                     'implementation',
                     'inference',
                     'information',
                     'infrastructure',
                     'ingestion',
                     'integration',
                     'integrity',
                     'intelligence',
                     'interface',
                     'iterating',
                     'java',
                     'jira', 
                     'jupyter', 
                     'kera',
                     'kubernetes',
                     'lake',
                     'language',
                     'large',
                     'largescale',
                     'latest',
                     'learning',
                     'library',
                     'lifecycle',
                     'linear', 
                     'linux',
                     'load',
                     'logical',
                     'logistic', 
                     'looker',
                     'machine',
                     'maintain',
                     'manipulate',
                     'manipulation',
                     'math',
                     'mathematical',
                     'mathematics',
                     'matlab',
                     'measure',
                     'measurement',
                     'method',
                     'methodology',
                     'metric',
                     'microsoft', 
                     'mine',
                     'mining',
                     'ml',
                     'model',
                     'modeler',
                     'modeling',
                     'modern',
                     'naive', 
                     'natural',
                     'network',
                     'next',
                     'neural', 
                     'nlp',
                     'nosql',
                     'notebook',
                     'numpy',
                     'object',
                     'objectoriented',
                     'office',
                     'open', 
                     'optimal',
                     'optimize',
                     'optimization',
                     'optimizing',
                     'oracle',
                     'panda', 
                     'pattern', 
                     'pipeline',
                     'pivot', 
                     'platform',
                     'power',
                     'predict',
                     'predictive',
                     'prescriptive',
                     'principle',
                     'probability',
                     'problem',
                     'procedure',
                     'process',
                     'processing',
                     'proficient',
                     'programming', 
                     'python',
                     'pytorch',
                     'quantitative',
                     'quantum',
                     'query',
                     'relational', 
                     'r',
                     'rd',
                     'random', 
                     'raw',
                     'recognition',
                     'recommendation', 
                     'recommender',
                     'redshift', 
                     'reduction',
                     'regression',
                     'reinforcement',
                     'relational',
                     'relationship',
                     'remote', 
                     'research',
                     'rest', 
                     'review',
                     'robotics',
                     's3', 
                     'sa',
                     'sagemaker',
                     'scala',
                     'scale',
                     'scalable',
                     'scenario',
                     'science',
                     'scientific',
                     'sciencebased',
                     'scripting',
                     'scikitlearn',
                     'scipy',
                     'sensing',
                     'sentiment',
                     'series',
                     'server',
                     'service',
                     'set',
                     'shell',
                     'signal', 
                     'simulation',
                     'skill',
                     'snowflake',
                     'software',
                     'source',
                     'sp',
                     'spark',
                     'speech', 
                     'spss',
                     'sql',
                     'stack',
                     'statistic',
                     'statistics',
                     'statistical',
                     'statistician',
                     'stateoftheart',
                     'store',
                     'structured',
                     'supervised', 
                     'system',
                     'table',
                     'tableau',
                     'tech',
                     'technical',
                     'technique',
                     'technology',
                     'tensorflow', 
                     'test',
                     'testing',
                     'text',
                     'time', 
                     'timely',
                     'tool',
                     'train',
                     'training',
                     'transformation',
                     'troubleshooting',
                     'tree',
                     'trend',
                     'uncover', 
                     'unstructured',
                     'user',
                     'unsupervised',
                     'validation',
                     'vector',
                     'version', 
                     'visio',
                     'visualization',
                     'vmware',
                     'volume',
                     'warehouse',
                     'warehousing',
                     'watson',
                     'web',
                     'wrangling']    

ds_soft_skill_terms = ['ad', 
                       'ability',
                       'agile',
                       'ambiguity',
                       'ambiguous',
                       'attention', 
                       'attitude',
                       'audience',
                     'best',
                     'boundary',
                     'business',
                     'cause',
                     'challenging', 
                     'clear', 
                     'clearly', 
                     'closely',
                     'cognitive',
                     'collaborate',
                     'collaborative',
                     'collaboration',
                     'collaboratively',
                     'collaborating',
                     'communicate',
                     'communicating', 
                     'communication',
                     'complex',
                     'concise',
                     'concisely',
                     'conclusion',
                     'confluence',
                     'connect',
                     'consulting',
                     'consultant',
                     'continuous', 
                     'contribute',
                     'contributor',
                     'creative',
                     'critical', 
                     'crossfunctional',
                     'crossfunctionally',
                     'curious',
                     'curiosity',
                     'deadline',
                     'decision', 
                     'deliver',
                     'delivering',
                     'detail',
                     'detailed',
                     'draw', 
                     'dynamic',
                     'efficient',
                     'efficiency',
                     'efficiently',
                     'effectively',
                     'environment',
                     'ethic',
                     'excellent',
                     'excellence',
                     'exercise', 
                     'experience',
                     'experienced',
                     'explain', 
                     'fast', 
                     'fastpaced',
                     'finding',
                     'flexible',
                     'flexibility',
                     'focus',
                     'git',
                     'guidance',
                     'hard',
                     'high', 
                     'highly', 
                     'hoc', 
                     'idea',
                     'jira', 
                     'impact',
                     'impactful',
                     'improvement',
                     'independently',
                     'individual', 
                     'influence',
                     'initiative',
                     'innovation',
                     'innovate',
                     'innovator',
                     'intelligence',
                     'intellectual', 
                     'intellectually',
                     'interact',
                     'interdisciplinary',
                     'interpersonal',
                     'interpret',
                     'judgment',
                     'learn',
                     'level',
                     'making',
                     'management',
                     'meet',
                     'meeting',
                     'member',
                     'minimal', 
                     'motivated',
                     'multidisciplinary',
                     'nontechnical', 
                     'novel',
                     'oral',
                     'organization',
                     'oriented',
                     'paced',
                     'passion',
                     'passionate',
                     'people',
                     'perspective',
                     'player',
                     'positive', 
                     'powerpoint',
                     'practice',
                     'present', 
                     'presentation',
                     'priority',
                     'prioritizing',
                     'prioritize',
                     'proactively',
                     'problem', 
                     'problemsolving',
                     'push', 
                     'quality',
                     'question',
                     'quickly',
                     'read',
                     'report',
                     'respect',
                     'root', 
                     'selfstarter',
                     'simultaneously',
                     'skill',
                     'solve',
                     'solver',
                     'solving',
                     'speed',
                     'strong',
                     'structure',
                     'study',
                     'supervision',
                     'team',
                     'technical',
                     'thinking',
                     'thorough',
                     'time', 
                     'together',
                     'unique',
                     'user', 
                     'value',
                     'verbal',
                     'verbally',
                     'willingness', 
                     'write',
                     'writing',
                     'written',
                     'word',
                     'work',
                     'workflow',
                     'working']  

ds_prof_skill_terms = ['ability',
                       'actionable', 
                       'acumen',
                       'advisor',
                       'agile',
                       'architecting',
                       'build',
                       'business',
                       'career', 
                       'case',
                       'challenge',
                       'change', 
                       'chosen',
                       'client',
                       'complex', 
                       'cross', 
                       'crossfunctional',
                       'crossfunctionally',
                       'customer', 
                       'data',
                       'dataset',
                       'datasets',
                       'decision', 
                       'decisionmaking',
                       'deep', 
                       'deliver', 
                       'delivery',
                       'deliverable',
                       'delivering',
                       'demonstrated', 
                       'decision',
                       'development',
                       'difference',
                       'differentiated',
                       'digital', 
                       'direct',
                       'director',
                       'domain',
                       'drive',
                       'driven',
                       'earn', 
                       'efficiency',
                       'end', 
                       'engagement',
                       'entrepreneurial',
                       'everchanging',
                       'evolving',
                       'executive',
                       'experience',
                       'experienced',
                       'expert',
                       'expertise',
                       'feedback',
                       'generate',
                       'focus',
                       'functional',
                       'governance',
                       'growth',
                       'help',
                       'helping',
                       'identify', 
                       'identifying',
                       'impact',
                       'impactful',
                       'improve',
                       'improvement',
                       'inform',
                       'innovative',
                       'insight',
                       'interdisciplinary',
                       'knowledge',
                       'lead',
                       'leader',
                       'leading',
                       'leadership',
                       'line',
                       'make', 
                       'maker',
                       'manage',
                       'management',
                       'managing',
                       'market',
                       'marketing',
                       'matter',
                       'meet',
                       'member',
                       'mentor',
                       'mentoring',
                       'mentorship',
                       'mission',
                       'multidisciplinary',
                       'need',
                       'objective',
                       'operating',
                       'operational', 
                       'opportunity',
                       'outcome',
                       'owner',
                       'ownership',
                       'paper',
                       'partner',
                       'peer', 
                       'problem',
                       'problemsolving',
                       'process',
                       'product',
                       'professional',
                       'program',
                       'project', 
                       'proposal',
                       'prototype,'
                       'prototyping',
                       'proven',
                       'question',
                       'rapid', 
                       'record',
                       'recommendation',
                       'requirement',
                       'review',
                       'risk',
                       'sale',
                       'science',
                       'service',
                       'sigma',
                       'six', 
                       'skill',
                       'solution',
                       'solve',
                       'solving',
                       'stakeholder',
                       'story',
                       'strategically',
                       'strategy',
                       'stream',
                       'subject',
                       'success',
                       'team',
                       'technical',
                       'tell', 
                       'think', 
                       'thought',
                       'track',
                       'transform',
                       'transformation',
                       'trust',
                       'trusted', 
                       'understanding',
                       'use',
                       'user',
                       'value',
                       'white', 
                       'win',
                       'work',
                       'working',
                       'workstreams'] 

ds_skills_combined = ds_cred_terms + ds_tech_skill_terms + ds_soft_skill_terms + ds_prof_skill_terms

# for x in ds_prof_skill_terms:
#     print(x, end=', ')









# execute cleaning and field parsing
df_raw        = load_and_concat_csvs(csv_path)
calculate_raw_csv_stats(df_raw)
df_clean      = clean_raw_csv(df_raw)
df            = parse_date_scraped_field(df_clean)
series_of_interest = df['job_description']
terms_for_nlp  = clean_for_nlp(series_of_interest)
visualize_indeed_data(df)

# execute nlp   NEED TO DECONFLICT SKILL LISTS AND STOPWORD LISTS
n_gram_count = 1
n_gram_range_start, n_gram_range_stop  = 0, 200 # 3900, 4000 # NEXT - advance the range
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


