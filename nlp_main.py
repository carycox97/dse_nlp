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
    print('***** Data Ingest Statistics *****')
    print(f'Records imported: {df_raw.shape[0]} \n')
    print(f'Unique job titles: {df_raw.job_title.nunique()} \n')
    print(f'Nulls are present:\n{df_raw.isna().sum()} \n')
    print(f'Records missing job_title field: {(df_raw.job_title.isna().sum() / df_raw.shape[0] * 100).round(3)}%')
    print(f'Records missing job_Description field: {(df_raw.job_Description.isna().sum() / df_raw.shape[0] * 100).round(3)}% \n')
    print('***** Data Cleaning Statistics *****\n')
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
                                            'effort', 'define', 'conduct', 'potential', 'used', 'patient', 'inc',
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
                                            'exceptional', 'energy', 'aspect', 'expected', 'discovery', 'setting', 'material',
                                            'produce', 'see', 'improving', 'academic', 'startup', 'exciting', 'recognized',
                                            'contribution', 'answer', 'creation', 'play', 'robust', 'payment', 'integrate',
                                            'class', 'explore', 'overall', 'establish', 'move', 'integrated', 'three', 'desire',
                                            'obtain', 'pricing', 'regular', 'maintenance', 'interested', 'along', 'increase',
                                            'utilize', 'manufacturing', 'built', 'acquisition', 'engage', 'thats', 'submit',
                                            'roadmap', 'coordinate', 'prepare', 'utilizing', 'foundation', 'encourage', 'cycle',
                                            'vendor', 'become', 'device', 'smart', 'ass', 'effectiveness', 'mean',
                                            'adoption', 'located', 'preferably', 'division', 'mindset', 'scope', 'exposure',
                                            'progress', 'transforming', 'lab', 'enabling', 'interaction',  'look', 'promote',
                                            'example', 'foster', 'specification', 'associated', 'daily', 'realtime', 'campaign',
                                            'independent', 'storage', 'competency', 'website', 'medicine', 'incentive', 'follow',
                                            'facility', 'coverage', 'validate', 'continuously', 'defining', 'bank', 'serving',
                                            'claim', 'creativity', 'entire', 'outstanding', 'successfully', 'excited', 'conducting',
                                            'actively', 'input', 'realworld', 'gain', 'principal', 'onsite', 'towards', 'selection',
                                            'accelerate', 'among', 'presenting', 'output', 'worldwide', 'generous', 'channel',
                                            'video', 'special', 'taking', 'advertising', 'selected', 'custom', 'posting', 'title',
                                            'accountability', 'corporation', 'especially', 'tracking', 'target', 'industrial',
                                            'advantage', 'secure', 'performs', 'game', 'transportation', 'five', 'founded', 'art',
                                            'survey', 'format', 'equipment', 'even', 'specialist', 'creates', 'situation',
                                            'simple', 'nation', 'safe', 'shared', 'delivers', 'step', 'ecommerce', 'visual',
                                            'intelligent', 'applies', 'sustainable', 'dod', 'topic', 'personnel', 'dont',
                                            'assignment', 'welcome', 'willing', 'phase', 'preference', 'combine', 'accessible',
                                            'defined', 'back', 'specifically', 'budget', 'evaluating', 'either', 'encourages',
                                            'given', 'rule', 'established', 'gathering', 'familiar', 'connected', 'recommend',
                                            'update',  'transfer', 'context', 'could', 'ensures',  'cancer', 'expect',
                                            'definition', 'mind', 'gather', 'environmental', 'strongly', 'enables', 'skilled',
                                            'pharmaceutical', 'preparation',  'usa', 'single', 'highquality', 'turn', 'submission',
                                            'laboratory', 'usage', 'ca', 'serf', 'posted', '25', 'allows', 'deloitte', 'close',
                                            'kafka', 'brings', 'headquartered', '40', 'sophisticated', 'distribution', 'hr',
                                            'handle', 'stage', 'connection', 'streaming', 'four', 'traditional', 'powerful',
                                            'marketplace', 'specialized', 'kpis', 'electronic', 'capacity', 'file',
                                            'consistently', 'nature', 'valued', 'informed', 'american', 'oversight', 'valuable',
                                            'spectrum', 'campus', 'highest', 'desirable', 'exemption', 'verification', 'supported',
                                            'greater', 'reduce',  'align', 'broader', 'integrating', 'entity', 'succeed', 
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
                                            'organized', 'hear', 'easy', 'superior', 'tuning', 'nearly', 'promotes',
                                            'satisfaction', 'unlock', 'behind', 'screening', 'begin', 'inquiry',  'page',
                                            'particularly', 'bestinclass', 'demonstrating', 'element', 'ask', 'understands',
                                            'availability', 'oncology', 'specialization', 'empowering', 'criterion', 'loan',
                                            'pioneering', 'advancing', 'exploring', 'aptitude', 'stand', 'institutional', 
                                            'comply', 'respond', 'derive', 'message',  'easily', 'recent', 'journal', 'lending',
                                            'networking', 'continually', 'workshop', 'economy', 'completed', 'choose',
                                            'utilization', 'alongside', 'demonstrable',  'administrative', 'named', 'created',
                                            'last', 'dream', 'requisition', 'invest', 'section', 'apple', 'combining', 'committee',
                                            'held', 'authority', 'deal', 'managed', 'contributes', 'daytoday', 'iot', 'toward',
                                            'achieving', 'learner', 'spirit', 'fee', 'agent', 'certain',  'cc', 'transition',
                                            'roadmaps', 'emr', 'champion', 'virginia', 'region', 'failure', 'establishing',
                                            'comfort', 'eager',  'combined', 'contingent', 'instruction', 'powered', 'utility',
                                            'functionality', 'worked', 'ideally', 'coordination', 'intended', 'underlying',
                                            'proper', 'operates', 'refine', 'unable', 'represent', 'evolve', 'might', 'advocate',
                                            'youve', 'mobility', 'wealth', 'aligned', 'found', 'known', 'funding', 'embedded',
                                            'fostering', 'fuel', 'intermediate',  'awardwinning', 'often', 'tomorrow', 'carry',
                                            'synthesize', 'smarter', 'quo', 'scalability', 'grant', 'useful', 'routine',
                                            'workload', '200', 'foundational', 'reusable', 'retrieval', 'talk', 'incumbent',
                                            'pursue', 'attribute', 'retailer', 'span', 'disabled', 'board', 'face', 'zeta', 'co',
                                            'met', 'validating', 'shaping', 'tactical', 'sample', 'ambitious', 'hub', 'aid',
                                            'presence', 'command', 'subsidiary', 'participation', 'anticipate', 'therefore',
                                            'supplier', 'enablement', 'backed', 'rating', 'advise', 'preparing', 'little', 
                                            'accessibility', 'incorporate', 'beginning', 'car', 'everywhere', 'ethical',
                                            'supportive', 'departmental', 'newly', 'autonomy', 'still', 'varying', 'sustainability',
                                            'texas', 'improved', 'included', 'macro', 'approximately', 'difficult', 'interpreting',
                                            'inhouse', 'generating', 'relation', 'fund', 'already', 'implemented', 'highlevel',
                                            'huge', 'exchange', 'inspired', 'net', 'legacy', 'fl', 'continues', 'premium', 
                                            'realize', 'covered', 'reality', 'verify', 'configuration', 'llc', 'physician',
                                            'accomplish', 'headquarters', 'breadth', 'membership', 'trading', 'visibility',
                                            'room', 'migration', 'head', 'parent', 'true', 'pursuit', 'spotify', '13',
                                            'producing', 'md', 'ultimately', 'ever', 'appointment', 'engaged', 'craft',
                                            'ambition', 'fellow', 'fastest', 'yet', 'dive', 'specialty', 'approval', 'paypal',
                                            'main', 'prospective', 'affordable', 'affect', 'feed', 'tv', 'expanding', 'quick',
                                            'continuing', 'completing', 'frequently', 'connecting', 'investigate', 'audio',
                                            'safer', 'display', 'focusing', 'decade', 'liaison', 'onboarding', 'grade', 'enabled',
                                            'ideation', 'led', 'worker', 'transcript', 'participating', 'article', 'ranging',
                                            'rely', 'bold', 'equality', 'fashion', 'integral', 'requested', 'evolution', 
                                            'appropriately', 'valid', 'advice', 'adhere', 'creatively', 'discus', 'exception',
                                            'actuarial', 'collective', 'coordinating', 'addressing', '9', 'motor', 'assisting',
                                            'biomedical', 'targeting', 'screen', 'maximum', 'cultural', 'promise',
                                            'initial', 'wayfair', 'living', 'ops', 'concern', 'cell', 'music', 'stipend', 'invite',
                                            'determined', 'profiling', 'eye', 'multitask', 'merit', 'represents', 'assessing',
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
                                            'streamline', 'ranked', 'reserve', 'ny', 'nj', 'conflict', 'initiate',  'atmosphere',
                                            'pace', 'chemical', 'therapy', 'robotic', 'blog', 'targeted', 'searching', 'extend',
                                            'ten', 'recently', 'staffing', 'measuring', 'sop', 'filled', 'exhibit', 'ii',
                                            'transparent', 'representation', 'entry', 'opinion', 'electric', 'speaking',
                                            'consultation', 'planet', 'consumption', 'climate', 'consistency', 'sit', 'sensitive',
                                            'picture', 'dollar', 'fastgrowing', 'arrangement', 'regional', 'closing', 'enjoys',
                                            'knowledgeable', 'attribution', 'break', 'fitness', 'europe', 'thrives', 'increased',
                                            'milestone', 'incredible', 'everyday', 'researching', 'foreign', 'minority',
                                            'nonprofit', 'construction', 'simplify', 'suitable', 'pharmacy', 'whose', 'mastery',
                                            'exempt', 'item', 'cro', 'native', 'intuitive', 'consult', 'participant', 'prevention',
                                            'official', 'conversion', 'fact', 'ahead', 'diagnostic', 'healthier', 'deloittes',
                                            'moment', 'gsk', 'philadelphia', 'frontend', 'barrier', 'intersection', 'determining',
                                            'directed', 'inperson', 'trade',  'forbes', 'enthusiastic', 'fill', 'enhanced',
                                            'prepares', 'az', 'correct', 'importance', 'significantly', 'educate', 'reputation',
                                            'here', 'effect', 'perfect', 'pet', 'tight', 'guiding', 'uncertainty', 'side',
                                            'scheduling', 'diagnostics', 'merchant', 'interacting', 'labor', 'currency', 'fintech',
                                            'street', 'seeker', 'evaluates', 'standardization', 'reading', 'investing', 'workday',
                                            'circle', 'mode', 'supervisor', 'extremely', 'receiving', 'organizing', 'magazine',
                                            'outlined', 'commensurate', 'florida', 'window', 'isnt', 'factory', 'district',
                                            'parttime', 'presented', 'pressure', 'occasional', 'pas', 'utilizes', 'calculation',
                                            'promoting', 'extended', 'dna', 'quarterly', 'snack', 'deeper', 'courage', 'schema',
                                            'acceptance', 'qa', 'perception', 'mitigate', 'nasdaq', 'attend', 'prospect', '24',
                                            'arm', 'air', 'shipping', 'fleet', 'score', 'penn', 'oh', 'gym', 'babs', '11', '60',
                                            'era', 'mitigation', 'introduction', 'civilian', '1st',  'surface', 'league', 'stored',
                                            'resulting', 'setup', 'lighthouse', 'narrative', 'vital', 'going', 'treat',
                                            'substantial', 'comparable', 'lift', 'seattle', 'e', 'highperforming', 'easier', '18',
                                            'urgency', 'delight', 'tailored', 'abreast', 'progressive', 'vibrant', 'book',
                                            'vertical', 'astrazeneca', 'tangible', 'loyalty', 'mechanism', 'lieu', 'competition',
                                            'nationwide', 'invent', 'owning', 'light', 'discovering', 'hope', 'fda', 'empathy',
                                            'joint', 'lambda', 'pushing', 'generally', 'follows', 'touch', 'spring', 'relevance',
                                            'water', 'discretion', 'extent', 'acquire', 'respected', 'positively', 'corp',
                                            'transformational', 'walk', 'qualify', 'owned', 'defines', 'payer', 'guided',
                                            'popular', '2019', 'standing', 'nyse',  'fulfill', 'adaptable', 'cash', 
                                            'moderate', 'dell', 'customized', 'designated', 'highlight', 'pc', 'selecting',
                                            'messaging', 'outreach', 'pursuing',  'ceo', 'gi', 'agenda', 'elevate', 'historical',
                                            'transactional', 'original', 'sometimes', 'buy', 'stability', 'capgemini', 'modify',
                                            'consultative', 'incident', 'dallas', 'advertiser', 'ci', 'publishing', 'vp',
                                            'tackling', 'apart', 'v', 'tolerate', 'feasibility', 'uptodate', 'productiongrade',
                                            'roi', 'collectively', 'supplement', 'diving', 'guidehouse', 'tax', 'sdlc',
                                            'parameter', 'lasting', 'philosophy', 'balancing', 'cant', 'auto', 'holding',
                                            'benchmark', 'cdisc', 'adjustment', 'thoughtful', 'publish', 'telework', 'fidelity',
                                            '14', 'shop', 'november', 'medicare', 'considers', 'modernization', 'lower', 'near',
                                            'commission', 'outlook', 'surveillance', 'mi', 'impacting', 'aimed', 'navy',
                                            'affiliate', 'duration', 'leveraged', 'demonstration', 'adept', 'blend', 'norm',
                                            'maintainable', 'buying', 'adherence', 'association', 'intent', 'underwriting',
                                            'interprets', 'clinician', 'shopper', 'imagine', 'confidentiality', 'happen',
                                            'circumstance', 'specializing', 'curriculum', 'denver', 'maryland', 'rare',
                                            'propensity', 'considering', 'dimension', '23', 'index', 'negotiation',
                                            'inspiring', 'speaker', 'crafting', 'tier', 'motion', 'academia', 'truth',
                                            'road', 'motivate', 'briefing', 'forum', 'creator', 'fedex', 'heavily', 'constant',
                                            'upload', 'pilot', 'theyre', 'allocation', 'traveler', 'manuscript', 'shopping',
                                            'limit', 'devise', 'provision', 'forth', 'serious', 'turning', 'catalog', 'competence',
                                            'understandable', 'pound', 'phoenix', 'scoring', 'coming', 'enrich', 'seven',
                                            'disruptive', 'behalf', 'covering', 'individually', 'welcoming', 'attendance',
                                            'strives', 'bar', 'omnichannel', 'afraid', 'brightest', 'holder', 'classical',
                                            'valuation', 'optional', 'obsessed', 'temporary', 'massachusetts', 'normal',
                                            'expansion', 'dig', 'grown', 'indeed', 'protecting', 'annually', 'club', 'consultancy',
                                            'frequent', 'seamlessly', 'epic', 'accelerating', 'disclaimer', 'questionnaire',
                                            'telephone', 'obtaining', 'enough', 'aggregation', 'ultimate', 'sell', 'maturity',
                                            'anyone', 'fastestgrowing', 'unparalleled', 'say', 'onthejob', 'patent', 'employing',
                                            'ba', 'reflect', 'visible', 'recommended', 'linguistics', 'efficacy', 'uk', 'ui',
                                            'aviation', 'mandatory', 'graphical', 'simply', 'internship', 'concurrent',
                                            'moody', 'else', 'translates', 'cultivate', 'anccon', 'react', 'cognizant',
                                            'strengthen', 'raise', 'drawing', 'affair', 'internationally', 'door', 'portal',
                                            'seller', 'plant', 'buyer', 'accomplishment', '1000', 'traffic', 'intervention',
                                            'honest', 'specified', 'terabyte', 'executes', 'backlog', 'w', 'published', 'army',
                                            'relates', 'eight', 'recommends', 'park', 'motivation', 'called', 'sitting',
                                            'fulfillment', 'na', 'walmart', 'administrator', 'ac_consulting21', 'verisk', 'spent',
                                            'michigan', 'slack', 'red', 'crime', 'anticipated', 'desk', '5000', 'derived',
                                            'grasp', 'dot', 'west', 'personalize', 'meaning', 'complicated', 'calling', 'bigger',
                                            'reducing', 'seamless', 'module', 'clarity', 'convenient', 'pennsylvania', 'integrates',
                                            'achieved', 'putting', 'rank', 'retaining', 'suggest', 'piece', 'throughput',
                                            'launched', 'loading', 'exist', 'john', 'customercentric', 'relating', 'truck', 
                                            'centered', '90', 'winning', '0', 'facilitation', 'enrollment', 'arise', 'rather',
                                            'inspires', '2018', 'evaluated', 'pro', 'footprint', 'selfdriving', 'doe', 'finish',
                                            'subcontractor', 'scoping', 'telecommunication', 'raised', 'b2b', 'informal', 'wa',
                                            'lifelong', 'caffe', 'ip', 'psychology', 'marketer', 'relentless', 'prefer', 'abuse',
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
                                            'cohesive', 'acceptable', 'almost', 'added', 'cnn', 'ship', 'procurement', 'abstract',
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
                                            'survival', 'kpi', 'try', 'establishes', 'satisfy', '75', 'memory', 'missiondriven',
                                            'aspiration', 'enduring', 'alone', 'separation', 'acquiring', 'recovery', 'intuition',
                                            'competitor', 'stated', 'trillion', 'baseline', 'churn', 'lifestyle', 'incorporating',
                                            'launching', 'acquired', 'erp', 'body', 'tough', 'wider', 'internalexternal',
                                            'wherever', 'addressed', 'latency', 'gas', 'virtually', 'advising', 'black',
                                            'potentially', 'validity', 'voluntary', 'sign', 'pull', 'tactic', 'comprised',
                                            'adjust', 'founding', 'brief', 'resultsoriented', 'respective', 'envision',
                                            'obtained', 'uphold', 'terminology', 'belong', 'diego', 'taken', 'pertaining',
                                            'governing', 'webbased', 'diagram', 'india', 'yield', 'highgrowth', 'insurer',
                                            'carvana', 'engages', 'bot', 'consists', 'ford', 'arizona', 'earned', 'summarizing',
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
                                            'eastern', 'beautiful', 'organisation', 'obligation', 'extra', 'scorecard', 'gps',
                                            'trained', 'presales', 'appreciation', 'nursing', 'furthermore', 'restriction',
                                            'staying', 'obstacle', 'button', 'responsive', 'heritage', 'heard', 'revolution',
                                            'modification', 'impossible', 'influential', 'powering', 'seriously', 'percent',
                                            'household', 'dissemination', 'dl', 'cm', 'packaging', 'strict', 'asking',
                                            'enforcement', 'extreme', 'intern', 'encompassing', 'reproducibility', '19', 'loop',
                                            'triage', 'aligning', 'temporarily', 'youd', 'regulated', 'brain', 'hq', 'tune',
                                            'distance', 'continual', 'representing', 'solely', 'beneficial', 'biopharmaceutical',
                                            'hat', 'instrumentation', 'ey', 'critically', 'fan', 'interacts', 'synthesizing',
                                            'sent', 'nyc', 'qualifying', 'helped', 'hsa', 'centric', 'culturally', 'mexico',
                                            'zero', 'routinely', 'reveal', 'row', 'recommending', 'tasked', 'synthesis', 'trait',
                                            'possessing', 'perceived', 'flink', 'seen', 'merchandising', 'properly', '360',
                                            'count', 'deepen', 'sharp', 'doesnt', 'ex', 'cellular', 'underrepresented',
                                            'discretionary', 'standardized', 'deriving', 'spread', 'determines', 'enrichment',
                                            'limitless', 'nationality', 'consume', 'arent', 'variant', 'contacted', 'zoom',
                                            'independence', 'protein', 'ryder', 'centralized', 'nike', 'unixlinux', 'demographic',
                                            'deemed', 'longer', 'fulfilling', 'proudly', 'positioned', 'whatever', 'attack',
                                            'encouraging', 'gained', 'providence', 'unmatched', 'pocs', 'entrylevel', 'spouse',
                                            'adding', 'redefining', 'ohio', 'sole', 'mn', 'aipowered', 'modernize', 'tiktok',
                                            'confidently', 'asia', 'appreciate', 'fire', 'guidewire', 'police', 'endless',
                                            'conventional', 'improves', 'toptier', 'heshe', 'adp', 'backbone', 'stable',
                                            'registry', 'koverse', 'noise', 'viable', 'lifechanging', 'resiliency', 'bottom',
                                            'spoken', 'remediation', '57', 'iii', 'curve', 'publisher', 'democratize', 'portion',
                                            'restaurant', 'prohibit', 'velocity', 'diligence', 'yr', 'advises', 'venue',
                                            'adapting', 'ssrs', 'bay', 'wireless', 'visually', 'caring', 'linking',
                                            'television', 'ease', 'robot', 'rooted', 'ia', 'awesome', 'observation', 'chewy', 
                                            'shareholder', 'usajobs', 'visitor', 'equipped', 'listener', 'intend', 'star',
                                            'updated', 'alert', 'nonrelational', 'retrieve', 'semiconductor', 'correction',
                                            'licensed', 'anything', 'alexa', 'org', 'genuine', 'prevents', 'ar', 'quantity',
                                            'minimize', 'invitation', 'mclean', 'depends', 'enhances', '400', 'hardest',
                                            'generalized', 'accomplished', 'ticket', 'downtown', 'transformer', 'utilized',
                                            'depend', 'ibmibm', '3d', 'poc', 'challenged', 'writes', 'revolutionary', 'majority',
                                            'allowance', 'ally', 'sustainment', 'parking', 'servicing', 'mac', 'thank',
                                            'mantech', 'suggestion', 'compliant', 'instrumental', 'novartis', 'controlled',
                                            'proofofconcept', 'enterprisewide', 'persona', 'humility', 'columbia', '80', 'ssa',
                                            'securely', 'contractual', 'multiyear']))) + ds_skills_combined
    
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
# NEXT: NEED TO DECONFLICT SKILL LISTS AND STOPWORD LISTS, and create assertion gates to make sure every word is accounted for and every list overlap is accounted for

# finalize bar plot of count of jobs in states
# for visualization: branded for NLP/ML insights 
# find a better mask for the word cloud
# parse the date_scraped field for the parent scrape (e.g., ds, ml, etc.)
# expand stopwords, aggressively
# create searches for key lists
# create list of cloud tech
# word clouds based only on the verbs in the job descriptions
# hold = df[df['job_description'].str.contains('attention to detail')]
# hold = df[df['job_description'].str.contains('|'.join(['passion','collaborate','teamwork','team work', 'interpersonal','flexibility','flexible','listening','listener','listen','empathy','empathetic']))]
# create seaborne charts for n-grams
# add timing
# figure out how to brand-color the word clouds (probably just use the mako colors as my brand colors)
# select favorite sns color pallete, and maybe use that for the branding colors!
# determine optimal sns chat size
# think about grouping/stacking compaisons of n_grams based on job type/title
# consider a function to identify and collapse key terms, like 'scientist' and 'science', 'analytics' and 'analysis', 'algorithm' and 'technique', 'oral' and 'verbal',
#     'model' and 'modeling', 'strong', 'excellent', 'writing' and 'written', 'discipline' and 'field',
#     'collaborate' and 'collaboratively' and 'work together' and 'work closely'; all numbers (1 vs one);
#     'cloud' and 'cloudbased'; 'gcp' into google cloud platform;  'sa' back into 'sas'; scrub the alphabetical lists;
#     'speech recognition' to 'nlp'; 'ai' 'ml' and 'aiml'; 'latest', 'newest', 'cutting edge'; 'recommender' and 'recommendation
#     in the systems;'making decison' bigram to 'decisionmaking'; 'approach' and 'method' and all cousins; 'principle', 'method, apprach, etc.
#     'experience' and 'experienced'; ETL with extract transform and load; 'bachelor' undergraduate'; 'time' and 'timeseries'
#     'hpc' for all high powered computing; "sklearn" and scikitlearn and their variants; 
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
    print(clean_ngram_list)


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
                 'aerospace',
                 'analyst',
                 'analytics',
                 'associate',
                 'bachelor',
                 'biology',
                 'biological',
                 'bioinformatics',
                 'biostatistics',
                 'biotech',
                 'biotechnology',
                 'bsba',
                 'business',
                 'capability',
                 'certificate',
                 'certification',
                 'certified',
                 'chemistry',
                 'clearance',
                 'college', 
                 'collegeuniversity',
                 'combination',
                 'computer',
                 'computational', 
                 'computation',
                 'course',
                 'coursework',
                 'credential',
                 'data',
                 'degree',
                 'demonstrated', 
                 'demonstrates',
                 'diploma',
                 'discipline',
                 'doctoral',
                 'doctorate',
                 'economics',
                 'econometrics',
                 'econometric',
                 'education',
                 'electrical',
                 'engineer',
                 'engineering',
                 'epidemiology',
                 'equivalent',
                 'experience',
                 'experienced',
                 'expert',
                 'field',
                 'fluency',
                 'fluent',
                 'genomic',
                 'genomics',
                 'graduate',
                 'handson',
                 'high', 
                 'higher',
                 'industry',
                 'informatics',
                 'knowledge',
                 'linkedin',
                 'major',
                 'master',
                 'mathematics', 
                 'mba',
                 'military', 
                 'molecular',
                 'neuroscience',
                 'operation', 
                 'peerreviewed', 
                 'phd',
                 'physic',
                 'physical',
                 'polygraph',
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
                 'sci',
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
                 'undergraduate',
                 'university',
                 'work',
                 'working',
                 'year']  

ds_tech_skill_terms = ['ab',
                       'access',
                       'accurate', 
                       'accurately',
                       'accuracy',
                       'activation',
                       'adobe', 
                       'adopter',
                       'adversarial',
                       'aggregate',
                       'aggregating',
                       'agile',
                       'agility',
                       'ai',
                       'aidriven',
                       'aiml',
                       'airflow',
                       'advanced',
                       'algebra',
                       'algorithm',
                       'algorithmic',
                       'alteryx',
                       'amazon',
                     'analysis',
                     'analytical',
                     'analytic',
                     'analytics',
                     'analyze',
                     'analyzes',
                     'analyzing',
                     'anomaly', 
                     'apache', 
                     'api',
                     'apis',
                     'application',
                     'applied', 
                     'applying',
                     'approach',
                     'arcgis',
                     'architect',
                     'architecting',
                     'architecture',
                     'architectural',
                     'artificial',
                     'asaservice',
                     'assumption',
                     'athena',
                     'aurora',
                     'automated',
                     'automation', 
                     'automate',
                     'automating',
                     'autonomously',
                     'aws', 
                     'azure',
                     'bash',
                     'batch',
                     'bayes',
                     'bayesian',
                     'behavioral',
                     'bi',
                     'big',
                     'bigquery',
                     'blockchain',
                     'boosting',
                     'breakthrough',
                     'build', 
                     'c',
                     'cassandra',
                     'causal', 
                     'center',
                     'cicd',
                     'chart',
                     'chatbots',
                     'classification',
                     'classifier',
                     'clean',
                     'cleaning',
                     'cleansing',
                     'cloud', 
                     'cloudbased',
                     'cluster',
                     'clustering',
                     'code', 
                     'coding',
                     'collect',
                     'collected',
                     'collecting', 
                     'collection',
                     'cognitive',
                     'cognos',
                     'compile',
                     'complex',
                     'compute',
                     'computer',
                     'computing',
                     'concept',
                     'conceptual',
                     'confidence',
                     'confluence',
                     'constraint',
                     'container',
                     'containerization',
                     'continuous',
                     'control',
                     'correlation',
                     'crm',
                     'cs',
                     'curation',
                     'cutting', 
                     'cuttingedge',
                     'cyber',
                     'cybersecurity',
                     'dashboard',
                     'dashboarding',
                     'dask',
                     'data',
                     'databricks',
                     'datadriven',
                     'datarobot',
                     'dataset',
                     'datasets',
                     'database',
                     'databased',
                     'debug',
                     'debugging',
                     'decision', 
                     'deep',
                     'demonstrate',
                     'deploy',
                     'design', 
                     'designed',
                     'designing',
                     'designer',
                     'detect',
                     'detection',
                     'develop',
                     'developed',
                     'develops',
                     'developer',
                     'developing',
                     'development',
                     'devops',
                     'dictionary',
                     'differential',
                     'dimensional',
                     'dimensionality', 
                     'distributed', 
                     'docker',
                     'django',
                     'early', 
                     'ec2',
                     'ecosystem',
                     'edge',
                     'elastic',
                     'elasticsearch',
                     'emerging',
                     'empirical',
                     'endtoend',
                     'engine',
                     'engineer',
                     'engineering',
                     'ensemble',
                     'enterprise',
                     'environment',
                     'error',
                     'estimate',
                     'estimation',
                     'etl',
                     'evaluate',
                     'evaluation',
                     'evidence',
                     'excel',
                     'experiment',
                     'experimental', 
                     'experimentation',
                     'exploration',
                     'exploratory',
                     'extract', 
                     'extraction',
                     'extracting',
                     'fastapi',
                     'feature',
                     'filtering',
                     'flask',
                     'flow',
                     'forecast',
                     'forecasting',
                     'forest',
                     'formulation',
                     'framework',
                     'fraud', 
                     'full', 
                     'fullstack',
                     'function',
                     'fundamental',
                     'fusion',
                     'generation',
                     'generated',
                     'generative',
                     'geography',
                     'geographic',
                     'geospatial',
                     'gcp',
                     'git',
                     'github', 
                     'gitlab',
                     'glue',
                     'google',
                     'gpu',
                     'gpus',
                     'gradient', 
                     'graph',
                     'graphic',
                     'groundbreaking',
                     'h2o',
                     'hadoop',
                     'hardware',
                     'hbase',
                     'hidden',
                     'highperformance',
                     'hive',
                     'hpc',
                     'html', 
                     'hypothesis', 
                     'image',
                     'imaging',
                     'imagery',
                     'implement',
                     'implementation',
                     'inference',
                     'informatica',
                     'information',
                     'infrastructure',
                     'ingest', 
                     'ingestion',
                     'integration',
                     'integrity',
                     'intelligence',
                     'interactive',
                     'interface',
                     'interpretation',
                     'interpret',
                     'interpreting',
                     'iterate', 
                     'iterating',
                     'iteration', 
                     'iterative', 
                     'iteratively',
                     'kinesis',
                     'java',
                     'javascript',
                     'jenkins', 
                     'jira', 
                     'json',
                     'julia',
                     'jupyter', 
                     'kera',
                     'kubeflow',
                     'kubernetes',
                     'label',
                     'lake',
                     'language',
                     'large',
                     'largescale',
                     'latest',
                     'leadingedge',
                     'learning',
                     'library',
                     'lifecycle',
                     'linear', 
                     'linux',
                     'literacy', 
                     'load',
                     'logic',
                     'logical',
                     'logistic', 
                     'looker',
                     'loss',
                     'machine',
                     'machinelearning',
                     'maintain',
                     'manipulate',
                     'manipulation',
                     'map',
                     'mapping',
                     'mapreduce',
                     'math',
                     'mathematical',
                     'mathematics',
                     'matlab',
                     'matplotlib', 
                     'matrix',
                     'measure',
                     'measurable',
                     'measurement',
                     'metadata',
                     'method',
                     'methodology',
                     'methodological',
                     'metric',
                     'microservices',
                     'microsoft', 
                     'mine',
                     'mining',
                     'ml',
                     'mlai',
                     'mlflow',
                     'mlops',
                     'model',
                     'modeler',
                     'modeling',
                     'modelling',
                     'modern',
                     'mongodb',
                     'multidimensional',
                     'multivariate',
                     'mxnet',
                     'mysql',
                     'naive', 
                     'natural',
                     'neo4j',
                     'network',
                     'next',
                     'nextgeneration',
                     'neural', 
                     'nlp',
                     'nltk',
                     'nodejs', 
                     'normalization',
                     'nosql',
                     'notebook',
                     'numerical',
                     'numpy',
                     'nvidia',
                     'object',
                     'objectoriented',
                     'office',
                     'ontology',
                     'open', 
                     'opensource',
                     'optimal',
                     'optimize',
                     'optimized',
                     'optimization',
                     'optimizing',
                     'oracle',
                     'orchestration',
                     'outlier',
                     'panda', 
                     'parallel',
                     'pattern', 
                     'pearson',
                     'perl',
                     'petabyte',
                     'pig',
                     'pipeline',
                     'pivot', 
                     'platform',
                     'postgres', 
                     'postgresql', 
                     'power',
                     'powerbi',
                     'predict',
                     'predicting',
                     'predictive',
                     'prediction',
                     'preprocessing',
                     'prescriptive',
                     'presto',
                     'principle',
                     'probability',
                     'probabilistic',
                     'problem',
                     'procedure',
                     'process',
                     'processing',
                     'proficient',
                     'programming', 
                     'python',
                     'pythonr',
                     'pyspark',
                     'pytorch',
                     'qlik',
                     'qlikview',
                     'qualitative',
                     'quantify',
                     'quantitative',
                     'quantum',
                     'query',
                     'relational', 
                     'r',
                     'rd',
                     'random', 
                     'raw',
                     'rdbms',
                     'reason',
                     'reasoning',
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
                     'repeatable',
                     'repository', 
                     'reproducible',
                     'research',
                     'rest', 
                     'retraining',
                     'review',
                     'rigor',
                     'rigorous', 
                     'robotics',
                     'rpa',
                     'ruby',
                     's3', 
                     'sa',
                     'saas',
                     'sagemaker',
                     'sampling',
                     'sap',
                     'satellite',
                     'scala',
                     'scale',
                     'scalable',
                     'scenario',
                     'schema',
                     'science',
                     'scientific',
                     'sciencebased',
                     'script', 
                     'scripting',
                     'scrum',
                     'scikit',
                     'scikitlearn',
                     'scipy',
                     'search',
                     'segment',
                     'segmentation',
                     'semantic',
                     'semistructured'
                     'sensing',
                     'sensor',
                     'sentiment',
                     'series',
                     'server',
                     'service',
                     'set',
                     'shell',
                     'shiny',
                     'signal', 
                     'simulation',
                     'skill',
                     'sklearn',
                     'snowflake',
                     'software',
                     'source',
                     'sp',
                     'spacy',
                     'spark',
                     'spatial',
                     'speech', 
                     'splunk', 
                     'spotfire',
                     'spreadsheet',
                     'sprint',
                     'spss',
                     'sql',
                     'ssis',
                     'stack',
                     'statistic',
                     'statistics',
                     'statistical',
                     'statistician',
                     'stateoftheart',
                     'stata',
                     'store',
                     'structured',
                     'supervised', 
                     'svm',
                     'system',
                     'table',
                     'tableau',
                     'tech',
                     'technical',
                     'technically',
                     'technique',
                     'technology',
                     'technological',
                     'tensorflow', 
                     'terabyte',
                     'teradata',
                     'test',
                     'testing',
                     'text',
                     'theoretical',
                     'theory',
                     'time', 
                     'timeseries',
                     'timely',
                     'tool',
                     'tooling',
                     'train',
                     'training',
                     'translating',
                     'transformation',
                     'transformative',
                     'translation',
                     'troubleshoot',
                     'troubleshooting', 
                     'tree',
                     'trend',
                     'uncover', 
                     'unix',
                     'unstructured',
                     'user',
                     'unsupervised',
                     'ux',
                     'validation',
                     'variable',
                     'vba',
                     'vector',
                     'version', 
                     'visio',
                     'visualize',
                     'visualization',
                     'visualizing',
                     'vmware',
                     'volume',
                     'warehouse',
                     'warehousing',
                     'watson',
                     'web',
                     'wrangling',
                     'xgboost']    

ds_soft_skill_terms = ['ad', 
                       'adhoc',
                       'adapt',
                       'adaptive',
                       'ability',
                       'agile',
                       'ambiguity',
                       'ambiguous',
                       'articulate',
                       'assumption',
                       'attention', 
                       'attitude',
                       'audience',
                       'authenticity',
                     'best',
                     'boundary',
                     'business',
                     'cando',
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
                     'collaborates',
                     'collaborator',
                     'communicate',
                     'communicates',
                     'communicating', 
                     'communication',
                     'communicator',
                     'complex',
                     'complexity',
                     'concise',
                     'concisely',
                     'conclusion',
                     'confident',
                     'confluence',
                     'connect',
                     'consulting',
                     'consultant',
                     'continuous', 
                     'contribute',
                     'contributor',
                     'coworkers',
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
                     'detailoriented',
                     'document',
                     'documenting',
                     'draw', 
                     'dynamic',
                     'efficient',
                     'efficiency',
                     'efficiently',
                     'effectively',
                     'energetic', 
                     'enthusiasm',
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
                     'faster',
                     'finding',
                     'flex',
                     'flexible',
                     'flexibility',
                     'focus',
                     'forwardthinking',
                     'git',
                     'guidance',
                     'hard',
                     'hardworking',
                     'high', 
                     'highly', 
                     'hoc', 
                     'holistic',
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
                     'innovating',
                     'inquisitive',
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
                     'listen',
                     'listening', 
                     'literature',
                     'making',
                     'management',
                     'manages',
                     'managerial',
                     'meet',
                     'meeting',
                     'member',
                     'minimal', 
                     'motivated',
                     'multidisciplinary',
                     'multitasking',
                     'nontechnical', 
                     'novel',
                     'oral',
                     'organize',
                     'organization',
                     'oriented',
                     'paced',
                     'partnering',
                     'passion',
                     'passionate',
                     'people',
                     'perspective',
                     'player',
                     'positive', 
                     'powerpoint',
                     'practice',
                     'pragmatic',
                     'precision',
                     'present', 
                     'presentation',
                     'priority',
                     'prioritization',
                     'prioritizing',
                     'prioritize',
                     'proactive',
                     'proactively',
                     'problem', 
                     'problemsolving',
                     'productive',
                     'productivity',
                     'push', 
                     'quality',
                     'question',
                     'quickly',
                     'read',
                     'reliable',
                     'reliability',
                     'report',
                     'resilient',
                     'resilience',
                     'respect',
                     'respectful',
                     'rigorous',
                     'root', 
                     'savvy',
                     'selfdirected',
                     'selfstarter',
                     'selfmotivated',
                     'simultaneously',
                     'skill',
                     'solve',
                     'solver',
                     'solving',
                     'speed',
                     'storytelling',
                     'strong',
                     'structure',
                     'study',
                     'supervise',
                     'supervision',
                     'team',
                     'teammate',
                     'teamwork',
                     'teamoriented',
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
                     'versatile',
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
                       'assumption',
                       'build',
                       'business',
                       'career', 
                       'case',
                       'challenge',
                       'change', 
                       'chosen',
                       'client',
                       'clientfacing',
                       'coach',
                       'coaching',
                       'complex', 
                       'cross', 
                       'crossfunctional',
                       'crossfunctionally',
                       'customer', 
                       'customerfacing',
                       'data',
                       'dataset',
                       'datasets',
                       'decision', 
                       'decisionmaking',
                       'decisionmakers',
                       'deep', 
                       'deliver', 
                       'delivery',
                       'deliverable',
                       'delivering',
                       'delivered',
                       'demonstrated', 
                       'development',
                       'difference',
                       'differentiated',
                       'digital', 
                       'direct',
                       'director',
                       'disparate',
                       'domain',
                       'drive',
                       'driven',
                       'earn', 
                       'efficiency',
                       'end', 
                       'engagement',
                       'entrepreneurial',
                       'entrepreneur', 
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
                       'highimpact',
                       'identify', 
                       'identifies',
                       'identifying',
                       'identification',
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
                       'lean',
                       'line',
                       'make', 
                       'maker',
                       'manage',
                       'management',
                       'managing',
                       'manages',
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
                       'oversee',
                       'oversees',
                       'overseeing',
                       'owner',
                       'ownership',
                       'paper',
                       'partner',
                       'partnering',
                       'peer', 
                       'pm',
                       'problem',
                       'problemsolving',
                       'process',
                       'product',
                       'professional',
                       'professionalism', 
                       'profitable',
                       'profitability',
                       'program',
                       'programmatic',
                       'project', 
                       'proposal',
                       'prototype',
                       'prototyping',
                       'proven',
                       'question',
                       'rapid', 
                       'record',
                       'recommendation',
                       'requirement',
                       'review',
                       'reviewed',
                       'risk',
                       'sale',
                       'science',
                       'service',
                       'sigma',
                       'six', 
                       'skill',
                       'sme',
                       'solution',
                       'solve',
                       'solving',
                       'stakeholder',
                       'story',
                       'strategically',
                       'strategy',
                       'strategist',
                       'stream',
                       'subject',
                       'success',
                       'supervisory',
                       'team',
                       'teammate',
                       'teamwork',
                       'teamoriented',
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


