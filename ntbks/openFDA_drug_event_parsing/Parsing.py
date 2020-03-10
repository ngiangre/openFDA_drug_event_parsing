#!/usr/bin/env python
# coding: utf-8

# ## openFDA Drug Event data parsing, processing, and output

# import libraries

# In[1]:


import os
import io
import urllib
import requests
import zipfile
import json
import time
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize


# read in api token and put in header for api call

# In[2]:


api_token = pd.read_csv('../../.openFDA.params').api_key.values[0]


# In[3]:


headers = {'Content-Type': 'application/json',
           'Authorization': 'Bearer {0}'.format(api_token)}


# get openFDA drug event links

# In[4]:


filehandle, _ = urllib.request.urlretrieve("https://api.fda.gov/download.json")


# In[5]:


with open(filehandle) as json_file:
    data = json.load(json_file)


# how mmany records are there?

# In[6]:


data['results']['drug']['event']['total_records']


# how many files do we have?

# In[7]:


len(data['results']['drug']['event']['partitions'])


# put all files into a list

# In[8]:


drug_event_files = [x['file'] for x in data['results']['drug']['event']['partitions']]


# create output directory

# In[9]:


data_dir = "../../data/"
out=data_dir+'openFDA_drug_event/'
try:
    os.mkdir(out)
except:
    print(out+' exists')
    
out_report=out+'report/'
try:
    os.mkdir(out_report)
except:
    print(out_report+' exists')
    
out_meta=out+'meta/'
try:
    os.mkdir(out_meta)
except:
    print(out_meta+' exists')
    
out_patient=out+'patient/'
try:
    os.mkdir(out_patient)
except:
    print(out_patient+' exists')
    
out_patient_drug=out+'patient_drug/'
try:
    os.mkdir(out_patient_drug)
except:
    print(out_patient_drug+' exists')
    
out_patient_drug_openfda=out+'patient_drug_openfda/'
try:
    os.mkdir(out_patient_drug_openfda)
except:
    print(out_patient_drug_openfda+' exists')

out_patient_drug_openfda_rxcui=out+'patient_drug_openfda_rxcui/'
try:
    os.mkdir(out_patient_drug_openfda_rxcui)
except:
    print(out_patient_drug_openfda_rxcui+' exists')

out_patient_reaction=out+'patient_reaction/'
try:
    os.mkdir(out_patient_reaction)
except:
    print(out_patient_reaction+' exists')


# ## drug event attributes 

# ### get attributes

# In[10]:


filehandle, _ = urllib.request.urlretrieve('https://open.fda.gov/fields/drugevent.yaml')


# In[11]:


import yaml

with open(filehandle, 'r') as stream:
    try:
        attribute_map = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


# In[12]:


attribute_map['properties']


# ## functions

# ### retrive data from files

# In[13]:


def request_and_generate_data(drug_event_file,headers=headers,stream=True):
    response = requests.get(drug_event_file,headers=headers,stream=True)
    zip_file_object = zipfile.ZipFile(io.BytesIO(response.content))
    first_file = zip_file_object.namelist()[0]
    file = zip_file_object.open(first_file)
    content = file.read()
    data = json.loads(content.decode())
    return data


# ### report data formatting/mapping function

# In[14]:


def report_formatter(df):
    
    attributes_dict = attribute_map['properties']

    cols = np.intersect1d(list(attributes_dict.keys()),df.columns)

    for col in cols:
        try:
            if attributes_dict[col]['possible_values']['type']=='one_of':
                attributes_dict_col = attributes_dict[col]['possible_values']['value']
                df[col] = df[col].astype(float)
                df[col] = (df[col].
                 apply(lambda x : str(int(x)) if (x>=0) else x).
                 map(attributes_dict_col)
                )
        except:
            pass
    return df


# ### report primarysource formatting/mapping function

# In[15]:


def primarysource_formatter(df):

    keyword = 'primarysource'

    attributes_dict = attribute_map['properties'][keyword]['properties']

    cols = np.intersect1d(list(attributes_dict.keys()),[x.replace(keyword+'.','') for x in df.columns])

    for col in cols:
        try:
            if attributes_dict[col]['possible_values']['type']=='one_of':
                attributes_dict_col = attributes_dict[col]['possible_values']['value']
                df[keyword+'.'+col] = df[keyword+'.'+col].astype(float)
                df[keyword+'.'+col] = (df[keyword+'.'+col].
                 apply(lambda x : str(int(x)) if (x>=0) else x).
                 map(attributes_dict_col)
                )                
        except:
            pass
    return df


# ### report serious formatting/mapping function

# In[16]:


def report_serious_formatter(df):

    attributes_dict = attribute_map['properties']

    col = 'serous'

    try:
        attributes_dict_col = attributes_dict[col]['possible_values']['value']
        df[col] = df[col].astype(float)
        df[col] = (df[col].
         apply(lambda x : str(int(x)) if (x>=0) else x).
         map(attributes_dict_col)
        )
    except:
        pass
    return df


# ### patient data formatting/mapping function

# In[17]:


def patient_formatter(df):
    attributes_dict = attribute_map['properties']['patient']['properties']

    cols = np.intersect1d(list(attributes_dict.keys()),[x.replace('patient.','') for x in df.columns])

    for col in cols:
        try:
            if attributes_dict[col]['possible_values']['type']=='one_of':
                attributes_dict_col = attributes_dict[col]['possible_values']['value']
                df['patient.'+col] = df['patient.'+col].astype(float)
                df['patient.'+col] = (df['patient.'+col].
                 apply(lambda x : str(int(x)) if (x>=0) else x).
                 map(attributes_dict_col)
                )                
        except:
            pass
        if 'date' in col:
            df[col] = pd.to_datetime(df[col],infer_datetime_format=True)
            
    aged = df.copy()
    aged = aged[['patient.patientonsetage','patient.patientonsetageunit']].dropna()
    year_reports = (aged[aged['patient.patientonsetageunit'].astype(str)=='Year']).index.values
    month_reports = (aged[aged['patient.patientonsetageunit'].astype(str)=='Month']).index.values
    day_reports = (aged[aged['patient.patientonsetageunit'].astype(str)=='Day']).index.values
    decade_reports = (aged[aged['patient.patientonsetageunit'].astype(str)=='Decade']).index.values
    week_reports = (aged[aged['patient.patientonsetageunit'].astype(str)=='Week']).index.values
    hour_reports = (aged[aged['patient.patientonsetageunit'].astype(str)=='Hour']).index.values

    aged['master_age'] = np.nan
    
    aged['master_age'].loc[year_reports] = aged['patient.patientonsetage'].loc[year_reports].astype(int)
    aged['master_age'].loc[month_reports] = aged['patient.patientonsetage'].loc[month_reports].astype(int)/12.
    aged['master_age'].loc[week_reports] = aged['patient.patientonsetage'].loc[week_reports].astype(int)/365.
    aged['master_age'].loc[day_reports] = aged['patient.patientonsetage'].loc[day_reports].astype(int)/365.
    aged['master_age'].loc[decade_reports] = aged['patient.patientonsetage'].loc[decade_reports].astype(int)*10.
    aged['master_age'].loc[hour_reports] = aged['patient.patientonsetage'].loc[hour_reports].astype(int)/365./24.
    
    
    return df.join(aged[['master_age']])


# ### parse patient.drug data formatting/mapping function

# #### patient.drug formatting/mapping function

# In[18]:


def patient_drug_formatter(df):
    attributes_dict = attribute_map['properties']['patient']['properties']['drug']['items']['properties']

    cols = np.intersect1d(list(attributes_dict.keys()),df.columns)

    for col in cols:
        try:
            if attributes_dict[col]['possible_values']['type']=='one_of':
                attributes_dict_col = attributes_dict[col]['possible_values']['value']
                df[col] = df[col].astype(float)
                if col=='drugadministrationroute':
                    df[col] = (df[col].
                     apply(lambda x : ''.join(np.repeat('0',3-len(str(int(x)))))+str(int(x)) if (x>=0) else x).
                     map(attributes_dict_col)
                    )
                else:
                    df[col] = (df[col].
                     apply(lambda x : str(int(x)) if (x>=0) else x).
                     map(attributes_dict_col)
                    )
        except:
            pass
    return df


# #### main parser formatting/mapping function

# In[19]:


def parse_patient_drug_data(results):
    dict_name = 'patient.drug'
    patientdrugs = []
    for reportid in results['safetyreportid'].unique():
        lst = []
        dict_or_list = results[[dict_name]].loc[reportid].values[0]
        if type(dict_or_list)==dict:
            lst.extend(dict_or_list)
        if type(dict_or_list)==list:
            lst = dict_or_list
        if type(dict_or_list)==np.ndarray:
            lst = dict_or_list[0]
        for i,l in enumerate(lst):
            l = l.copy()
            dict_ = {}
            try:
                del l['openfda']
            except:
                pass
            dict_[str(reportid)] = l
            patientdrug = (pd.DataFrame(dict_).
                           T.
                           rename_axis('safetyreportid').
                           reset_index()
                          )
            patientdrug['entry'] = i
            patientdrugs.append(patientdrug)
    allpatientdrugs = pd.concat(patientdrugs,sort=True)
    cols_to_keep = allpatientdrugs.columns[[type(x)==str for x in allpatientdrugs.columns]]
    
    return patient_drug_formatter(allpatientdrugs[cols_to_keep])


# ### patient.drug.openfda formatting/mapping function

# #### main parser formatting/mapping function

# In[20]:


def parse_patient_drug_openfda_data(results):
    dict_name = 'patient.drug'
    openfdas = []
    for reportid in results['safetyreportid'].unique():
        lst = []
        dict_or_list = results[[dict_name]].loc[reportid].values[0]
        if type(dict_or_list)==dict:
            lst.extend(dict_or_list)
        if type(dict_or_list)==list:
            lst = dict_or_list
        if type(dict_or_list)==np.ndarray:
            lst = dict_or_list[0]
        for i,l in enumerate(lst):
            try:
                openfda = (pd.concat(
                    {k: pd.Series(v) for k, v in l['openfda'].items()}
                ).
                 reset_index().
                 drop('level_1',axis=1).
                 rename(columns={'level_0' : 'key',0 : 'value'})
                )
                openfda['safetyreportid']=np.repeat(reportid,openfda.shape[0])
                openfda['entry'] = i
                openfdas.append(openfda)
            except:
                pass
        
    openfdas_df = pd.concat(openfdas,sort=True)
    
    return openfdas_df


# ### parse patient.reaction data formatting/mapping function

# #### patient.reaction formatter function

# In[21]:


def patient_reactions_formatter(df):
    
    attributes_dict = attribute_map['properties']['patient']['properties']['reaction']['items']['properties']

    cols = np.intersect1d(list(attributes_dict.keys()),df.columns)

    for col in cols:
        try:
            if attributes_dict[col]['possible_values']['type']=='one_of':
                attributes_dict_col = attributes_dict[col]['possible_values']['value']
                df[col] = df[col].astype(float)
                df[col] = (df[col].
                 apply(lambda x : str(int(x)) if (x>=0) else x).
                 map(attributes_dict_col)
                )
        except:
            pass
    return df


# #### main parser

# In[22]:


def parse_patient_reaction_data(results):
    dict_name='patient.reaction'
    allpatientreactions = []
    for reportid in results['safetyreportid'].unique():
        lst = []
        dict_or_list = results[[dict_name]].loc[reportid].values[0]
        if type(dict_or_list)==dict:
            lst.extend(dict_or_list)
        if type(dict_or_list)==list:
            lst = dict_or_list
        if type(dict_or_list)==np.ndarray:
            lst = dict_or_list[0]
        rxs = []
        for i,l in enumerate(lst):
            rx = (pd.DataFrame(l,index=[reportid]).
             rename_axis('safetyreportid').
             reset_index()
            )
            rx['entry'] = i
            rxs.append(rx)
        allpatientreactions.append(pd.concat(rxs,sort=True))
    return patient_reactions_formatter(pd.concat(allpatientreactions,sort=True)).reset_index(drop=True)


# ### main parsing function

# In[23]:


def parsing_main(drug_event_file):
    t0 = time.time()
    
    file_lst = drug_event_file.split('/')[2:]
    out_file = '_'.join(file_lst[3:]).split('.')[0]
    
    print('\nparsing '+out_file+"...\n")
    
    try:
        data = request_and_generate_data(drug_event_file,headers=headers,stream=True)
    
        #parse metadata
        meta = json_normalize(data['meta'])
        (meta.
         to_csv(out_meta+out_file+'_meta.csv.gzip',
                    compression='gzip'))
        del meta
        
        results = json_normalize(data['results'])

        #parse and output report data
        results.index = results['safetyreportid'].values
        results = results.rename_axis('safetyreportid')
        report = results.drop(['patient.drug','patient.reaction'],axis=1)

        try:
            report_df = (primarysource_formatter(
                report_formatter(
                    report_serious_formatter(report)
                )
            ).drop(report.columns.
                   values[report.
                          columns.
                          str.contains('patient.')],axis=1).
             reset_index(drop=True))
            (report_df.to_csv(out_report+out_file+'_report.csv.gzip',
                    compression='gzip'))
            del report
            del report_df
        except:
            print('could not parse report data in '+out_file)
            pass

        try:
            patient_df = (results.
                          loc[:,results.
                              columns.values[results.columns.str.contains('patient.')]]
                          ).drop(['patient.drug','patient.reaction'],axis=1)
            (patient_formatter(patient_df).
             reset_index().
             to_csv(out_patient+out_file+'_patient.csv.gzip',
                    compression='gzip'))
            del patient_df
        except:
            print('could not parse patient data in '+out_file)
            pass

        try:
            patientdrug_df = parse_patient_drug_data(results)
            (patientdrug_df.
             reset_index(drop=True).
             to_csv(out_patient_drug+out_file+'_patient_drug.csv.gzip',
                    compression='gzip'))
            del patientdrug_df
        except:
            print('could not parse patient.drug data in '+out_file)
            pass

        try:
            openfdas_df = parse_patient_drug_openfda_data(results).reset_index(drop=True)
            (openfdas_df.to_csv(
                out_patient_drug_openfda+out_file+'_patient_drug_openfda.csv.gzip',
                compression='gzip'))
            (openfdas_df.
             query('key=="rxcui"').
             to_csv(
                out_patient_drug_openfda_rxcui+out_file+'_patient_drug_openfda_rxcui.csv.gzip',
                compression='gzip'))
            del openfdas_df
        except:
            print('could not parse patient.drug.openfda data in '+out_file)
            pass

        try:
            patientreactions = parse_patient_reaction_data(results)
            (patientreactions.
             reset_index(drop=True).
             to_csv(out_patient_reaction+out_file+'_patient_reaction.csv.gzip',
                    compression='gzip'))
            del patientreactions
        except:
            print('could not parse patient.reaction data in '+out_file)
            pass

        t1 = time.time()
        print("\n"+str(np.round(t1-t0,0))+' seconds to parse '+out_file+"\n")
        
    except requests.exceptions.ConnectionError:
        print('cannot connect to json data')
        pass


# 
# ## main

# In[24]:


from joblib import Parallel, delayed
#from dask import delayed, compute, persist
#from dask.distributed import Client, LocalCluster, progress

n_jobs = 50

#if __name__=='__main__':
t0_loop = time.time()

Parallel(n_jobs=n_jobs)(delayed(parsing_main)(drug_event_file) for drug_event_file in drug_event_files)
         
#cluster = LocalCluster(n_workers=n_jobs, threads_per_worker=1)
#c = Client(cluster) 
#results = [delayed(parsing_main)(drug_event_file) for drug_event_file in drug_event_files]
#compute(*results[:1])      # convert to final result when done if desired  

t1_loop = time.time()
print("\n"+str(np.round(t1_loop-t0_loop,0))+' seconds to parse files')


# In[ ]:




