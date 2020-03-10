#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import numpy as np
import pandas as pd
import pickle

data_dir = "../../data/openFDA_drug_event/"
er_dir = data_dir+'er_tables/'


# In[2]:


primarykey='safetyreportid'


# In[3]:


patients = pd.read_csv(er_dir+'patient.csv.gz',
                       compression='gzip',
                       index_col=0,dtype={
                           'safetyreportid' : 'str',
                           'patient_custom_master_age' : 'float'
                       })


# In[7]:


age_col='patient_onsetage'
aged = patients[patients[age_col].notnull()].reset_index(drop=True).copy()


# In[8]:


col = 'nichd'

neonate = aged[age_col].apply(lambda x : float(x)>0 and float(x)<=(1/12))
infant = aged[age_col].apply(lambda x : float(x)>(1/12) and float(x)<=1)
toddler = aged[age_col].apply(lambda x : float(x)>1 and float(x)<=2)
echildhood = aged[age_col].apply(lambda x : float(x)>2 and float(x)<=5)
mchildhood = aged[age_col].apply(lambda x : float(x)>5 and float(x)<=11)
eadolescence = aged[age_col].apply(lambda x : float(x)>11 and float(x)<=18)
ladolescence = aged[age_col].apply(lambda x : float(x)>18 and float(x)<=21)

aged[col] = np.nan

aged.loc[neonate,col] = 'term_neonatal'
aged.loc[infant,col] = 'infancy'
aged.loc[toddler,col] = 'toddler'
aged.loc[echildhood,col] = 'early_childhood'
aged.loc[mchildhood,col] = 'middle_childhood'
aged.loc[eadolescence,col] = 'early_adolescence'
aged.loc[ladolescence,col] = 'late_adolescence'


# In[9]:


col = 'ich_ema'

term_newborn_infants = (aged[age_col].
                        apply(lambda x : float(x)>0 and float(x)<=(1/12)))
infants_and_toddlers = (aged[age_col].
                       apply(lambda x : float(x)>(1/12) and float(x)<=2))
children = aged[age_col].apply(lambda x : float(x)>2 and float(x)<=11)
adolescents = aged[age_col].apply(lambda x : float(x)>11 and float(x)<=17)

aged[col] = np.nan

aged.loc[term_newborn_infants,col] = 'term_newborn_infants'
aged.loc[infants_and_toddlers,col] = 'infants_and_toddlers'
aged.loc[children,col] = 'children'
aged.loc[adolescents,col] = 'adolescents'


# In[10]:


col = 'fda'

neonates = (aged[age_col].
                        apply(lambda x : float(x)>0 and float(x)<(1/12)))
infants = (aged[age_col].
                       apply(lambda x : float(x)>=(1/12) and float(x)<2))
children = aged[age_col].apply(lambda x : float(x)>=2 and float(x)<11)
adolescents = aged[age_col].apply(lambda x : float(x)>=11 and float(x)<16)

aged[col] = np.nan

aged.loc[neonates,col] = 'neonates'
aged.loc[infants,col] = 'infants'
aged.loc[children,col] = 'children'
aged.loc[adolescents,col] = 'adolescents'


# In[11]:


pediatric_patients = (aged.
                      dropna(subset=['nichd']).
                      reset_index(drop=True))
print(pediatric_patients.shape)
print(pediatric_patients.head())


# In[12]:


del patients
del aged


# In[13]:


pediatric_patients.head()


# In[16]:


report = (pd.read_csv(er_dir+'report.csv.gz',
                      compression='gzip',
                     dtype={
                         'safetyreportid' : 'str'
                     }))
report.head()


# In[19]:


df1 = pediatric_patients.copy()
ped_reports = df1.safetyreportid.unique()
df2 = report.copy()
print(df1.shape)
print(df2.shape)
df1[primarykey] = df1[primarykey].astype(str)
df2[primarykey] = df2[primarykey].astype(str)
pediatric_patients_report = pd.merge(df1,
         df2,
         on=primarykey,
         how='inner').query('safetyreportid in @ped_reports')
print(pediatric_patients_report.shape)


# In[20]:


del pediatric_patients
del report


# In[22]:


report_serious = pd.read_csv(er_dir+'report_serious.csv.gz',compression='gzip')
report_serious.head()


# In[23]:


df1 = pediatric_patients_report.copy()
df2 = report_serious.copy()
print(df1.shape)
print(df2.shape)
df1[primarykey] = df1[primarykey].astype(str)
df2[primarykey] = df2[primarykey].astype(str)
pediatric_patients_report_serious = pd.merge(df1,
         df2,
         on=primarykey,
         how='inner')
print(pediatric_patients_report_serious.shape)


# In[24]:


pediatric_patients_report_serious.head()


# In[25]:


del report_serious
del pediatric_patients_report


# In[26]:


reporter = pd.read_csv(er_dir+'reporter.csv.gz',compression='gzip')
reporter.head()


# In[27]:


df1 = pediatric_patients_report_serious.copy()
df2 = reporter.copy()
print(df1.shape)
print(df2.shape)
df1[primarykey] = df1[primarykey].astype(str)
df2[primarykey] = df2[primarykey].astype(str)
pediatric_patients_report_serious_reporter = pd.merge(df1,
         df2,
         on=primarykey,
         how='inner')
print(pediatric_patients_report_serious_reporter.shape)


# In[28]:


pediatric_patients_report_serious_reporter.head()


# In[29]:


pediatric_patients_report_serious_reporter.info()


# In[30]:


del reporter


# In[31]:


del pediatric_patients_report_serious


# In[32]:


(pediatric_patients_report_serious_reporter.
 to_csv('../../data/pediatric_patients_report_serious_reporter.csv.gz',
       compression='gzip')
)


# In[33]:


ped_reports = pediatric_patients_report_serious_reporter.safetyreportid.astype(str).unique()
len(ped_reports)


# In[34]:


pediatric_patients_report_serious_reporter = (pd.
 read_csv('../../data/pediatric_patients_report_serious_reporter.csv.gz',
       compression='gzip',
         index_col=0)
)
pediatric_patients_report_serious_reporter.head()


# In[35]:


pediatric_standard_drugs_atc = (pd.
                            read_csv('../../data/openFDA_drug_event/er_tables/standard_drugs_atc.csv.gz',
                                     compression='gzip',
                                    dtype={
                                        'safetyreportid' : 'str'
                                    }).
                            query('safetyreportid in @ped_reports')
                           )
pediatric_standard_drugs_atc.safetyreportid = pediatric_standard_drugs_atc.safetyreportid.astype(str) 
pediatric_standard_drugs_atc.ATC_concept_id = pediatric_standard_drugs_atc.ATC_concept_id.astype(int)
pediatric_standard_drugs_atc.head()


# In[36]:


pediatric_standard_reactions = (pd.
                  read_csv(er_dir+'standard_reactions.csv.gz',
                           compression='gzip')
                      ).query('safetyreportid in @ped_reports')
pediatric_standard_reactions.safetyreportid = pediatric_standard_reactions.safetyreportid.astype(str) 
pediatric_standard_reactions.MedDRA_concept_id = pediatric_standard_reactions.MedDRA_concept_id.astype(int)
pediatric_standard_reactions.head()


# In[37]:


print(pediatric_patients_report_serious_reporter.head())
print(pediatric_standard_drugs_atc.head())
print(pediatric_standard_reactions.head())


# In[38]:


len(np.intersect1d(
    pediatric_standard_drugs_atc.safetyreportid.astype(str).unique(),
    pediatric_standard_reactions.safetyreportid.astype(str).unique()
))


# In[39]:


pediatric_patients_report_serious_reporter_drugs_reactions = (pediatric_patients_report_serious_reporter.
 set_index('safetyreportid').
 join(pediatric_standard_drugs_atc.
      set_index('safetyreportid')
     ).
 dropna(subset=['ATC_concept_id']).
 join(pediatric_standard_reactions.
     set_index('safetyreportid')
     ).
 dropna(subset=['MedDRA_concept_id']).
 reset_index()
)
pediatric_patients_report_serious_reporter_drugs_reactions = (pediatric_patients_report_serious_reporter_drugs_reactions.
 reindex(np.sort(pediatric_patients_report_serious_reporter_drugs_reactions.columns),axis=1))

pediatric_patients_report_serious_reporter_drugs_reactions.ATC_concept_id = pediatric_patients_report_serious_reporter_drugs_reactions.ATC_concept_id.astype(int).copy()

pediatric_patients_report_serious_reporter_drugs_reactions.MedDRA_concept_code = pediatric_patients_report_serious_reporter_drugs_reactions.MedDRA_concept_code.astype(int).copy()

pediatric_patients_report_serious_reporter_drugs_reactions.MedDRA_concept_id = pediatric_patients_report_serious_reporter_drugs_reactions.MedDRA_concept_id.astype(int).copy()

print(pediatric_patients_report_serious_reporter_drugs_reactions.shape)
print(pediatric_patients_report_serious_reporter_drugs_reactions.head())
print(pediatric_patients_report_serious_reporter_drugs_reactions.safetyreportid.nunique())


# In[40]:


(pediatric_patients_report_serious_reporter_drugs_reactions.
 to_csv('../../data/pediatric_patients_report_serious_reporter_drugs_reactions.csv.gz',
       compression='gzip')
)


# In[41]:


del pediatric_patients_report_serious_reporter


# In[42]:


pediatric_standard_drugs = (pd.
                            read_csv('../../data/openFDA_drug_event/er_tables/standard_drugs.csv.gz',
                                     compression='gzip',
                                    dtype={
                                        'safetyreportid' : 'str'
                                    }).
                            query('safetyreportid in @ped_reports')
                           )
pediatric_standard_drugs.safetyreportid = pediatric_standard_drugs.safetyreportid.astype(str) 
pediatric_standard_drugs.RxNorm_concept_id = pediatric_standard_drugs.RxNorm_concept_id.astype(int)
pediatric_standard_drugs.head()


# In[51]:


import os
rxfiles = os.listdir('../../RxNorm_relationships_tables/')
rxfile_dict={}
for rxfile in rxfiles:
    key=rxfile.split('.')[0]
    rxfile_dict[key] = pd.read_csv('../../RxNorm_relationships_tables/'+rxfile,engine='c',index_col=0)


# In[58]:


tobrand=[]
for rxfile in rxfile_dict.keys():
    tobrand.append(rxfile_dict[rxfile].query('concept_class_id_2=="Brand Name"'))


# In[69]:


a = pediatric_standard_drugs.copy()
print(a[primarykey].nunique())
m = (pd.merge(
    a,
    pd.concat(tobrand),
    left_on='RxNorm_concept_id',
    right_on='concept_id_1'
)
)
m[primarykey].nunique()


# In[74]:


m_renamed = (m.
 loc[:,
     [primarykey,'concept_class_id_2','concept_code_2','concept_name_2','concept_id_2']
    ].
 rename(columns={
     'concept_class_id_2' : 'RxNorm_concept_class_id',
     'concept_code_2' : 'RxNorm_concept_code',
     'concept_name_2' : 'RxNorm_concept_name',
     'concept_id_2' : 'RxNorm_concept_id'})
)


# In[75]:


(m_renamed.
 to_csv('../../data/pediatric_patients_report_drug_brands.csv.gz',
       compression='gzip')
)


# In[ ]:




