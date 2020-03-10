#!/usr/bin/env python
# coding: utf-8

# In[205]:


import glob
import numpy as np
import pandas as pd
from dask import delayed, compute
import dask.dataframe as dd
import pickle

data_dir = "../../data/openFDA_drug_event/"
er_dir = data_dir+'er_tables/'

try:
    os.mkdir(er_dir)
except:
    print(er_dir+" exists")


# ## dask delayed functions

# In[206]:


primarykey = 'safetyreportid'

def read_file(file):
    return pd.read_csv(file,compression='gzip',index_col=0,dtype={primarykey : 'str'})


# ## ER tables

# ### report

# #### report_df

# In[207]:


dir_ = data_dir+'report/'
files = glob.glob(dir_+'*.csv.gzip')
results = []
for file in files:
    df = delayed(read_file)(file)
    results.append(df)
report_df = (pd.concat(compute(*results),sort=True))
report_df[primarykey] = (report_df[primarykey].astype(str))
print(report_df.columns.values)
report_df.head()


# #### report_er_df

# In[15]:


columns = [primarykey,'receiptdate',
           'receivedate',
           'transmissiondate']
rename_columns = {'receiptdate' : 'mostrecent_receive_date',
                  'receivedate' : 'receive_date',
                  'transmissiondate' : 'lastupdate_date'}

report_er_df = (report_df[columns].
                rename(columns=rename_columns).
                set_index(primarykey).
                sort_index().
                reset_index().
                dropna(subset=[primarykey]).
                drop_duplicates()
               )
report_er_df = report_er_df.reindex(np.sort(report_er_df.columns),axis=1)
report_er_df[primarykey] = report_er_df[primarykey].astype(str)       
report_er_df = report_er_df.reindex(np.sort(report_er_df.columns),axis=1)
print(report_er_df.info())
report_er_df.head()


# In[18]:


(report_er_df.
 groupby(primarykey).
 agg(max).
 reset_index().
 dropna(subset=[primarykey])
).to_csv(er_dir+'report.csv.gz',compression='gzip',index=False)


# In[ ]:


del report_er_df


# ### report_serious

# In[19]:


columns = [primarykey,'serious',
           'seriousnesscongenitalanomali',
           'seriousnesslifethreatening',
          'seriousnessdisabling',
          'seriousnessdeath',
          'seriousnessother']
rename_columns = {           
    'seriousnesscongenitalanomali' : 'congenital_anomali',
    'seriousnesslifethreatening' : 'life_threatening',
    'seriousnessdisabling' : 'disabling',
    'seriousnessdeath' : 'death',
    'seriousnessother' : 'other'}

report_serious_er_df = (report_df[columns].
                        rename(columns=rename_columns).
                        set_index(primarykey).
                        sort_index().
                        reset_index().
                        dropna(subset=[primarykey]).
                        drop_duplicates().
                        groupby(primarykey).
                        first().
                        reset_index().
                        dropna(subset=[primarykey])
                       )
report_serious_er_df[primarykey] = report_serious_er_df[primarykey].astype(str)       
report_serious_er_df = report_serious_er_df.reindex(np.sort(report_serious_er_df.columns),axis=1)
print(report_serious_er_df.info())
report_serious_er_df.head()


# In[29]:


(report_serious_er_df).to_csv(er_dir+'report_serious.csv.gz',compression='gzip',index=False)


# ### reporter

# In[1]:


columns = [primarykey,'companynumb',
           'primarysource.qualification',
           'primarysource.reportercountry']
rename_columns = {'companynumb' : 'reporter_company',
                  'primarysource.qualification' : 'reporter_qualification',
                  'primarysource.reportercountry' : 'reporter_country'}

reporter_er_df = (report_df[columns].
                  rename(columns=rename_columns).
                  set_index(primarykey).
                  sort_index().
                  reset_index().
                  dropna(subset=[primarykey]).
                  drop_duplicates().
                  groupby(primarykey).
                  first().
                  reset_index().
                  dropna(subset=[primarykey])
                 )
reporter_er_df[primarykey] = reporter_er_df[primarykey].astype(str)  
reporter_er_df = reporter_er_df.reindex(np.sort(reporter_er_df.columns),axis=1)
print(reporter_er_df.info())
reporter_er_df.head()


# In[36]:


(reporter_er_df).to_csv(er_dir+'reporter.csv.gz',compression='gzip',index=False)


# In[41]:


try:
    del df
except:
    pass
try:
    del report_df
except:
    pass
try:
    del report_serious_er_df
except:
    pass
try:
    del report_er_df
except:
    pass
try:
    del reporter_er_df
except:
    pass


# ### patient

# #### patient_df

# In[42]:


dir_ = data_dir+'patient/'
files = glob.glob(dir_+'*.csv.gzip')
results = []
for file in files:
    df = delayed(read_file)(file)
    results.append(df)
patient_df = (pd.concat(compute(*results),sort=True))
patient_df[primarykey] = (patient_df[primarykey].astype(str))
print(patient_df.columns.values)
patient_df.head()


# #### patient_er_df

# In[43]:


columns = [primarykey,
              'patient.patientonsetage',
              'patient.patientonsetageunit',
              'master_age',
              'patient.patientsex',
              'patient.patientweight'
             ]
rename_columns = {
              'patient.patientonsetage' : 'patient_onsetage',
              'patient.patientonsetageunit' : 'patient_onsetageunit',
              'master_age': 'patient_custom_master_age',
              'patient.patientsex' : 'patient_sex',
              'patient.patientweight' : 'patient_weight'
}

patient_er_df = (patient_df[columns].
                 rename(columns=rename_columns).
                 set_index(primarykey).
                 sort_index().
                 reset_index().
                 dropna(subset=[primarykey]).
                 drop_duplicates().
                 groupby(primarykey).
                 first().
                 reset_index().
                 dropna(subset=[primarykey])
                )
patient_er_df = patient_er_df.reindex(np.sort(patient_er_df.columns),axis=1)
print(patient_er_df.info())
patient_er_df.head()


# In[47]:


(patient_er_df).to_csv(er_dir+'patient.csv.gz',compression='gzip',index=False)


# In[48]:


del df 
del patient_df


# ### drug_characteristics

# #### patient.drug

# In[49]:


dir_ = data_dir+'patient_drug/'
files = glob.glob(dir_+'*.csv.gzip')
results = []
for file in files:
    df = delayed(read_file)(file)
    results.append(df)
patient_drug_df = (pd.concat(compute(*results),sort=True))
patient_drug_df[primarykey] = (patient_drug_df[primarykey].astype(str))
print(patient_drug_df.columns.values)
patient_drug_df.head()


# #### drugcharacteristics_er_df

# In[60]:


columns = [primarykey,
           'medicinalproduct',
           'drugcharacterization',
           'drugadministrationroute',
           'drugindication'
          ]
rename_columns = {
              'medicinalproduct' : 'medicinal_product',
              'drugcharacterization' : 'drug_characterization',
              'drugadministrationroute': 'drug_administration',
    'drugindication' : 'drug_indication'
}

drugcharacteristics_er_df = (patient_drug_df[columns].
                             rename(columns=rename_columns).
                             set_index(primarykey).
                             sort_index().
                             reset_index().
                             drop_duplicates().
                             dropna(subset=[primarykey])
                            )
drugcharacteristics_er_df = (drugcharacteristics_er_df.
                             reindex(np.sort(drugcharacteristics_er_df.columns),axis=1))
print(drugcharacteristics_er_df.info())
drugcharacteristics_er_df.head()


# In[63]:


(drugcharacteristics_er_df
).to_csv(er_dir+'drugcharacteristics.csv.gz',compression='gzip',index=False)


# In[64]:


del drugcharacteristics_er_df
del patient_drug_df
del df


# ### drugs

# #### patient.drug.openfda.rxcui_df

# In[65]:


dir_ = data_dir+'patient_drug_openfda_rxcui/'
files = glob.glob(dir_+'*.csv.gzip')
results = []
for file in files:
    df = delayed(read_file)(file)
    results.append(df)
patient_drug_openfda_rxcui_df = (pd.concat(compute(*results),sort=True))
print(patient_drug_openfda_rxcui_df.columns.values)
patient_drug_openfda_rxcui_df[primarykey] = (patient_drug_openfda_rxcui_df[primarykey].
                                       astype(str))
patient_drug_openfda_rxcui_df.value = (patient_drug_openfda_rxcui_df.
                                 value.astype(int))
patient_drug_openfda_rxcui_df.head()


# #### drugs_er_df

# In[66]:


columns = [primarykey,
              'value'
             ]
rename_columns = {
              'value' : 'rxcui'
}

drugs_er_df = (patient_drug_openfda_rxcui_df[columns].
               rename(columns=rename_columns).
               set_index(primarykey).
               sort_index().
               reset_index().
               drop_duplicates().
               dropna(subset=[primarykey])
              )
drugs_er_df = drugs_er_df.reindex(np.sort(drugs_er_df.columns),axis=1)
print(drugs_er_df.info())
drugs_er_df.head()


# In[70]:


drugs_er_df['rxcui'] = drugs_er_df['rxcui'].astype(int)


# In[71]:


drugs_er_df[primarykey] = drugs_er_df[primarykey].astype(str)


# In[72]:


(drugs_er_df).to_csv(er_dir+'drugs.csv.gz',compression='gzip',index=False)


# In[73]:


del patient_drug_openfda_rxcui_df
del drugs_er_df
del df


# ### reactions

# #### patient.reaction_df

# In[74]:


dir_ = data_dir+'patient_reaction/'
files = glob.glob(dir_+'*.csv.gzip')
results = []
for file in files:
    df = delayed(read_file)(file)
    results.append(df)
patient_reaction_df = (pd.concat(compute(*results),sort=True))
patient_reaction_df[primarykey] = (patient_reaction_df[primarykey].astype(str))
print(patient_reaction_df.columns.values)
patient_reaction_df.head()


# #### patient_reaction_er_df

# In[75]:


columns = [primarykey,
              'reactionmeddrapt',
           'reactionoutcome'
             ]
rename_columns = {
              'reactionmeddrapt' : 'reaction_meddrapt',
    'reactionoutcome' : 'reaction_outcome'
}

reactions_er_df = (patient_reaction_df[columns].
                   rename(columns=rename_columns).
                   set_index(primarykey).
                   sort_index().
                   reset_index().
                   dropna(subset=[primarykey]).
                   drop_duplicates()
                  )
reactions_er_df[primarykey] = reactions_er_df[primarykey].astype(str)
reactions_er_df = reactions_er_df.reindex(np.sort(reactions_er_df.columns),axis=1)
print(reactions_er_df.info())
reactions_er_df.head()


# In[76]:


(reactions_er_df).to_csv(er_dir+'reactions.csv.gz',compression='gzip',index=False)


# In[77]:


del patient_reaction_df
del reactions_er_df
del df


# ### omop tables for joining

# In[110]:


concept = (pd.read_csv('../../vocabulary_SNOMED_MEDDRA_RxNorm_ATC/CONCEPT.csv',sep='\t',
                      dtype={
                          'concept_id' : 'int'
                      }))
concept.head()


# In[111]:


concept_relationship = (pd.
                        read_csv('../../vocabulary_SNOMED_MEDDRA_RxNorm_ATC/'+
                                 'CONCEPT_RELATIONSHIP.csv',sep='\t',
                                dtype={
                                    'concept_id_1' : 'int',
                                    'concept_id_2' : 'int'
                                }))
concept_relationship.head()


# ### standard_drugs

# In[5]:


drugs = (pd.read_csv(
    er_dir+'drugs.csv.gz',
    compression='gzip',
    dtype={
        'safetyreportid' : 'str'
    }
)
        )


# In[6]:


drugs['rxcui'] = drugs['rxcui'].astype(int)


# In[7]:


urxcuis = drugs['rxcui'].unique()


# In[8]:


print(len(urxcuis))
urxcuis[:5]


# In[9]:


rxnorm_concept = concept.query('vocabulary_id=="RxNorm"')


# In[10]:


concept_codes = rxnorm_concept['concept_code'].astype(int).unique()
print(len(concept_codes))
print(len(urxcuis))

intersect = np.intersect1d(concept_codes,urxcuis)

print(len(intersect))
print(len(intersect)/len(urxcuis))


# In[11]:


del urxcuis
del concept_codes


# In[12]:


rxnorm_concept = concept.query('vocabulary_id=="RxNorm"')

rxnorm_concept_ids = (rxnorm_concept.
                      query('concept_code in @intersect')['concept_id'].
                      astype(int).
                      unique()
                     )
all_rxnorm_concept_ids = (rxnorm_concept['concept_id'].
                          unique()
                         )

r = (concept_relationship.
     copy().
     loc[:,['concept_id_1','concept_id_2','relationship_id']].
     drop_duplicates()
    )
c = rxnorm_concept.copy()
c['concept_id'] = c['concept_id'].astype(int)
c['concept_code'] = c['concept_code'].astype(int)

joined = (drugs.
          set_index('rxcui').
          join(
              c. 
              query('vocabulary_id=="RxNorm"').
              loc[:,['concept_id','concept_code','concept_name','concept_class_id']].
              drop_duplicates().
              set_index('concept_code')
          ).
          dropna().
          rename_axis('RxNorm_concept_code').
          reset_index().
          rename(
              columns={
                  'concept_class_id' : 'RxNorm_concept_class_id',
                  'concept_name' : 'RxNorm_concept_name',
                  'concept_id' : 'RxNorm_concept_id'
              }
          ).
          dropna(subset=['RxNorm_concept_id']).
          drop_duplicates()
         )
joined = (joined.
          reindex(np.sort(joined.columns),axis=1)
         )
print(joined.shape)
print(joined.head())


# In[13]:


len(np.intersect1d(joined.RxNorm_concept_code.unique(),intersect))/len(intersect)


# In[14]:


ids = joined.RxNorm_concept_id.dropna().astype(int).unique()


# In[117]:


pickle.dump(
    ids,
    open('../../data/all_openFDA_rxnorm_concept_ids.pkl','wb')
)


# In[ ]:


(joined.to_csv(er_dir+'standard_drugs.csv.gz',compression='gzip',index=False))


# In[15]:


del joined


# ### standard_reactions

# In[25]:


patient_reaction_df = (pd.read_csv(
    er_dir+'reactions.csv.gz',
    compression='gzip',
                               dtype={
                                   'safetyreportid' : 'str'
                               }
                              ))
all_reports = patient_reaction_df.safetyreportid.unique()
print(patient_reaction_df.columns)
print(patient_reaction_df.safetyreportid.nunique())
print(patient_reaction_df.reaction_meddrapt.nunique())


# In[18]:


patient_reaction_df.head()


# In[19]:


meddra_concept = concept.query('vocabulary_id=="MedDRA"')
meddra_concept.head()


# In[20]:


reactions = patient_reaction_df.reaction_meddrapt.copy().astype(str).str.title().unique()
print(len(reactions))
concept_names = meddra_concept.concept_name.astype(str).str.title().unique()
print(len(concept_names))

intersect_title = np.intersect1d(reactions,concept_names)
print(len(intersect_title))

print(len(intersect_title)/len(reactions))


# In[21]:


patient_reaction_df['reaction_meddrapt'] = (patient_reaction_df['reaction_meddrapt'].
                                            astype(str).
                                            str.
                                            title())
meddra_concept['concept_name'] = (meddra_concept['concept_name'].
                                  astype(str).
                                  str.
                                  title())
print(patient_reaction_df.shape[0])

joined = ((patient_reaction_df.
  set_index('reaction_meddrapt').
  join(
      meddra_concept.
      query('concept_class_id=="PT"').
      loc[:,['concept_id','concept_name','concept_code','concept_class_id']].
      drop_duplicates().
      set_index('concept_name')
  ).
           rename(columns={'concept_id' : 'MedDRA_concept_id',
                          'concept_code' : 'MedDRA_concept_code',
                          'concept_class_id' : 'MedDRA_concept_class_id'}).
           drop_duplicates()
 )
).rename_axis('MedDRA_concept_name').reset_index()
joined = joined.reindex(np.sort(joined.columns),axis=1)
print(joined.shape[0])
print(joined.head())


# In[22]:


del meddra_concept
del patient_reaction_df


# In[23]:


joined_notnull = joined[joined.MedDRA_concept_id.notnull()]
print(joined_notnull.shape[0])
joined_notnull['MedDRA_concept_id'] = joined_notnull['MedDRA_concept_id'].astype(int)
print(joined_notnull.head())


# In[27]:


print(
    len(
        np.intersect1d(
            all_reports,
            joined_notnull.safetyreportid.astype(str).unique()
        )
    )/len(all_reports)
)


# In[24]:


print(joined_notnull.MedDRA_concept_class_id.value_counts())
print(joined_notnull.safetyreportid.nunique())
print(joined_notnull.MedDRA_concept_id.nunique())


# In[15]:


pickle.dump(
    joined_notnull.MedDRA_concept_id.astype(int).unique,
    open('../../data/all_openFDA_meddra_concept_ids.pkl','wb')
)


# In[18]:


(joined_notnull.to_csv(er_dir+'standard_reactions.csv.gz',compression='gzip',index=False))


# In[16]:


del joined_notnull


# In[19]:


del joined


# ### standard_drugs_atc

# In[74]:


standard_drugs = (pd.read_csv(
    er_dir+'standard_drugs.csv.gz',
    compression='gzip',
    dtype={
        'safetyreportid' : 'str'
    }
))


# In[75]:


all_reports = standard_drugs.safetyreportid.unique()
len(all_reports)


# In[76]:


standard_drugs.RxNorm_concept_id = standard_drugs.RxNorm_concept_id.astype(int)


# In[77]:


standard_drugs.head()


# In[78]:


rxnorm_concept = concept.query('vocabulary_id=="RxNorm"')
rxnorm_concept_ids = rxnorm_concept['concept_id'].unique()


# In[79]:


openfda_concept_ids = standard_drugs.RxNorm_concept_id.dropna().astype(int).unique()


# In[80]:


atc_concept = concept.query('vocabulary_id=="ATC" & concept_class_id=="ATC 5th"')

r = (concept_relationship.
     copy().
     loc[:,['concept_id_1','concept_id_2','relationship_id']].
     drop_duplicates()
    )
                            
r['concept_id_1'] = r['concept_id_1'].astype(int)
r['concept_id_2'] = r['concept_id_2'].astype(int)
ac = atc_concept.copy()
ac['concept_id'] = ac['concept_id'].astype(int)
atc_concept_ids = ac['concept_id'].unique()
rc = rxnorm_concept.copy()
rc['concept_id'] = rc['concept_id'].astype(int)
rxnorm_concept_ids = rc['concept_id'].unique()


# In[81]:


rxnorm_to_atc_relationships = (r.
                         query('concept_id_1 in @openfda_concept_ids & '\
                               'concept_id_2 in @atc_concept_ids').
                         set_index('concept_id_1').
                         join(
                             rc. # standard concepts for 1
                             loc[:,['concept_id','concept_code',
                                    'concept_name','concept_class_id']].
                             drop_duplicates().
                             set_index('concept_id')
                         ).
                         rename_axis('RxNorm_concept_id').
                         reset_index().
                         dropna().
                         rename(
                             columns={
                                 'concept_code' : 'RxNorm_concept_code',
                                 'concept_class_id' : 'RxNorm_concept_class_id',
                                 'concept_name' : 'RxNorm_concept_name',
                                 'concept_id_2' : 'ATC_concept_id',
                             }
                         ).
                         set_index('ATC_concept_id').
                         join(
                             ac. # standard concepts for 2
                             loc[:,['concept_id','concept_code',
                                    'concept_name','concept_class_id']].
                             drop_duplicates().
                             set_index('concept_id')
                         ).
                         dropna().
                         rename_axis('ATC_concept_id').
                         reset_index().
                         rename(
                             columns={
                                 'concept_code' : 'ATC_concept_code',
                                 'concept_class_id' : 'ATC_concept_class_id',
                                 'concept_name' : 'ATC_concept_name'
                             }
                         )
                        )
rxnorm_to_atc_relationships.RxNorm_concept_id = (rxnorm_to_atc_relationships.RxNorm_concept_id.
astype(int))
rxnorm_to_atc_relationships.ATC_concept_id = (rxnorm_to_atc_relationships.ATC_concept_id.
astype(int))

rxnorm_to_atc_relationships = (rxnorm_to_atc_relationships.
                            reindex(np.sort(rxnorm_to_atc_relationships.columns),axis=1)
                           )
print(rxnorm_to_atc_relationships.shape)
print(rxnorm_to_atc_relationships.head())


# In[82]:


rxnorm_to_atc_relationships.ATC_concept_class_id.value_counts()


# In[83]:


del r
del ac
del rc


# In[85]:


standard_drugs_atc = (standard_drugs.
                      loc[:,['RxNorm_concept_id','safetyreportid']].
                      drop_duplicates().
                      set_index('RxNorm_concept_id').
                      join(rxnorm_to_atc_relationships.
                           set_index('RxNorm_concept_id')
                          ).
                      drop_duplicates().
                      reset_index(drop=True).
                      drop(['RxNorm_concept_code','RxNorm_concept_name',
                            'RxNorm_concept_class_id','relationship_id'],axis=1).
                      dropna(subset=['ATC_concept_id']).
                      drop_duplicates()
                     )

standard_drugs_atc = standard_drugs_atc.reindex(np.sort(standard_drugs_atc.columns),axis=1)
standard_drugs_atc.ATC_concept_id = standard_drugs_atc.ATC_concept_id.astype(int)
print(len(
    np.intersect1d(all_reports,
                   standard_drugs_atc.safetyreportid.unique()
                  )
)/len(all_reports))
print(standard_drugs_atc.shape)
print(standard_drugs_atc.info())
print(standard_drugs_atc.head())


# In[37]:


del standard_drugs
del rxnorm_to_atc_relationships


# In[38]:


standard_drugs_atc.to_csv(er_dir+'standard_drugs_atc.csv.gz',compression='gzip',index=False)


# In[39]:


del standard_drugs_atc


# ### standard_drugs_rxnorm_ingredients

# In[152]:


all_openFDA_rxnorm_concept_ids = pickle.load(
    open('../../data/all_openFDA_rxnorm_concept_ids.pkl','rb')
)


# In[153]:


all_openFDA_rxnorm_concept_ids


# In[154]:


all_rxnorm_concept_ids = (concept.
                          query('vocabulary_id=="RxNorm"').
                          concept_id.
                          astype(int).
                          unique()
                         )


# In[155]:


r = (concept_relationship.
     loc[:,['concept_id_1','concept_id_2','relationship_id']].
    drop_duplicates().
    dropna().
     copy()
    )
r.concept_id_1 = r.concept_id_1.astype(int)
r.concept_id_2 = r.concept_id_2.astype(int)


# In[156]:


c = (concept.
    query('vocabulary_id=="RxNorm" & standard_concept=="S"').
    loc[:,['concept_id','concept_code',
          'concept_class_id','concept_name']].
    drop_duplicates().
    dropna().
     copy()
    )
c.concept_id = c.concept_id.astype(int).copy()


# In[157]:


all_rxnorm_concept_ids = concept.query('vocabulary_id=="RxNorm"').concept_id.astype(int).unique()
rxnorm_relationships = (r.
 query('concept_id_1 in @all_rxnorm_concept_ids & '+
       'concept_id_2 in @all_rxnorm_concept_ids').
 relationship_id.
 value_counts()
)
rxnorm_relationships


# In[158]:


first_second_relations = (r.
                          query('concept_id_1 in @all_openFDA_rxnorm_concept_ids').
                          set_index('concept_id_1').
                          join(c.
                               set_index('concept_id')).
                          rename(
                              columns={
                                  'concept_id_1' : 'RxNorm_concept_id_1',
                                  'concept_code' : 'RxNorm_concept_code_1',
                                  'concept_class_id' : 'RxNorm_concept_class_id_1',
                                  'concept_name' : 'RxNorm_concept_name_1'
                              }
                          ).
                          rename_axis('RxNorm_concept_id_1').
                          reset_index().
                          set_index('concept_id_2').
                          join(c.
                               set_index('concept_id')
                              ).
                          rename(
                              columns={'concept_id_2' : 'RxNorm_concept_id_2',
                                       'concept_code' : 'RxNorm_concept_code_2',
                                       'concept_class_id' : 'RxNorm_concept_class_id_2',
                                       'concept_name' : 'RxNorm_concept_name_2',
                                       'relationship_id' :'relationship_id_12'
                                      }
                          ).
                          rename_axis('RxNorm_concept_id_2').
                          reset_index().
                          dropna().
                          drop_duplicates()
                         )
first_second_relations = first_second_relations[
    first_second_relations.RxNorm_concept_id_1!=first_second_relations.RxNorm_concept_id_2
]
print(first_second_relations.shape)
first_second_relations = (first_second_relations.
                          reindex(np.sort(first_second_relations.columns),
                                  axis=1)
                         )
print(first_second_relations.head())


# In[159]:


(first_second_relations.loc[:,['RxNorm_concept_class_id_1','RxNorm_concept_class_id_2']].
groupby(['RxNorm_concept_class_id_1','RxNorm_concept_class_id_2']).
 count()
)


# In[160]:


ids = first_second_relations.RxNorm_concept_id_2.astype(int).unique()

second_third_relations = (r.
                          query('concept_id_1 in @ids').
                          set_index('concept_id_1').
                          join(c.
                               set_index('concept_id')).
                          rename(
                              columns={
                                  'concept_id_1' : 'RxNorm_concept_id_2',
                                  'concept_code' : 'RxNorm_concept_code_2',
                                  'concept_class_id' : 'RxNorm_concept_class_id_2',
                                  'concept_name' : 'RxNorm_concept_name_2'
                              }
                          ).
                          rename_axis('RxNorm_concept_id_2').
                          reset_index().
                          set_index('concept_id_2').
                          join(c.
                               set_index('concept_id')
                              ).
                          rename(
                              columns={'concept_id_2' : 'RxNorm_concept_id_3',
                                       'concept_code' : 'RxNorm_concept_code_3',
                                       'concept_class_id' : 'RxNorm_concept_class_id_3',
                                       'concept_name' : 'RxNorm_concept_name_3',
                                       'relationship_id' :'relationship_id_23'
                                      }
                          ).
                          rename_axis('RxNorm_concept_id_3').
                          reset_index().
                          dropna().
                          drop_duplicates()
                         )
second_third_relations = second_third_relations[
    second_third_relations.RxNorm_concept_id_2!=second_third_relations.RxNorm_concept_id_3
]
print(second_third_relations.shape)
second_third_relations = (second_third_relations.
                          reindex(np.sort(second_third_relations.columns),
                                  axis=1)
                         )
print(second_third_relations.head())


# In[161]:


(second_third_relations.loc[:,['RxNorm_concept_class_id_2','RxNorm_concept_class_id_3']].
groupby(['RxNorm_concept_class_id_2','RxNorm_concept_class_id_3']).
 count()
)


# In[162]:


ids = second_third_relations.RxNorm_concept_id_3.astype(int).unique()

third_fourth_relations = (r.
                          query('concept_id_1 in @ids').
                          set_index('concept_id_1').
                          join(c.
                               set_index('concept_id')).
                          rename(
                              columns={
                                  'concept_id_1' : 'RxNorm_concept_id_3',
                                  'concept_code' : 'RxNorm_concept_code_3',
                                  'concept_class_id' : 'RxNorm_concept_class_id_3',
                                  'concept_name' : 'RxNorm_concept_name_3'
                              }
                          ).
                          rename_axis('RxNorm_concept_id_3').
                          reset_index().
                          set_index('concept_id_2').
                          join(c.
                               set_index('concept_id')
                              ).
                          rename(
                              columns={'concept_id_2' : 'RxNorm_concept_id_4',
                                       'concept_code' : 'RxNorm_concept_code_4',
                                       'concept_class_id' : 'RxNorm_concept_class_id_4',
                                       'concept_name' : 'RxNorm_concept_name_4',
                                       'relationship_id' :'relationship_id_34'
                                      }
                          ).
                          rename_axis('RxNorm_concept_id_4').
                          reset_index().
                          dropna().
                          drop_duplicates()
                         )
third_fourth_relations = third_fourth_relations[
    third_fourth_relations.RxNorm_concept_id_3!=third_fourth_relations.RxNorm_concept_id_4
]
print(third_fourth_relations.shape)
third_fourth_relations = (third_fourth_relations.
                          reindex(np.sort(third_fourth_relations.columns),
                                  axis=1)
                         )
print(third_fourth_relations.head())


# In[163]:


(third_fourth_relations.loc[:,['RxNorm_concept_class_id_3','RxNorm_concept_class_id_4']].
groupby(['RxNorm_concept_class_id_3','RxNorm_concept_class_id_4']).
 count()
)


# In[164]:


ids = third_fourth_relations.RxNorm_concept_id_4.astype(int).unique()

fourth_fifth_relations = (r.
                          query('concept_id_1 in @ids').
                          set_index('concept_id_1').
                          join(c.
                               set_index('concept_id')).
                          rename(
                              columns={
                                  'concept_id_1' : 'RxNorm_concept_id_4',
                                  'concept_code' : 'RxNorm_concept_code_4',
                                  'concept_class_id' : 'RxNorm_concept_class_id_4',
                                  'concept_name' : 'RxNorm_concept_name_4'
                              }
                          ).
                          rename_axis('RxNorm_concept_id_4').
                          reset_index().
                          set_index('concept_id_2').
                          join(c.
                               set_index('concept_id')
                              ).
                          rename(
                              columns={'concept_id_2' : 'RxNorm_concept_id_5',
                                       'concept_code' : 'RxNorm_concept_code_5',
                                       'concept_class_id' : 'RxNorm_concept_class_id_5',
                                       'concept_name' : 'RxNorm_concept_name_5',
                                       'relationship_id' :'relationship_id_45'
                                      }
                          ).
                          rename_axis('RxNorm_concept_id_5').
                          reset_index().
                          dropna().
                          drop_duplicates()
                         )
fourth_fifth_relations = fourth_fifth_relations[
    fourth_fifth_relations.RxNorm_concept_id_4!=fourth_fifth_relations.RxNorm_concept_id_5
]
print(fourth_fifth_relations.shape)
fourth_fifth_relations = (fourth_fifth_relations.
                          reindex(np.sort(fourth_fifth_relations.columns),
                                  axis=1)
                         )
print(fourth_fifth_relations.head())


# In[165]:


(fourth_fifth_relations.loc[:,['RxNorm_concept_class_id_4','RxNorm_concept_class_id_5']].
groupby(['RxNorm_concept_class_id_4','RxNorm_concept_class_id_5']).
 count()
)


# In[166]:


ids = fourth_fifth_relations.RxNorm_concept_id_4.astype(int).unique()

fifth_sixth_relations = (r.
                          query('concept_id_1 in @ids').
                          set_index('concept_id_1').
                          join(c.
                               set_index('concept_id')).
                          rename(
                              columns={
                                  'concept_id_1' : 'RxNorm_concept_id_5',
                                  'concept_code' : 'RxNorm_concept_code_5',
                                  'concept_class_id' : 'RxNorm_concept_class_id_5',
                                  'concept_name' : 'RxNorm_concept_name_5'
                              }
                          ).
                          rename_axis('RxNorm_concept_id_5').
                          reset_index().
                          set_index('concept_id_2').
                          join(c.
                               set_index('concept_id')
                              ).
                          rename(
                              columns={'concept_id_2' : 'RxNorm_concept_id_6',
                                       'concept_code' : 'RxNorm_concept_code_6',
                                       'concept_class_id' : 'RxNorm_concept_class_id_6',
                                       'concept_name' : 'RxNorm_concept_name_6',
                                       'relationship_id' :'relationship_id_56'
                                      }
                          ).
                          rename_axis('RxNorm_concept_id_6').
                          reset_index().
                          dropna().
                          drop_duplicates()
                         )
fifth_sixth_relations = fifth_sixth_relations[
    fifth_sixth_relations.RxNorm_concept_id_5!= fifth_sixth_relations.RxNorm_concept_id_6
]
print(fifth_sixth_relations.shape)
fifth_sixth_relations = (fifth_sixth_relations.
                          reindex(np.sort(fifth_sixth_relations.columns),
                                  axis=1)
                         )
print(fifth_sixth_relations.head())


# In[167]:


(fifth_sixth_relations.loc[:,['RxNorm_concept_class_id_5','RxNorm_concept_class_id_6']].
groupby(['RxNorm_concept_class_id_5','RxNorm_concept_class_id_6']).
 count()
)


# In[168]:


rxnorm_to_ings123 = (first_second_relations.
 set_index(['RxNorm_concept_id_2','RxNorm_concept_code_2',
            'RxNorm_concept_name_2','RxNorm_concept_class_id_2']).
 join(second_third_relations.
      set_index(['RxNorm_concept_id_2','RxNorm_concept_code_2',
                 'RxNorm_concept_name_2','RxNorm_concept_class_id_2'])
     ).
 query('RxNorm_concept_class_id_3=="Ingredient" & '+
       '(RxNorm_concept_class_id_1!=RxNorm_concept_class_id_3)').
                  reset_index()
)
print(rxnorm_to_ings123.shape)
print(rxnorm_to_ings123.head())


# In[169]:


len(np.intersect1d(
    rxnorm_to_ings123.RxNorm_concept_id_1.dropna().astype(int).unique(),
    all_openFDA_rxnorm_concept_ids
))/len(all_openFDA_rxnorm_concept_ids)


# In[170]:


(rxnorm_to_ings123.
loc[:,['RxNorm_concept_name_1','RxNorm_concept_name_3']].
drop_duplicates()
).head()


# In[171]:


(rxnorm_to_ings123.
loc[:,['RxNorm_concept_class_id_1','RxNorm_concept_class_id_2',
       'RxNorm_concept_class_id_3']].
 drop_duplicates()
)


# In[172]:


rxnorm_to_ings123_to_add = (rxnorm_to_ings123.
loc[:,['RxNorm_concept_id_1','RxNorm_concept_code_1',
       'RxNorm_concept_name_1','RxNorm_concept_class_id_1',
       'RxNorm_concept_id_3','RxNorm_concept_code_3',
       'RxNorm_concept_name_3','RxNorm_concept_class_id_3']].
 drop_duplicates().
 rename(
     columns={
         'RxNorm_concept_id_3' : 'RxNorm_concept_id_2',
         'RxNorm_concept_code_3' : 'RxNorm_concept_code_2',
         'RxNorm_concept_name_3' : 'RxNorm_concept_name_2',
         'RxNorm_concept_class_id_3' : 'RxNorm_concept_class_id_2'
     })
                            .drop_duplicates()
)
print(rxnorm_to_ings123_to_add.shape)
rxnorm_to_ings123_to_add.head()


# In[173]:


rxnorm_to_ings1234 = (first_second_relations.
 set_index(['RxNorm_concept_id_2','RxNorm_concept_code_2',
            'RxNorm_concept_name_2','RxNorm_concept_class_id_2']).
 join(second_third_relations.
      set_index(['RxNorm_concept_id_2','RxNorm_concept_code_2',
                 'RxNorm_concept_name_2','RxNorm_concept_class_id_2'])
     ).
 query('RxNorm_concept_class_id_3!="Ingredient" & '+
       '(RxNorm_concept_class_id_1!=RxNorm_concept_class_id_3)').
                  reset_index().
                      set_index(
                          ['RxNorm_concept_id_3','RxNorm_concept_code_3',
                           'RxNorm_concept_name_3','RxNorm_concept_class_id_3']
                      ).
                      join(third_fourth_relations.
                          set_index(
                          ['RxNorm_concept_id_3','RxNorm_concept_code_3',
                           'RxNorm_concept_name_3','RxNorm_concept_class_id_3']
                          )
                          ).
 query('RxNorm_concept_class_id_4=="Ingredient"').
                      reset_index()
)
rxnorm_to_ings1234 = rxnorm_to_ings1234.reindex(np.sort(rxnorm_to_ings1234.columns),axis=1)
print(rxnorm_to_ings1234.shape)
rxnorm_to_ings1234.head()


# In[174]:


(rxnorm_to_ings1234.
loc[:,['RxNorm_concept_name_1','RxNorm_concept_name_4']].
drop_duplicates()
).head()
len(np.intersect1d(rxnorm_to_ings1234.RxNorm_concept_id_1.dropna().astype(int).unique(),
                  all_openFDA_rxnorm_concept_ids
                  ))/len(all_openFDA_rxnorm_concept_ids)


# In[175]:


(rxnorm_to_ings1234.
loc[:,['RxNorm_concept_class_id_1','RxNorm_concept_class_id_2',
       'RxNorm_concept_class_id_3','RxNorm_concept_class_id_4']].
 drop_duplicates()
)


# In[176]:


rxnorm_to_ings1234_to_add = (rxnorm_to_ings1234.
loc[:,['RxNorm_concept_id_1','RxNorm_concept_code_1',
       'RxNorm_concept_name_1','RxNorm_concept_class_id_1',
       'RxNorm_concept_id_4','RxNorm_concept_code_4',
       'RxNorm_concept_name_4','RxNorm_concept_class_id_4']].
 drop_duplicates().
 rename(
     columns={
         'RxNorm_concept_id_4' : 'RxNorm_concept_id_2',
         'RxNorm_concept_code_4' : 'RxNorm_concept_code_2',
         'RxNorm_concept_name_4' : 'RxNorm_concept_name_2',
         'RxNorm_concept_class_id_4' : 'RxNorm_concept_class_id_2'
     })
                            .drop_duplicates()
)
print(rxnorm_to_ings1234_to_add.shape)
rxnorm_to_ings1234_to_add.head()


# In[177]:


len(
    np.intersect1d(
        np.union1d(
            rxnorm_to_ings123.RxNorm_concept_id_1.dropna().astype(int).unique(),
            rxnorm_to_ings1234.RxNorm_concept_id_1.dropna().astype(int).unique()
        ),
        all_openFDA_rxnorm_concept_ids
    )
                  )/len(all_openFDA_rxnorm_concept_ids)


# In[178]:


rxnorm_to_ings12345 = (first_second_relations.
 set_index(['RxNorm_concept_id_2','RxNorm_concept_code_2',
            'RxNorm_concept_name_2','RxNorm_concept_class_id_2']).
 join(second_third_relations.
      set_index(['RxNorm_concept_id_2','RxNorm_concept_code_2',
                 'RxNorm_concept_name_2','RxNorm_concept_class_id_2'])
     ).
 query('RxNorm_concept_class_id_3!="Ingredient" & '+
       '(RxNorm_concept_class_id_1!=RxNorm_concept_class_id_3)').
                  reset_index().
                      set_index(
                          ['RxNorm_concept_id_3','RxNorm_concept_code_3',
                           'RxNorm_concept_name_3','RxNorm_concept_class_id_3']
                      ).
                      join(third_fourth_relations.
                          set_index(
                          ['RxNorm_concept_id_3','RxNorm_concept_code_3',
                           'RxNorm_concept_name_3','RxNorm_concept_class_id_3']
                          )
                          ).
 query('RxNorm_concept_class_id_4!="Ingredient" & '+
       '(RxNorm_concept_class_id_2!=RxNorm_concept_class_id_4)').
                      reset_index().
                      set_index(
                          ['RxNorm_concept_id_4','RxNorm_concept_code_4',
                           'RxNorm_concept_name_4','RxNorm_concept_class_id_4']
                      ).
                      join(fourth_fifth_relations.
                          set_index(
                          ['RxNorm_concept_id_4','RxNorm_concept_code_4',
                           'RxNorm_concept_name_4','RxNorm_concept_class_id_4']
                          )
                          ).
 query('RxNorm_concept_class_id_5=="Ingredient"').
                      reset_index()
)
rxnorm_to_ings12345 = rxnorm_to_ings12345.reindex(np.sort(rxnorm_to_ings12345.columns),axis=1)
print(rxnorm_to_ings12345.shape)
rxnorm_to_ings12345.head()


# In[179]:


(rxnorm_to_ings12345.
loc[:,['RxNorm_concept_name_1','RxNorm_concept_name_5']].
drop_duplicates()
).head()
len(np.intersect1d(rxnorm_to_ings12345.RxNorm_concept_id_1.dropna().astype(int).unique(),
                  all_openFDA_rxnorm_concept_ids
                  ))/len(all_openFDA_rxnorm_concept_ids)


# In[180]:


(rxnorm_to_ings12345.
loc[:,['RxNorm_concept_class_id_1','RxNorm_concept_class_id_2',
       'RxNorm_concept_class_id_3','RxNorm_concept_class_id_4',
       'RxNorm_concept_class_id_5']].
 drop_duplicates()
)


# In[181]:


rxnorm_to_ings12345_to_add = (rxnorm_to_ings12345.
loc[:,['RxNorm_concept_id_1','RxNorm_concept_code_1',
       'RxNorm_concept_name_1','RxNorm_concept_class_id_1',
       'RxNorm_concept_id_5','RxNorm_concept_code_5',
       'RxNorm_concept_name_5','RxNorm_concept_class_id_5']].
 drop_duplicates().
 rename(
     columns={
         'RxNorm_concept_id_5' : 'RxNorm_concept_id_2',
         'RxNorm_concept_code_5' : 'RxNorm_concept_code_2',
         'RxNorm_concept_name_5' : 'RxNorm_concept_name_2',
         'RxNorm_concept_class_id_5' : 'RxNorm_concept_class_id_2'
     })
                            .drop_duplicates()
)
print(rxnorm_to_ings12345_to_add.shape)
rxnorm_to_ings12345_to_add.head()


# In[182]:


len(
    np.intersect1d(
        np.union1d(
            np.union1d(
                rxnorm_to_ings123.RxNorm_concept_id_1.dropna().astype(int).unique(),
                rxnorm_to_ings1234.RxNorm_concept_id_1.dropna().astype(int).unique()
            ),
            rxnorm_to_ings12345.RxNorm_concept_id_1.dropna().astype(int).unique()
        ),
        all_openFDA_rxnorm_concept_ids
    )
                  )/len(all_openFDA_rxnorm_concept_ids)


# In[183]:


np.setdiff1d(
        all_openFDA_rxnorm_concept_ids,
    np.union1d(
            np.union1d(
                rxnorm_to_ings123.RxNorm_concept_id_1.dropna().astype(int).unique(),
                rxnorm_to_ings1234.RxNorm_concept_id_1.dropna().astype(int).unique()
            ),
            rxnorm_to_ings12345.RxNorm_concept_id_1.dropna().astype(int).unique()
        )
    )


# In[184]:


rxnorm_to_ings123456 = (first_second_relations.
 set_index(['RxNorm_concept_id_2','RxNorm_concept_code_2',
            'RxNorm_concept_name_2','RxNorm_concept_class_id_2']).
 join(second_third_relations.
      set_index(['RxNorm_concept_id_2','RxNorm_concept_code_2',
                 'RxNorm_concept_name_2','RxNorm_concept_class_id_2'])
     ).
 query('RxNorm_concept_class_id_3!="Ingredient" & '+
       '(RxNorm_concept_class_id_1!=RxNorm_concept_class_id_3)').
                  reset_index().
                      set_index(
                          ['RxNorm_concept_id_3','RxNorm_concept_code_3',
                           'RxNorm_concept_name_3','RxNorm_concept_class_id_3']
                      ).
                      join(third_fourth_relations.
                          set_index(
                          ['RxNorm_concept_id_3','RxNorm_concept_code_3',
                           'RxNorm_concept_name_3','RxNorm_concept_class_id_3']
                          )
                          ).
 query('RxNorm_concept_class_id_4!="Ingredient" & '+
       '(RxNorm_concept_class_id_2!=RxNorm_concept_class_id_4)').
                      reset_index().
                      set_index(
                          ['RxNorm_concept_id_4','RxNorm_concept_code_4',
                           'RxNorm_concept_name_4','RxNorm_concept_class_id_4']
                      ).
                      join(fourth_fifth_relations.
                          set_index(
                          ['RxNorm_concept_id_4','RxNorm_concept_code_4',
                           'RxNorm_concept_name_4','RxNorm_concept_class_id_4']
                          )
                          ).
 query('RxNorm_concept_class_id_5!="Ingredient" & '+
       '(RxNorm_concept_class_id_3!=RxNorm_concept_class_id_5)').
                      reset_index().
                      set_index(
                          ['RxNorm_concept_id_5','RxNorm_concept_code_5',
                           'RxNorm_concept_name_5','RxNorm_concept_class_id_5']
                      ).
                      join(fifth_sixth_relations.
                          set_index(
                          ['RxNorm_concept_id_5','RxNorm_concept_code_5',
                           'RxNorm_concept_name_5','RxNorm_concept_class_id_5']
                          )
                          ).
 query('RxNorm_concept_class_id_6=="Ingredient"').
                      reset_index()
)
rxnorm_to_ings123456 = rxnorm_to_ings123456.reindex(np.sort(rxnorm_to_ings123456.columns),axis=1)
print(rxnorm_to_ings123456.shape)
rxnorm_to_ings123456.head()


# In[185]:


(rxnorm_to_ings123456.
loc[:,['RxNorm_concept_name_1','RxNorm_concept_name_6']].
drop_duplicates()
).head()
len(np.intersect1d(rxnorm_to_ings123456.RxNorm_concept_id_1.dropna().astype(int).unique(),
                  all_openFDA_rxnorm_concept_ids
                  ))/len(all_openFDA_rxnorm_concept_ids)


# In[186]:


(rxnorm_to_ings123456.
loc[:,['RxNorm_concept_class_id_1','RxNorm_concept_class_id_2',
       'RxNorm_concept_class_id_3','RxNorm_concept_class_id_4',
       'RxNorm_concept_class_id_5','RxNorm_concept_class_id_6']].
 drop_duplicates()
)


# In[187]:


rxnorm_to_ings123456_to_add = (rxnorm_to_ings123456.
loc[:,['RxNorm_concept_id_1','RxNorm_concept_code_1',
       'RxNorm_concept_name_1','RxNorm_concept_class_id_1',
       'RxNorm_concept_id_6','RxNorm_concept_code_6',
       'RxNorm_concept_name_6','RxNorm_concept_class_id_6']].
 drop_duplicates().
 rename(
     columns={
         'RxNorm_concept_id_6' : 'RxNorm_concept_id_2',
         'RxNorm_concept_code_6' : 'RxNorm_concept_code_2',
         'RxNorm_concept_name_6' : 'RxNorm_concept_name_2',
         'RxNorm_concept_class_id_6' : 'RxNorm_concept_class_id_2'
     }).
                            drop_duplicates()
)
print(rxnorm_to_ings123456_to_add.shape)
rxnorm_to_ings123456_to_add.head()


# In[188]:


len(
    np.intersect1d(
        np.union1d(
            np.union1d(
                np.union1d(
                    rxnorm_to_ings123.RxNorm_concept_id_1.dropna().astype(int).unique(),
                    rxnorm_to_ings1234.RxNorm_concept_id_1.dropna().astype(int).unique()
                ),
                rxnorm_to_ings12345.RxNorm_concept_id_1.dropna().astype(int).unique()
            ),
            rxnorm_to_ings123456.RxNorm_concept_id_1.dropna().astype(int).unique()
        ),
        all_openFDA_rxnorm_concept_ids
    )
                  )/len(all_openFDA_rxnorm_concept_ids)


# In[189]:


np.setdiff1d(
        all_openFDA_rxnorm_concept_ids,
        np.union1d(
            np.union1d(
                np.union1d(
                    rxnorm_to_ings123.RxNorm_concept_id_1.dropna().astype(int).unique(),
                    rxnorm_to_ings1234.RxNorm_concept_id_1.dropna().astype(int).unique()
                ),
                rxnorm_to_ings12345.RxNorm_concept_id_1.dropna().astype(int).unique()
            ),
            rxnorm_to_ings123456.RxNorm_concept_id_1.dropna().astype(int).unique()
        )
)


# In[190]:


rxnorm_to_ings_all = pd.concat(
    [
        rxnorm_to_ings123_to_add,
        rxnorm_to_ings1234_to_add,
        rxnorm_to_ings12345_to_add,
        rxnorm_to_ings123456_to_add
    ]
).dropna().drop_duplicates()
rxnorm_to_ings_all.RxNorm_concept_id_2 = rxnorm_to_ings_all.RxNorm_concept_id_2.astype(int)
print(rxnorm_to_ings_all.shape)
rxnorm_to_ings_all.head()


# In[191]:


len(
    np.intersect1d(
        rxnorm_to_ings_all.RxNorm_concept_id_1,
        all_openFDA_rxnorm_concept_ids
    )
)/len(all_openFDA_rxnorm_concept_ids)


# In[202]:


standard_drug = (pd.
                 read_csv(er_dir+'standard_drugs.csv.gz',
                          compression='gzip',
                          dtype={
                              'safetyreportid' : 'str'
                          })
                )
standard_drug.RxNorm_concept_id = standard_drug.RxNorm_concept_id.astype(int)
all_reports = standard_drug.safetyreportid.astype(str).unique()
print(standard_drug.shape)
standard_drug.head()


# In[203]:


standard_drug_ingredients = ((standard_drug.
  loc[:,['RxNorm_concept_id','safetyreportid']].
  drop_duplicates().
set_index(
    [
        'RxNorm_concept_id'
    ]
)
).join(rxnorm_to_ings_all.
       loc[:,['RxNorm_concept_id_1','RxNorm_concept_id_2',
             'RxNorm_concept_code_2','RxNorm_concept_name_2',
             'RxNorm_concept_class_id_2']].
       drop_duplicates().
set_index(
    [
        'RxNorm_concept_id_1'
    ]
)
).drop_duplicates().
 rename(
     columns={
         'RxNorm_concept_id_2':'RxNorm_concept_id',
         'RxNorm_concept_code_2':'RxNorm_concept_code',
         'RxNorm_concept_name_2':'RxNorm_concept_name',
         'RxNorm_concept_class_id_2':'RxNorm_concept_class_id'
     }).
                             reset_index(drop=True).
                             dropna().
                             drop_duplicates()
       )
standard_drug_ingredients = (standard_drug_ingredients.
                             reindex(np.sort(standard_drug_ingredients.columns),axis=1)
                            )
print(standard_drug_ingredients.shape)
standard_drug_ingredients.head()


# In[204]:


print(len(
    np.intersect1d(
        standard_drug_ingredients.safetyreportid.astype(str).unique(),
        all_reports
    )
)/len(all_reports))


# In[ ]:


(standard_drug_ingredients.
 to_csv(er_dir+'standard_drugs_rxnorm_ingredients.csv.gz',compression='gzip',index=False))


# ### standard_reactions_meddra_relationships

# In[46]:


standard_reactions = (pd.
                      read_csv(er_dir+'standard_reactions.csv.gz',
                               compression="gzip",
                               dtype={
                                   'safetyreportid' : 'str'
                               }
                              )
                     )
all_reports = (standard_reactions.safetyreportid.unique())
print(standard_reactions.shape)
print(standard_reactions.head())


# In[47]:


reactions = standard_reactions.MedDRA_concept_id.astype(int).unique()
print(len(reactions))
meddra_concept_ids = concept.query('vocabulary_id=="MedDRA"').concept_id.astype(int).unique()
len(meddra_concept_ids)

intersect = np.intersect1d(reactions,meddra_concept_ids)
print(len(intersect))
print(len(intersect)/len(reactions))


# In[48]:


meddra_concept = concept.query('vocabulary_id=="MedDRA"')
meddra_concept.concept_id = meddra_concept.concept_id.astype(int)
all_meddra_concept_ids = meddra_concept.concept_id.unique()

r = (concept_relationship.
     copy().
     loc[:,['concept_id_1','concept_id_2','relationship_id']].
     drop_duplicates()
    )
r.concept_id_1 = r.concept_id_1.astype(int)
r.concept_id_2 = r.concept_id_2.astype(int)


# In[49]:


(r.
query('concept_id_1 in @all_meddra_concept_ids & '+
     'concept_id_2 in @all_meddra_concept_ids').
relationship_id.value_counts()
)


# In[50]:


c = meddra_concept.copy()

all_meddra_relationships = (r.
 query('concept_id_1 in @meddra_concept_ids & '+\
       'concept_id_2 in @meddra_concept_ids').
 set_index('concept_id_1').
 join(
     c. # standard concepts for 1
     query('vocabulary_id=="MedDRA"').
     loc[:,['concept_id','concept_code','concept_name','concept_class_id']].
     drop_duplicates().
     set_index('concept_id')
    ).
 rename_axis('MedDRA_concept_id_1').
 reset_index().
 rename(
     columns={
         'concept_code' : 'MedDRA_concept_code_1',
         'concept_class_id' : 'MedDRA_concept_class_id_1',
         'concept_name' : 'MedDRA_concept_name_1',
         'concept_id_2' : 'MedDRA_concept_id_2',
         'relationship_id' : 'relationship_id_12'
     }
 ).
set_index('MedDRA_concept_id_2').
 join(
     c. # standard concepts for 2
     query('vocabulary_id=="MedDRA"').
     loc[:,['concept_id','concept_code','concept_name','concept_class_id']].
     drop_duplicates().
     set_index('concept_id')
    ).
 rename_axis('MedDRA_concept_id_2').
 reset_index().
 rename(
     columns={
         'concept_code' : 'MedDRA_concept_code_2',
         'concept_class_id' : 'MedDRA_concept_class_id_2',
         'concept_name' : 'MedDRA_concept_name_2'
     }
 ))
all_meddra_relationships = (all_meddra_relationships.
                            reindex(np.sort(all_meddra_relationships.columns),axis=1)
                           )
print(all_meddra_relationships.shape)
print(all_meddra_relationships.head())


# In[51]:


print(all_meddra_relationships.MedDRA_concept_class_id_1.value_counts())
print(all_meddra_relationships.MedDRA_concept_class_id_2.value_counts())


# In[52]:


all_meddra_relationships.MedDRA_concept_id_1 = (all_meddra_relationships.
                                                  MedDRA_concept_id_1.
                                                  astype(int)
                                                 )
all_meddra_relationships.MedDRA_concept_code_1 = (all_meddra_relationships.
                                                  MedDRA_concept_code_1.
                                                  astype(int)
                                                 )
all_meddra_relationships.MedDRA_concept_id_2 = (all_meddra_relationships.
                                                  MedDRA_concept_id_2.
                                                  astype(int)
                                                 )
all_meddra_relationships.MedDRA_concept_code_2 = (all_meddra_relationships.
                                                  MedDRA_concept_code_2.
                                                  astype(int)
                                                 )


# In[53]:


first_rxs = reactions
first_relations = (all_meddra_relationships.
                   query('MedDRA_concept_id_1 in @first_rxs & '+
                         'MedDRA_concept_class_id_2=="HLT"')
                  ).reset_index(drop=True)
first_relations = (first_relations[
    first_relations.MedDRA_concept_id_1!=first_relations.MedDRA_concept_id_2
])
print(first_relations.shape)
print(first_relations.head())
print(first_relations.MedDRA_concept_class_id_2.value_counts())


# In[54]:


second_rxs = first_relations.MedDRA_concept_id_2.unique()
second_relations = (all_meddra_relationships.
                    query('MedDRA_concept_id_1 in @second_rxs & '+
                         'MedDRA_concept_class_id_2=="HLGT"').
                    rename(columns={
                        'MedDRA_concept_id_2' : 'MedDRA_concept_id_3',
                        'MedDRA_concept_code_2' : 'MedDRA_concept_code_3',
                        'MedDRA_concept_name_2' : 'MedDRA_concept_name_3',
                        'MedDRA_concept_class_id_2' : 'MedDRA_concept_class_id_3',
                        'MedDRA_concept_id_1' : 'MedDRA_concept_id_2',
                        'MedDRA_concept_code_1' : 'MedDRA_concept_code_2',
                        'MedDRA_concept_name_1' : 'MedDRA_concept_name_2',
                        'MedDRA_concept_class_id_1' : 'MedDRA_concept_class_id_2',
                        'relationship_id_12' : 'relationship_id_23'
                    }
                          )
                  ).reset_index(drop=True)
second_relations = (second_relations[
    second_relations.MedDRA_concept_id_2!=second_relations.MedDRA_concept_id_3
])
print(second_relations.shape)
print(second_relations.head())
print(second_relations.MedDRA_concept_class_id_2.value_counts())
print(second_relations.MedDRA_concept_class_id_3.value_counts())


# In[55]:


third_rxs = second_relations.MedDRA_concept_id_3.unique()
third_relations = (all_meddra_relationships.
                    query('MedDRA_concept_id_1 in @third_rxs & '+
                         'MedDRA_concept_class_id_2=="SOC"').
                    rename(columns={
                        'MedDRA_concept_id_2' : 'MedDRA_concept_id_4',
                        'MedDRA_concept_code_2' : 'MedDRA_concept_code_4',
                        'MedDRA_concept_name_2' : 'MedDRA_concept_name_4',
                        'MedDRA_concept_class_id_2' : 'MedDRA_concept_class_id_4',
                        'MedDRA_concept_id_1' : 'MedDRA_concept_id_3',
                        'MedDRA_concept_code_1' : 'MedDRA_concept_code_3',
                        'MedDRA_concept_name_1' : 'MedDRA_concept_name_3',
                        'MedDRA_concept_class_id_1' : 'MedDRA_concept_class_id_3',
                        'relationship_id_12' : 'relationship_id_34'
                    }
                          )
                  ).reset_index(drop=True)
third_relations = (third_relations[
    third_relations.MedDRA_concept_id_3!=third_relations.MedDRA_concept_id_4
])
print(third_relations.shape)
print(third_relations.head())
print(third_relations.MedDRA_concept_class_id_3.value_counts())
print(third_relations.MedDRA_concept_class_id_4.value_counts())


# In[56]:


first_second_third_relations = (first_relations.
 set_index('MedDRA_concept_id_2').
 join(second_relations.
      loc[:,['MedDRA_concept_id_2','MedDRA_concept_id_3',
             'MedDRA_concept_name_3','MedDRA_concept_class_id_3',
             'MedDRA_concept_code_3','relationship_id_23']].
      set_index('MedDRA_concept_id_2')
     ).
 reset_index()
)
first_second_third_relations = (first_second_third_relations.
 reindex(np.sort(first_second_third_relations.columns),
         axis=1)
)
first_second_third_relations['MedDRA_concept_id_3'] = first_second_third_relations['MedDRA_concept_id_3'].astype(int)
print(first_second_third_relations.shape)
print(first_second_third_relations.head())
print(first_second_third_relations.MedDRA_concept_class_id_1.value_counts())
print(first_second_third_relations.MedDRA_concept_class_id_2.value_counts())
print(first_second_third_relations.MedDRA_concept_class_id_3.value_counts())


# In[57]:


first_second_third_fourth_relations = (first_relations.
 set_index('MedDRA_concept_id_2').
 join(second_relations.
      loc[:,['MedDRA_concept_id_2','MedDRA_concept_id_3',
             'MedDRA_concept_name_3','MedDRA_concept_class_id_3',
             'MedDRA_concept_code_3','relationship_id_23']].
      drop_duplicates().
      set_index('MedDRA_concept_id_2')
     ).
 reset_index().
 set_index('MedDRA_concept_id_3').
 join(third_relations.
      loc[:,['MedDRA_concept_id_3','MedDRA_concept_id_4',
             'MedDRA_concept_name_4','MedDRA_concept_class_id_4',
             'MedDRA_concept_code_4','relationship_id_34']].
      drop_duplicates().
      set_index('MedDRA_concept_id_3')
     ).
 reset_index()
)
first_second_third_fourth_relations = (first_second_third_fourth_relations.
 reindex(np.sort(first_second_third_fourth_relations.columns),
         axis=1)
)
first_second_third_fourth_relations['MedDRA_concept_id_4'] = first_second_third_fourth_relations['MedDRA_concept_id_4'].astype(int)
print(first_second_third_fourth_relations.shape)
print(first_second_third_fourth_relations.head())
print(first_second_third_fourth_relations.MedDRA_concept_class_id_1.value_counts())
print(first_second_third_fourth_relations.MedDRA_concept_class_id_2.value_counts())
print(first_second_third_fourth_relations.MedDRA_concept_class_id_3.value_counts())
print(first_second_third_fourth_relations.MedDRA_concept_class_id_4.value_counts())


# In[58]:


len(np.setdiff1d(reactions,first_second_third_fourth_relations.MedDRA_concept_id_1.unique()))


# In[59]:


left_over = np.setdiff1d(reactions,first_second_third_fourth_relations.MedDRA_concept_id_1.unique())

all_meddra_relationships.query('MedDRA_concept_id_1 in @left_over')


# In[60]:


df1 = (standard_reactions.
       loc[:,['MedDRA_concept_id']].
       drop_duplicates().
       dropna().
       set_index('MedDRA_concept_id')
      )
print(df1.shape)


# In[61]:


df2 = (first_second_third_fourth_relations.
       set_index('MedDRA_concept_id_1')
      )
print(df2.shape)


# In[62]:


joined = df1.join(df2).rename_axis('MedDRA_concept_id_1').reset_index().dropna()
joined = joined.reindex(np.sort(joined.columns),axis=1)
joined.MedDRA_concept_id_1 = joined.MedDRA_concept_id_1.astype(int).copy()
joined.MedDRA_concept_id_2 = joined.MedDRA_concept_id_2.astype(int).copy()
joined.MedDRA_concept_id_3 = joined.MedDRA_concept_id_3.astype(int).copy()
joined.MedDRA_concept_id_4 = joined.MedDRA_concept_id_4.astype(int).copy()
joined.MedDRA_concept_code_1 = joined.MedDRA_concept_code_1.astype(int).copy()
joined.MedDRA_concept_code_2 = joined.MedDRA_concept_code_2.astype(int).copy()
joined.MedDRA_concept_code_3 = joined.MedDRA_concept_code_3.astype(int).copy()
joined.MedDRA_concept_code_4 = joined.MedDRA_concept_code_4.astype(int).copy()
print(joined.shape)
print(joined.head())


# In[63]:


print(joined.MedDRA_concept_class_id_1.value_counts())
print(joined.MedDRA_concept_class_id_2.value_counts())
print(joined.MedDRA_concept_class_id_3.value_counts())
print(joined.MedDRA_concept_class_id_4.value_counts())


# In[23]:


(joined.
 to_csv(er_dir+'standard_reactions_meddra_relationships.csv.gz',
        compression='gzip',index=False)
)


# In[64]:


pt_to_soc = (joined.
loc[:,['MedDRA_concept_id_1','MedDRA_concept_code_1',
       'MedDRA_concept_name_1','MedDRA_concept_class_id_1',
       'MedDRA_concept_id_4','MedDRA_concept_code_4',
       'MedDRA_concept_name_4','MedDRA_concept_class_id_4']].
query('MedDRA_concept_class_id_4=="SOC"').
              drop_duplicates()
)
print(pt_to_soc.shape)
print(pt_to_soc.head())


# In[65]:


pt_to_hlgt = (joined.
loc[:,['MedDRA_concept_id_1','MedDRA_concept_code_1',
       'MedDRA_concept_name_1','MedDRA_concept_class_id_1',
       'MedDRA_concept_id_3','MedDRA_concept_code_3',
       'MedDRA_concept_name_3','MedDRA_concept_class_id_3']].
query('MedDRA_concept_class_id_3=="HLGT"').
              drop_duplicates()
)
print(pt_to_hlgt.shape)
print(pt_to_hlgt.head())


# In[66]:


pt_to_hlt = (joined.
loc[:,['MedDRA_concept_id_1','MedDRA_concept_code_1',
       'MedDRA_concept_name_1','MedDRA_concept_class_id_1',
       'MedDRA_concept_id_2','MedDRA_concept_code_2',
       'MedDRA_concept_name_2','MedDRA_concept_class_id_2']].
query('MedDRA_concept_class_id_2=="HLT"').
             drop_duplicates()
)
print(pt_to_hlt.shape)
print(pt_to_hlt.head())


# In[67]:


standard_reactions_pt_to_hlt = (standard_reactions.
 loc[:,['safetyreportid','MedDRA_concept_id']].
 drop_duplicates().
 set_index(['MedDRA_concept_id']).
 join(pt_to_hlt.
      loc[:,['MedDRA_concept_id_1','MedDRA_concept_id_2',
           'MedDRA_concept_code_2','MedDRA_concept_name_2',
           'MedDRA_concept_class_id_2']].
      set_index('MedDRA_concept_id_1')
     ).
 reset_index(drop=True).
 rename(
     columns={
         'MedDRA_concept_id_2' : 'MedDRA_concept_id',
         'MedDRA_concept_code_2' : 'MedDRA_concept_code',
         'MedDRA_concept_name_2' : 'MedDRA_concept_name',
         'MedDRA_concept_class_id_2' : 'MedDRA_concept_class_id'
     }
 ).
 dropna().
 drop_duplicates()
)
standard_reactions_pt_to_hlt = (standard_reactions_pt_to_hlt.
                               reindex(np.sort(standard_reactions_pt_to_hlt.columns),axis=1)
                               )
print(standard_reactions_pt_to_hlt.shape)
print(standard_reactions_pt_to_hlt.head())


# In[68]:


print(
    len(
        np.intersect1d(
            all_reports,
            standard_reactions_pt_to_hlt.safetyreportid.astype(str).unique()
                        )
    )/len(all_reports)
)


# In[29]:


(standard_reactions_pt_to_hlt.
 to_csv(er_dir+'standard_reactions_meddra_hlt.csv.gz',
        compression='gzip',index=False)
)


# In[69]:


standard_reactions_pt_to_hlgt = (standard_reactions.
 loc[:,['safetyreportid','MedDRA_concept_id']].
 drop_duplicates().
 set_index(['MedDRA_concept_id']).
 join(pt_to_hlgt.
      loc[:,['MedDRA_concept_id_1','MedDRA_concept_id_3',
           'MedDRA_concept_code_3','MedDRA_concept_name_3',
           'MedDRA_concept_class_id_3']].
      set_index('MedDRA_concept_id_1')
     ).
 reset_index(drop=True).
 rename(
     columns={
         'MedDRA_concept_id_3' : 'MedDRA_concept_id',
         'MedDRA_concept_code_3' : 'MedDRA_concept_code',
         'MedDRA_concept_name_3' : 'MedDRA_concept_name',
         'MedDRA_concept_class_id_3' : 'MedDRA_concept_class_id'
     }
 ).
 dropna().
 drop_duplicates()
)
standard_reactions_pt_to_hlgt = (standard_reactions_pt_to_hlgt.
                               reindex(np.sort(standard_reactions_pt_to_hlgt.columns),axis=1)
                               )
print(standard_reactions_pt_to_hlgt.shape)
print(standard_reactions_pt_to_hlgt.head())


# In[70]:


print(
    len(
        np.intersect1d(
            all_reports,
            standard_reactions_pt_to_hlgt.safetyreportid.astype(str).unique()
                        )
    )/len(all_reports)
)


# In[32]:


(standard_reactions_pt_to_hlgt.
 to_csv(er_dir+'standard_reactions_meddra_hlgt.csv.gz',
        compression='gzip',index=False)
)


# In[71]:


standard_reactions_pt_to_soc = (standard_reactions.
 loc[:,['safetyreportid','MedDRA_concept_id']].
 drop_duplicates().
 set_index(['MedDRA_concept_id']).
 join(pt_to_soc.
      loc[:,['MedDRA_concept_id_1','MedDRA_concept_id_4',
           'MedDRA_concept_code_4','MedDRA_concept_name_4',
           'MedDRA_concept_class_id_4']].
      set_index('MedDRA_concept_id_1')
     ).
 reset_index(drop=True).
 rename(
     columns={
         'MedDRA_concept_id_4' : 'MedDRA_concept_id',
         'MedDRA_concept_code_4' : 'MedDRA_concept_code',
         'MedDRA_concept_name_4' : 'MedDRA_concept_name',
         'MedDRA_concept_class_id_4' : 'MedDRA_concept_class_id'
     }
 ).
 dropna().
 drop_duplicates()
)
standard_reactions_pt_to_soc = (standard_reactions_pt_to_soc.
                               reindex(np.sort(standard_reactions_pt_to_soc.columns),axis=1)
                               )
print(standard_reactions_pt_to_soc.shape)
print(standard_reactions_pt_to_soc.head())


# In[72]:


print(
    len(
        np.intersect1d(
            all_reports,
            standard_reactions_pt_to_soc.safetyreportid.astype(str).unique()
                        )
    )/len(all_reports)
)


# In[35]:


(standard_reactions_pt_to_soc.
 to_csv(er_dir+'standard_reactions_meddra_soc.csv.gz',
        compression='gzip',index=False)
)


# In[73]:


del c
del r
del first_relations
del second_relations
del first_second_third_relations
del all_meddra_relationships
del meddra_concept
del df1
del df2
del joined
del standard_reactions_pt_to_soc
del standard_reactions_pt_to_hlgt
del standard_reactions_pt_to_hlt


# ### standard_reactions_snomed

# In[86]:


standard_reactions_meddra_relationships = (pd.read_csv(
    er_dir+'standard_reactions_meddra_relationships.csv.gz',
    compression='gzip',
    dtype={
    'safetyreportid' : 'str'
    })
    )

print(standard_reactions_meddra_relationships.MedDRA_concept_id_1.nunique())
print(standard_reactions_meddra_relationships.MedDRA_concept_id_2.nunique())
print(standard_reactions_meddra_relationships.MedDRA_concept_id_3.nunique())
print(standard_reactions_meddra_relationships.MedDRA_concept_id_4.nunique())

standard_reactions_meddra_relationships.MedDRA_concept_id_1 = standard_reactions_meddra_relationships.MedDRA_concept_id_1.astype(int)

standard_reactions_meddra_relationships.MedDRA_concept_id_2 = standard_reactions_meddra_relationships.MedDRA_concept_id_2.astype(int)

standard_reactions_meddra_relationships.MedDRA_concept_id_3 = standard_reactions_meddra_relationships.MedDRA_concept_id_3.astype(int)

standard_reactions_meddra_relationships.MedDRA_concept_id_4 = standard_reactions_meddra_relationships.MedDRA_concept_id_4.astype(int)

print(standard_reactions_meddra_relationships.shape)
print(standard_reactions_meddra_relationships.head())


# In[87]:


reactions = standard_reactions_meddra_relationships.MedDRA_concept_id_1.unique()
print(len(reactions))
meddra_concept_ids = concept.query('vocabulary_id=="MedDRA"').concept_id.astype(int).unique()
len(meddra_concept_ids)

intersect = np.intersect1d(reactions,meddra_concept_ids)
print(len(intersect))
print(len(intersect)/len(reactions))


# In[88]:


m_to_s_r = (concept_relationship.
            query('relationship_id=="MedDRA - SNOMED eq"').
            loc[:,['concept_id_1','concept_id_2']].
            drop_duplicates().
            set_index('concept_id_2').
            join(concept.
                 query('vocabulary_id=="SNOMED"').
                 loc[:,['concept_id','concept_code','concept_class_id','concept_name']].
                 drop_duplicates().
                 set_index('concept_id')
                ).
            rename_axis('SNOMED_concept_id').
            reset_index().
            rename(columns={
                'concept_id_1' : 'MedDRA_concept_id',
                'concept_name' : 'SNOMED_concept_name',
                'concept_code' : 'SNOMED_concept_code',
                'concept_class_id' : 'SNOMED_concept_class_id'
            })
)
m_to_s_r.MedDRA_concept_id = m_to_s_r.MedDRA_concept_id.astype(int)
m_to_s_r = m_to_s_r.reindex(np.sort(m_to_s_r.columns),axis=1)
print(m_to_s_r.shape)
print(m_to_s_r.SNOMED_concept_class_id.value_counts())
print(m_to_s_r.head())


# In[89]:


r2s = m_to_s_r.MedDRA_concept_id.unique()


# In[90]:


pts = (standard_reactions_meddra_relationships.
       query('MedDRA_concept_class_id_1=="PT"').
       MedDRA_concept_id_1.
       unique())
print(len(np.intersect1d(pts,r2s))/len(pts))
print(len(np.intersect1d(pts,r2s))/len(r2s))

df = (standard_reactions_meddra_relationships.
      query('MedDRA_concept_id_1 in @r2s'))

print(df.shape)

joinedpt = (df.
           set_index('MedDRA_concept_id_1').
           join(m_to_s_r.
                query('MedDRA_concept_id in @pts').
                set_index('MedDRA_concept_id')
               ).
           rename_axis('MedDRA_concept_id_1').
           reset_index().
           rename(columns={
               'SNOMED_concept_id' : 'SNOMED_concept_id_1',
               'SNOMED_concept_code' : 'SNOMED_concept_code_1',
               'SNOMED_concept_name' : 'SNOMED_concept_name_1',
               'SNOMED_concept_class_id' : 'SNOMED_concept_class_id_1',
           }).
           dropna()
          )
joinedpt = joinedpt.reindex(np.sort(joinedpt.columns),axis=1)
print(joinedpt.shape)
print(joinedpt.head())


# In[91]:


hlts = (joinedpt.
       query('MedDRA_concept_class_id_2=="HLT"').
       MedDRA_concept_id_2.
       unique())
print(len(np.intersect1d(hlts,r2s))/len(hlts))
print(len(np.intersect1d(hlts,r2s))/len(r2s))

df = (joinedpt.copy())

print(df.shape)
print(df.head())


joinedhlt = (df.
           set_index('MedDRA_concept_id_2').
           join(m_to_s_r.
                query('MedDRA_concept_id in @hlts').
                set_index('MedDRA_concept_id')
               ).
           rename_axis('MedDRA_concept_id_2').
           reset_index().
           rename(columns={
               'SNOMED_concept_id' : 'SNOMED_concept_id_2',
               'SNOMED_concept_code' : 'SNOMED_concept_code_2',
               'SNOMED_concept_name' : 'SNOMED_concept_name_2',
               'SNOMED_concept_class_id' : 'SNOMED_concept_class_id_2',
           })
          )
joinedhlt = joinedhlt.reindex(np.sort(joinedhlt.columns),axis=1)
print(joinedhlt.shape)
print(joinedhlt.head())


# In[92]:


hlgts = (joinedhlt.
       query('MedDRA_concept_class_id_3=="HLGT"').
       MedDRA_concept_id_3.
       unique())
print(len(np.intersect1d(hlgts,r2s))/len(hlgts))
print(len(np.intersect1d(hlgts,r2s))/len(r2s))

df = (joinedhlt.copy())

print(df.shape)

joinedhlgt = (df.
           set_index('MedDRA_concept_id_3').
           join(m_to_s_r.
                query('MedDRA_concept_id in @hlgts').
                set_index('MedDRA_concept_id')
               ).
           rename_axis('MedDRA_concept_id_3').
           reset_index().
           drop_duplicates().
           rename(columns={
               'SNOMED_concept_id' : 'SNOMED_concept_id_3',
               'SNOMED_concept_code' : 'SNOMED_concept_code_3',
               'SNOMED_concept_name' : 'SNOMED_concept_name_3',
               'SNOMED_concept_class_id' : 'SNOMED_concept_class_id_3',
           })
          )
joinedhlgt = joinedhlgt.reindex(np.sort(joinedhlgt.columns),axis=1)
print(joinedhlgt.shape)
print(joinedhlgt.head())


# In[93]:


socs = (joinedhlgt.
       query('MedDRA_concept_class_id_4=="SOC"').
       MedDRA_concept_id_4.
       unique())
print(len(np.intersect1d(socs,r2s))/len(socs))
print(len(np.intersect1d(socs,r2s))/len(r2s))

df = (joinedhlgt.copy())

print(df.shape)
print(df.head())
print(m_to_s_r.shape)
print(m_to_s_r.head())

joinedsoc = (df.
           set_index('MedDRA_concept_id_4').
           join(m_to_s_r.
                query('MedDRA_concept_id in @socs').
                set_index('MedDRA_concept_id')
               ).
           rename_axis('MedDRA_concept_id_4').
           reset_index().
           drop_duplicates().
           rename(columns={
               'SNOMED_concept_id' : 'SNOMED_concept_id_4',
               'SNOMED_concept_code' : 'SNOMED_concept_code_4',
               'SNOMED_concept_name' : 'SNOMED_concept_name_4',
               'SNOMED_concept_class_id' : 'SNOMED_concept_class_id_4',
           })
          )
joinedsoc = joinedsoc.reindex(np.sort(joinedsoc.columns),axis=1)
print(joinedsoc.shape)
print(joinedsoc.head())


# In[94]:


smeddraconcepts = joinedpt.MedDRA_concept_id_1.unique()
print(len(smeddraconcepts))
allmeddraconcepts = (standard_reactions_meddra_relationships.
                     query('MedDRA_concept_class_id_1=="PT"').
                     MedDRA_concept_id_1.
                     unique())
print(len(allmeddraconcepts))

print(len(np.intersect1d(smeddraconcepts,allmeddraconcepts))/len(smeddraconcepts))
print(len(np.intersect1d(smeddraconcepts,allmeddraconcepts))/len(allmeddraconcepts))


# In[95]:


smeddraconcepts = joinedhlt.MedDRA_concept_id_2.unique()
print(len(smeddraconcepts))
allmeddraconcepts = (standard_reactions_meddra_relationships.
                     query('MedDRA_concept_class_id_2=="HLT"').
                     MedDRA_concept_id_2.
                     unique())
print(len(allmeddraconcepts))

print(len(np.intersect1d(smeddraconcepts,allmeddraconcepts))/len(smeddraconcepts))
print(len(np.intersect1d(smeddraconcepts,allmeddraconcepts))/len(allmeddraconcepts))


# In[96]:


smeddraconcepts = joinedhlgt.MedDRA_concept_id_3.unique()
print(len(smeddraconcepts))
allmeddraconcepts = (standard_reactions_meddra_relationships.
                     query('MedDRA_concept_class_id_3=="HLGT"').
                     MedDRA_concept_id_3.
                     unique())
print(len(allmeddraconcepts))

print(len(np.intersect1d(smeddraconcepts,allmeddraconcepts))/len(smeddraconcepts))
print(len(np.intersect1d(smeddraconcepts,allmeddraconcepts))/len(allmeddraconcepts))


# In[97]:


smeddraconcepts = joinedsoc.MedDRA_concept_id_4.unique()
print(len(smeddraconcepts))
allmeddraconcepts = (standard_reactions_meddra_relationships.
                     query('MedDRA_concept_class_id_4=="SOC"').
                     MedDRA_concept_id_4.
                     unique())
print(len(allmeddraconcepts))

print(len(np.intersect1d(smeddraconcepts,allmeddraconcepts))/len(smeddraconcepts))
print(len(np.intersect1d(smeddraconcepts,allmeddraconcepts))/len(allmeddraconcepts))


# In[98]:


print(joinedsoc.head())
print(joinedsoc.shape)
print(joinedsoc[joinedsoc.SNOMED_concept_id_1.notnull()].shape)
print(joinedsoc.SNOMED_concept_id_1.nunique())
print(joinedsoc[joinedsoc.SNOMED_concept_id_2.notnull()].shape)
print(joinedsoc.SNOMED_concept_id_2.nunique())
print(joinedsoc[joinedsoc.SNOMED_concept_id_3.notnull()].shape)
print(joinedsoc.SNOMED_concept_id_3.nunique())
print(joinedsoc[joinedsoc.SNOMED_concept_id_4.notnull()].shape)
print(joinedsoc.SNOMED_concept_id_4.nunique())


# In[99]:


joinedsoc.SNOMED_concept_code_1 = joinedsoc.SNOMED_concept_code_1.astype(int)
joinedsoc.SNOMED_concept_code_2 = joinedsoc.SNOMED_concept_code_2.astype(float)
joinedsoc.SNOMED_concept_code_3 = joinedsoc.SNOMED_concept_code_3.astype(float)
joinedsoc.SNOMED_concept_code_4 = joinedsoc.SNOMED_concept_code_4.astype(float)


# In[105]:


standard_reactions = (pd.
                      read_csv(er_dir+'standard_reactions.csv.gz',
                               compression="gzip",
                              dtype={
                                  'safetyreportid' : 'str'
                              })
                     )
all_reports = (standard_reactions.safetyreportid.unique())
print(standard_reactions.shape)
print(standard_reactions.head())


# In[106]:


standard_reactions_meddrapt_to_snomed = (joinedsoc.
 loc[:,['MedDRA_concept_id_1','SNOMED_concept_id_1',
   'SNOMED_concept_code_1','SNOMED_concept_name_1',
   'SNOMED_concept_class_id_1']].
 drop_duplicates().
 rename(
     columns={
         'SNOMED_concept_id_1' : 'SNOMED_concept_id',
         'SNOMED_concept_code_1' : 'SNOMED_concept_code',
         'SNOMED_concept_name_1' : 'SNOMED_concept_name',
         'SNOMED_concept_class_id_1' : 'SNOMED_concept_class_id'
     }
 ).
 set_index('MedDRA_concept_id_1').
 join(standard_reactions.
      drop_duplicates().
      set_index('MedDRA_concept_id')
     ).
 reset_index(drop=True).
 drop(['MedDRA_concept_code','MedDRA_concept_name',
      'MedDRA_concept_class_id'],axis=1).
 dropna()
)
standard_reactions_meddrapt_to_snomed = (standard_reactions_meddrapt_to_snomed.
 reindex(np.sort(standard_reactions_meddrapt_to_snomed.columns),
         axis=1))
print(standard_reactions_meddrapt_to_snomed.shape)
print(standard_reactions_meddrapt_to_snomed.head())


# In[107]:


print(
    len(
        np.intersect1d(
            all_reports,
            standard_reactions_meddrapt_to_snomed.safetyreportid.astype(str).unique()
        )
    )/len(all_reports)
)


# In[22]:


(standard_reactions_meddrapt_to_snomed.
 to_csv(er_dir+'standard_reactions_snomed.csv.gz',
        compression='gzip',index=False)
)


# In[103]:


standard_reactions_meddrahlt_to_snomed = (joinedsoc.
 query('MedDRA_concept_class_id_2=="HLT"').
 loc[:,['MedDRA_concept_id_1','SNOMED_concept_id_2',
   'SNOMED_concept_code_2','SNOMED_concept_name_2',
   'SNOMED_concept_class_id_2']].
 drop_duplicates().
 rename(
     columns={
         'SNOMED_concept_id_2' : 'SNOMED_concept_id',
         'SNOMED_concept_code_2' : 'SNOMED_concept_code',
         'SNOMED_concept_name_2' : 'SNOMED_concept_name',
         'SNOMED_concept_class_id_2' : 'SNOMED_concept_class_id'
     }
 ).
 set_index('MedDRA_concept_id_1').
 join(standard_reactions.
      drop_duplicates().
      set_index('MedDRA_concept_id')
     ).
 rename_axis('MedDRA_concept_id').
 reset_index().
 dropna(subset=['MedDRA_concept_id','SNOMED_concept_id','safetyreportid'])
)
standard_reactions_meddrahlt_to_snomed = (standard_reactions_meddrahlt_to_snomed.
 reindex(np.sort(standard_reactions_meddrahlt_to_snomed.columns),
         axis=1))
print(standard_reactions_meddrahlt_to_snomed.shape)
print(standard_reactions_meddrahlt_to_snomed.head())


# In[24]:


print(
    len(
        np.intersect1d(
            all_reports,
            standard_reactions_meddrahlt_to_snomed.safetyreportid.astype(str).unique()
        )
    )/len(all_reports)
)


# In[25]:


standard_reactions_meddrahlgt_to_snomed = (joinedsoc.
 query('MedDRA_concept_class_id_2=="HLGT"').
 loc[:,['MedDRA_concept_id_1','SNOMED_concept_id_3',
   'SNOMED_concept_code_3','SNOMED_concept_name_3',
   'SNOMED_concept_class_id_2']].
 drop_duplicates().
 rename(
     columns={
         'SNOMED_concept_id_3' : 'SNOMED_concept_id',
         'SNOMED_concept_code_3' : 'SNOMED_concept_code',
         'SNOMED_concept_name_3' : 'SNOMED_concept_name',
         'SNOMED_concept_class_id_3' : 'SNOMED_concept_class_id'
     }
 ).
 set_index('MedDRA_concept_id_1').
 join(standard_reactions.
      drop_duplicates().
      set_index('MedDRA_concept_id')
     ).
 rename_axis('MedDRA_concept_id').
 reset_index().
 dropna(subset=['MedDRA_concept_id','SNOMED_concept_id','safetyreportid'])
)
standard_reactions_meddrahlgt_to_snomed = (standard_reactions_meddrahlgt_to_snomed.
 reindex(np.sort(standard_reactions_meddrahlgt_to_snomed.columns),
         axis=1))
print(standard_reactions_meddrahlgt_to_snomed.shape)
print(standard_reactions_meddrahlgt_to_snomed.head())


# In[26]:


print(
    len(
        np.intersect1d(
            all_reports,
            standard_reactions_meddrahlgt_to_snomed.safetyreportid.astype(str).unique()
        )
    )/len(all_reports)
)


# In[27]:


standard_reactions_meddrasoc_to_snomed = (joinedsoc.
 query('MedDRA_concept_class_id_4=="SOC"').
 loc[:,['MedDRA_concept_id_1','SNOMED_concept_id_4',
   'SNOMED_concept_code_4','SNOMED_concept_name_34',
   'SNOMED_concept_class_id_4']].
 drop_duplicates().
 rename(
     columns={
         'SNOMED_concept_id_4' : 'SNOMED_concept_id',
         'SNOMED_concept_code_4' : 'SNOMED_concept_code',
         'SNOMED_concept_name_4' : 'SNOMED_concept_name',
         'SNOMED_concept_class_id_4' : 'SNOMED_concept_class_id'
     }
 ).
 set_index('MedDRA_concept_id_1').
 join(standard_reactions.
      drop_duplicates().
      set_index('MedDRA_concept_id')
     ).
 rename_axis('MedDRA_concept_id').
 reset_index().
 dropna(subset=['MedDRA_concept_id','SNOMED_concept_id','safetyreportid'])
)
standard_reactions_meddrasoc_to_snomed = (standard_reactions_meddrasoc_to_snomed.
 reindex(np.sort(standard_reactions_meddrasoc_to_snomed.columns),
         axis=1))
print(standard_reactions_meddrasoc_to_snomed.shape)
print(standard_reactions_meddrasoc_to_snomed.head())


# In[28]:


print(
    len(
        np.intersect1d(
            all_reports,
            standard_reactions_meddrasoc_to_snomed.safetyreportid.astype(str).unique()
        )
    )/len(all_reports)
)


# In[29]:


del m_to_s_r
del df
del joinedpt
del joinedhlt
del joinedhlgt
del joinedsoc
del all_reports
del standard_reactions
del standard_reactions_meddrapt_to_snomed
del standard_reactions_meddrahlt_to_snomed
del standard_reactions_meddrahlgt_to_snomed
del standard_reactions_meddra_relationships

