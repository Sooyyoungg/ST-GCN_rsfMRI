##
import pandas as pd
import numpy as np
import os
import re
##

# data path
targ_folder = [i for i in os.listdir('/scratch/bigdata/ABCD/abcd-fmriprep-rs/abcd-fmriprep-rs-time') if re.match('fmriprep-deri-',i)]
print(len(targ_folder)) # 9486

##
df = pd.read_csv('./demo.total.csv') # shape: (9658, 18)
data = df[['subjectkey','sex']]

data['subjectkey'] = data['subjectkey'].apply(lambda x : 'fmriprep-deri-'+x.replace('_',''))
data['sex'] = data['sex']-1 # 1/2 -> 0/1

data = data[data['subjectkey'].isin(targ_folder)] # 9486개에서 8299개 남음
print(data.shape)

data.to_csv('demo.txt',index=False,header=None, sep='\t')


