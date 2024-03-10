import streamlit as st
import pandas as pd
import numpy as np

# 타이틀 적용 예시
st.title('CP S/W LLM Benchmark')

# 특수 이모티콘 삽입 예시
# emoji: https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/
st.title('Leaderoard :sports_medal:')

filepath =  'D:\LLMWorkSpace\streamlit-tutorial\leaderboard.csv'
df = pd.read_csv(filepath, encoding='cp949')


benchmarklist = df.columns[3:]
modellist = df['Model Name']
resultslist = df.columns[2:]


df = df.set_index(keys=modellist, inplace=False, drop=False)

#print('benchmarklist:',benchmarklist)
#print('modellist:',modellist)

addfilters = st.checkbox('Add filters')

if addfilters:
      #rows
    dispdf = df.copy()
    
    #dispdf = dispdf.set_index(keys=modellist, inplace=False, drop=False)
    
   
    filterbymodel = st.multiselect(
    'Filter by model:', modellist, default=None)
    if filterbymodel:
        dispdf =  dispdf.loc[filterbymodel,:]
    st.write('You selected:', filterbymodel)
    
    #columns
    filterbybenchmark = st.multiselect(
    'Filter by benchmark:', benchmarklist, default=None)
    if filterbybenchmark:
        dispdf =  dispdf.loc[:,filterbybenchmark]

    st.write('You selected:', filterbybenchmark)
    
    filterbyresults = st.multiselect(
    'Filter by results:', resultslist, default=None)

    st.write('You selected:', filterbyresults)
    
    print('filterbymodel:',filterbymodel)
    print('filterbybenchmark:',filterbybenchmark)
    print('filterbyresults:',filterbyresults)
    
     
    st.dataframe(dispdf) # Row
    
clearemptyresults = st.checkbox('Clear empty entries')

if clearemptyresults:
    st.write('clearemptyresults selected!')


st.dataframe(df)  


