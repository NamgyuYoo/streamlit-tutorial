import streamlit as st
import pandas as pd
import numpy as np

# 타이틀 적용 예시
st.title('CP S/W LLM Benchmark')

# 특수 이모티콘 삽입 예시
# emoji: https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/
st.title('Leaderoard :sports_medal:')

filepath =  'D:\LLMWorkSpace\streamlit-tutorial\leaderboard.csv'
df =  pd.read_csv(filepath, encoding='cp949')

benchmarklist = df.columns[3:]
modellist = df['Model Name']

print('benchmarklist:',benchmarklist)
print('modellist:',modellist)

addfilters = st.checkbox('Add filters')

if addfilters:
    filterbymodel = st.multiselect(
    'Filter by model:', ['Green', 'Yellow', 'Red', 'Blue'], default=None)

    st.write('You selected:', filterbymodel)
    
    filterbybenchmark = st.multiselect(
    'Filter by benchmark:', ['Green', 'Yellow', 'Red', 'Blue'], default=None)

    st.write('You selected:', filterbybenchmark)
    
    filterbyresults = st.multiselect(
    'Filter by results:', ['Green', 'Yellow', 'Red', 'Blue'], default=None)

    st.write('You selected:', filterbyresults)
    
   
    
clearemptyresults = st.checkbox('Clear empty entries')

if clearemptyresults:
    st.write('clearemptyresults selected!')



st.dataframe(df)  # Same as st.write(df)

