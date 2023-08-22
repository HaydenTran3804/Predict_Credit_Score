import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from plotly.subplots import make_subplots


st.set_page_config(page_title='MainBoard',
                   page_icon=':bar_chart:',layout='wide')
df = pd.read_csv('df_train.csv')
df = df.set_index('Index')
chosen_col = ['Outstanding_Debt','Interest_Rate','Num_Credit_Card'
    ,'Changed_Credit_Limit','Num_of_Delayed_Payment','Annual_Income']
occupation_df = pd.read_csv('occupation.csv')
df['Occupation'] = occupation_df['Occupation']

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)

st.header('Credit Score DashBoard')

value = st.multiselect(
        "Filtration:",
        options=['Good','Standard','Poor'],
        default=['Good','Standard','Poor']
    )
dff = df[df['Credit_Score'].isin(value)]

fig_box= make_subplots(rows=1, cols=6,subplot_titles=chosen_col)
c = 1
for _ in chosen_col:
    fig_box.add_trace(px.box(data_frame=dff, x='Credit_Score', y=_).data[0], row=1, col=c)
    c = c + 1

fig_box.update_layout(
        height=250,
        title_font_size=8
    )
st.plotly_chart(fig_box,use_container_width=True)
left_bottom,right_bottom = st.columns([1,2])
occupation = dff['Occupation'].value_counts()
with left_bottom:
    tree_map = px.treemap(data_frame=occupation, names=occupation.index,
                            parents=['Occupation table'] * len(occupation.index),
                            values=occupation[occupation.index])
    tree_map.update_layout(
        height=270,
        width=500,
    )
    left_bottom.plotly_chart(tree_map)

with right_bottom:
    left,right = st.columns(2)
    hist_1 = px.histogram(data_frame=dff, x='Occupation', color='Credit_Score', barmode='group')
    hist_2 = px.histogram(data_frame=dff, x='Credit_Score', color='Credit_Score', barmode='group')
    hist_1.update_layout(
        height=270,
    width=500,
    )
    hist_2.update_layout(
        height=270,
    width=500,
    )
    left.plotly_chart(hist_1)
    right.plotly_chart(hist_2)
            
        
        
            
        
        
        
            
    
    
