import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from plotly.subplots import make_subplots
import matplotlib as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import statsmodels.api as sm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import preprocessing, metrics
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
nummatrix = dff.select_dtypes(include ='number')
fig_box= make_subplots(rows=1, cols=6,subplot_titles=chosen_col)

x = dff.select_dtypes(include='number')
y = dff['Credit_Score'].replace({'Good':2,'Standard':1,'Poor':0})

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 42)
DT_model = DecisionTreeClassifier(criterion = 'entropy',min_samples_leaf = 5,random_state = 100,max_depth = 3)
DT_model.fit(x_train,y_train)  
y_pred_DT = DT_model.predict(x_test)
c = 1
for _ in chosen_col:
    fig_box.add_trace(px.box(data_frame=dff, x='Credit_Score', y=_).data[0], row=1, col=c)
    c = c + 1

fig_box.update_layout(
        height=250,
        title_font_size=8
    )
st.plotly_chart(fig_box,use_container_width=True)
left_bottom,right_bottom = st.columns([1,1.5])
occupation = dff['Occupation'].value_counts()
with left_bottom:
    tree_map = px.treemap(data_frame=occupation, names=occupation.index,
                            parents=['Occupation table'] * len(occupation.index),
                            values=occupation[occupation.index])
    tree_map.update_layout(
        title_text='Number of Considered Occupation',
        height=270,
        width=500,
    )
    left_bottom.plotly_chart(tree_map)

with right_bottom:
    hist_1 = px.histogram(data_frame=dff, x='Occupation', color='Credit_Score', barmode='group')
    hist_1.update_layout(
        title_text='Number of Each Considered Occupation',
        height=270,
        width=400,
    )
    right_bottom.plotly_chart(hist_1,use_container_width=True)
    



corr = nummatrix.corr()
fig_heatmap = px.imshow(corr)
fig_heatmap.update_layout(
title_text='Correlation among numeric variables',
height=600,
)
st.plotly_chart(fig_heatmap,use_container_width=True)
  
    
    

xc = x.columns
y_tree = y
y_tree.replace({2:'Good',1:'Standard',0:'Poor'},inplace=True)
dot_data = tree.export_graphviz(DT_model,feature_names = xc, class_names = y,filled = True,impurity = True)
st.markdown('<strong>Picture of the Tree</strong>',unsafe_allow_html=True)
st.graphviz_chart(dot_data,use_container_width=True)

fig = confusion_matrix(y_test,y_pred_DT)
fig = px.imshow(fig,
    text_auto = True,
               labels=dict(x="Predicted",y="Truth")
                )
fig.update_layout(
    title='Confusion Matrix of Decision Tree',
    width=900,
    height=800,
)
st.plotly_chart(fig,use_container_width=True)
        
            
        
        
        
            
    
    
