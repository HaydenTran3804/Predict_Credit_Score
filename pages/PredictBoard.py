import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
st.set_page_config(layout='centered')
df = pd.read_csv('df_train.csv')
df = df.set_index('Index')
chosen_col = ['Outstanding_Debt','Interest_Rate','Num_Credit_Card'
    ,'Changed_Credit_Limit','Num_of_Delayed_Payment','Annual_Income']
x = df[chosen_col]
y = df['Credit_Score']
summary = df[chosen_col].mean()
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 100)
model = DecisionTreeClassifier(criterion = 'entropy',min_samples_leaf = 5,random_state = 100,max_depth = 3)
model.fit(x_train,y_train)
occupation_df = pd.read_csv('occupation.csv')
df['Occupation'] = occupation_df['Occupation']

descript = f'''
I built this DashBoard to predict the Credit Score by typing your customers' information here. I chose these features because \
they had the significant effects on Credit Score
'''
st.header(descript)
st.title('Predict Credit Score')
values = []
for _ in chosen_col:
    value = st.number_input(f'Input for {_}: from {round(min(df[_]))} to {round(max(df[_]))}',
                            max_value=round(max(df[_])),min_value=round(min(df[_])))
    values.append(value)

values = np.array(values)
value_button = st.button('Predict',use_container_width=True)
col1,col2 = st.columns(2)
with col1:
    st.header('Result: ')
if value_button:
    if None not in values:
        with col2:
            st.header(model.predict(values.reshape((1,-1)))[0])
    else:
        st.header('Not enough data to predict')
else:
    with col2:
        st.header(' ')
        
        
    
