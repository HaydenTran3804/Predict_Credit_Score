import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sb
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
st.set_page_config(layout='wide')
st.header('Credit Score DashBoard')
df = pd.read_csv('C:\AProgramming\APython\df_train.csv')
df = df.set_index('Index')
occupation_df = pd.read_csv('C:\AProgramming\APython\occupation.csv')
df['Occupation'] = occupation_df['Occupation']
#######################################################################################################################
value = st.multiselect(
        "Filtration for Credit Score types:",
        options=['Good','Standard','Poor'],
        default=['Good','Standard','Poor']
    )
dff = df[df['Credit_Score'].isin(value)]
choice = st.selectbox("Select",
             options=["Important Features","Occupations","Correlation among numeric variables","Picture of the Tree"
                      ,"Confusion Matrix of Decision Tree"],
             )

if choice == "Important Features":
    feat = st.multiselect("Choose freature",
                    options=['Outstanding_Debt','Interest_Rate','Num_Credit_Card'
    ,'Changed_Credit_Limit','Num_of_Delayed_Payment','Annual_Income'],
                    default=["Outstanding_Debt",'Interest_Rate','Num_Credit_Card'],
                    max_selections=3,
                    )
    fig_box = make_subplots(rows=1, cols=3, subplot_titles=feat)
    c = 1
    for _ in feat:
        fig_box.add_trace(px.box(data_frame=dff, x='Credit_Score', y=_).data[0], row=1, col=c)
        c = c + 1

    fig_box.update_layout(
        height=500,
        width = 500,
    )
    st.plotly_chart(fig_box,use_container_width=True)

elif choice == "Occupations":
    left_bottom, right_bottom = st.columns([1, 1.5])
    with left_bottom:
        occupation = dff['Occupation'].value_counts()
        tree_map = px.treemap(data_frame=occupation, names=occupation.index,
                              parents=['Occupation table'] * len(occupation.index),
                              values=occupation[occupation.index])
        tree_map.update_layout(
            title_text='Number of Considered Occupation',
            height=500,
            width=500,
        )
        left_bottom.plotly_chart(tree_map,use_container_width=True)
    with right_bottom:
        hist_1 = px.histogram(data_frame=dff, x='Occupation', color='Credit_Score', barmode='group')
        hist_1.update_layout(
            title_text='Number of Credit Score types that Each Considered Occupation has',
            height=500,
            width=400,
        )
        right_bottom.plotly_chart(hist_1, use_container_width=True)
elif choice == "Correlation among numeric variables":
    nummatrix = dff.select_dtypes(include='number')
    corr = nummatrix.corr()
    fig_heatmap, ax = plt.subplots(figsize=(10,3))
    sb.heatmap(corr, vmax=1, vmin=-1, cmap="YlGnBu", ax=ax,annot=True, fmt='.2f',annot_kws={"size":4},cbar_kws={ 'ticks' : [2, 2] })
    ax.xaxis.set_tick_params(labelsize=5,rotation = 80)
    ax.yaxis.set_tick_params(labelsize = 5)
    st.pyplot(fig_heatmap)
elif choice == "Picture of the Tree":
    x = dff.select_dtypes(include='number')
    y = dff['Credit_Score'].replace({'Good': 2, 'Standard': 1, 'Poor': 0})
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    DT_model = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=5, random_state=100, max_depth=3)
    DT_model.fit(x_train, y_train)
    y_pred_DT = DT_model.predict(x_test)
    xc = x.columns
    y_tree = y
    y_tree.replace({2: 'Good', 1: 'Standard', 0: 'Poor'}, inplace=True)
    dot_data = tree.export_graphviz(DT_model, feature_names=xc, class_names=y, filled=True, impurity=True)
    st.graphviz_chart(dot_data, use_container_width=True)
elif choice == "Confusion Matrix of Decision Tree":
    x = dff.select_dtypes(include='number')
    y = dff['Credit_Score'].replace({'Good': 2, 'Standard': 1, 'Poor': 0})
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    DT_model = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=5, random_state=100, max_depth=3)
    DT_model.fit(x_train, y_train)
    y_pred_DT = DT_model.predict(x_test)
    fig = confusion_matrix(y_test, y_pred_DT)
    fig = px.imshow(fig,
                    text_auto=True,
                    labels=dict(x="Predicted", y="Truth")
                    )
    fig.update_layout(
        title='Confusion Matrix of Decision Tree',
        width=900,
        height=800,
    )
    st.plotly_chart(fig, use_container_width=True)
