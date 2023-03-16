import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from PIL import Image
df=pd.read_csv('Data/diabetes.csv')
df['Outcome']=df['Outcome'].astype(object)
df['insulin_ml']=np.where((df['Insulin']<=30),'lowml',
           np.where((df['Insulin']>30)&(df['Insulin']<=35),"normalml",
           "highml"))
df['bp']=np.where((df['BloodPressure']<=60),'lowbp',
           np.where((df['BloodPressure']>60)&(df['BloodPressure']<=80),"normalbp",
           "highbp"))
df['level_glucose']=np.where((df['Glucose']<=90),'low',
           np.where((df['Glucose']>90)&(df['Glucose']>110),"normal",
           "high"))
df['insulin_ml']=df['insulin_ml'].astype(object)
df['level_glucose']=df['level_glucose'].astype(object)
df['bp']=df['bp'].astype(object)
st.title('Diabetes analysis')

def home():
    st.title("Welcome to diabetes data presentation")
    image = Image.open('diabetes.png')
    st.image(image, caption='diabetes') 
def data():
    st.header('Header of Dataframe')
    st.write(df.head())
def per():
    
    options=st.selectbox('select the variable', ["Pregnancies", "Age", "DiabetesPedigreeFunction"])
    if options == 'Pregnancies':
     Preg_plot()
    elif options == 'Age':
     Age_plot()
    elif options == 'DiabetesPedigreeFunction':
     Dia_plot()
def med():
    tab1, tab2, tab3, tab4  = st.tabs(["Glucose&Bp", "Insulin", "BMI", "SkinThickness"])

    with tab1:
     st.header("Glucose&Bp")
     fig = px.bar(df_selection, x='level_glucose', color='bp', facet_col='Outcome')
     st.plotly_chart(fig)
    with tab2:
     st.header("Insulin")
     fig = px.bar(df_selection, x='insulin_ml', color='Outcome')
     st.plotly_chart(fig)
    with tab3:
     st.header("BMI")
     fig = px.box(df_selection, y='BMI', color='Outcome')
     st.plotly_chart(fig)
     bmi=df.groupby("Outcome")["BMI"].agg({"min","max","count","median","mean"})
     bmi.reset_index()
     st.dataframe(bmi)
    
           
     def convert_df(df):
        return df.to_csv().encode('utf-8')
     csv = convert_df(bmi)
     st.download_button(
     label="Download",
     data=csv,
     file_name='BMI.csv',
     mime='text/csv',
     )
    with tab4:
     st.header("SkinThickness")
     fig = px.pie(df_selection, values='SkinThickness', names='Outcome')
     st.plotly_chart(fig)
     skt=df.groupby("Outcome")["SkinThickness"].agg({"min","max","count","median","mean"})
     skt.reset_index()
     st.dataframe(skt)
    
           
     def convert_df(df):
        return df.to_csv().encode('utf-8')
     csv = convert_df(skt)
     st.download_button(
     label="Download",
     data=csv,
     file_name='SkinThickness.csv',
     mime='text/csv',
     )
def rep():
    st.title("REPORT")
    col1, col2 = st.columns(2)
    with col1:
     st.header("Personal")
     st.write("Pregnancies - High chances of affecting diabetes if number of pregnancies is high.")
     st.write("DiabetesPedigreeFunction - High chances of affecting diabetes based on family history.")
     st.write("Age - High chances of affecting diabetes in the middle age")
     
    with col2:
     st.header("Medical")
     st.write("Glucose - High chances of affecting diabetes if glucose level is high.")
     st.write("BMI - High chances of affecting diabetes.")
     st.write("Insulin - High chances of affecting diabetes if the insulin level is high.")
     st.write("SkinThickness - Less chances of affecting diabetes.")
    
def Preg_plot():
    st.header("Pregnancies")
    st.sidebar.header("Please Filter Here:")
    df_selection = st.sidebar.multiselect("Select the Outcome:",
                   options=df["Outcome"].unique(),
                   default=df["Outcome"].unique())
    mask = df['Outcome'].isin(df_selection)
    fig = px.histogram(df[mask], x='Pregnancies', color='Outcome')
    st.plotly_chart(fig)
    pre=df.groupby("Outcome")["Pregnancies"].agg({"min","max","count","median","mean"})
    pre.reset_index()
    st.dataframe(pre)
    
           
    def convert_df(df):
        return df.to_csv().encode('utf-8')
    csv = convert_df(pre)
    st.download_button(
    label="Download",
    data=csv,
    file_name='Pregnancies.csv',
    mime='text/csv',
    )
    
    
       

def Age_plot():
 Outcome = df['Outcome'].unique().tolist()
 ages = df['Age'].unique().tolist()

 age_selection = st.slider('Age:',
                        min_value= min(ages),
                        max_value= max(ages),
                        value=(min(ages),max(ages)))

 df_selection = st.sidebar.multiselect('Outcome:',
                                    Outcome,
                                    default=Outcome)

# --- FILTER DATAFRAME BASED ON SELECTION
 mask = (df['Age'].between(*age_selection)) & (df['Outcome'].isin(df_selection))
 fig = px.scatter(df[mask], x='Age', y='Pregnancies', color='Outcome')
 st.plotly_chart(fig)
 

    
    
def Dia_plot():
    st.header("DiabetesPedigreeFunction")
    st.sidebar.header("Please Filter Here:")
    df_selection = st.sidebar.multiselect("Select the Outcome:",
                   options=df["Outcome"].unique(),
                   default=df["Outcome"].unique())
    mask = df['Outcome'].isin(df_selection)
    fig = px.box(df[mask], color='Outcome', y='DiabetesPedigreeFunction')
    st.plotly_chart(fig)
    dpf=df.groupby("Outcome")["DiabetesPedigreeFunction"].agg({"min","max","count","median","mean"})
    dpf.reset_index()
    st.dataframe(dpf)
    
           
    def convert_df(df):
        return df.to_csv().encode('utf-8')
    csv = convert_df(dpf)
    st.download_button(
    label="Download",
    data=csv,
    file_name='DiabetesPedigreeFunction.csv',
    mime='text/csv',
    )
    
st.sidebar.title('Navigation')
side = st.sidebar.radio('Select what you want to display:', ['Home', 'Data Header', 'Personal Info', "Medical Info", "Report"])

if side == 'Home':
 home()
elif side == 'Data Header':
 data()
elif side == 'Personal Info':
 
 per()
elif side == 'Medical Info':
    
 st.sidebar.header("Please Filter Here:")
 Outcome_level = st.sidebar.multiselect("Select the Outcome:",
 options=df["Outcome"].unique(),
 default=df["Outcome"].unique())
 df_selection = df.query(
    "Outcome == @Outcome_level"
)    
 med()
elif side == 'Report':
 rep()
