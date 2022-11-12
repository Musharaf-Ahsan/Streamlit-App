import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression


#st.title("Easy Salary Prediction App")

html_temp = """
    <div style="background-color:black;padding:10px">
    <h2 style="color:red;text-align:center;">Salary Prediction App</h2>
    <p style="color:red;text-align:center;" >This is a <b>Streamlit</b> app to predict <b>Salary</b>.</p>
    </div>
    """
st.markdown(html_temp,unsafe_allow_html=True)

#st.write("""This is a **Streamlit** app use for Salary prediction)


def user_input_features():
    age = st.sidebar.slider('age', 31.1, 40.5)
    distance = st.sidebar.slider('distance', 77.75, 101.25)
    YearsExperience = st.sidebar.slider('YearsExperience', 1.1, 10.50)
    
    data = {'age': age, 
            'distance': distance,
            'YearsExperience': YearsExperience}
    features = pd.DataFrame(data, index=[0])
    return features

st.sidebar.subheader('Enter Input Through Slider')
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)

dataset = pd.read_csv('ml_data_salary.csv')
X = dataset[['age', 'distance', 'YearsExperience']]
y = dataset['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = RandomForestClassifier()
model.fit(X_train, y_train)


prediction = model.predict(df)


st.subheader('Final Prediction')
st.error(prediction[0])
html_temp1 = """
    <div style="background-color:#f63366">
    <p style="color:green;text-align:center;" >Made By: <b>Musharaf Ahsan</b> </p>
    </div>
    """
st.markdown(html_temp1,unsafe_allow_html=True)