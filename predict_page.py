#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import pickle
import numpy as np

# In[4]:


def load_model():
    with open(r"C:\Users\Sahira\OneDrive - Universiti Malaya\WID3006\ML Cat Depression Indicator\saved_steps.pkl", 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
le_obesity = data["le_obesity"]
le_appetite = data["le_appetite"]
le_meow = data["le_meow"]
le_grooming = data["le_grooming"]
le_energy = data["le_energy"]

# In[5]:


def show_predict_page():
    st.title("Cat Depression Indicator")
    
    st.write("""### We need some information to predict how likely it is for your cat to be depressed""")

    name = st.text_input('Meow-Meow name')

    age = st.text_input('Meow-Meow age')

    breed = st.text_input('Meow-Meow breed')
    
    rating = st.slider("Rate how cute is your meow-meow from 1-10", 0, 10) 

    x1 = st.slider("Is your cat obese?", 0, 1)
    x2 = st.slider("Is your cat experiencing a loss of appetite?", 0, 1)
    x3 = st.slider("Is your cat meowing more than usual?", 0, 1)
    x4 = st.slider("Does your cat show poor grooming behaviour?", 0, 1)
    x5 = st.slider("Does your cat seem to lack energy?", 0, 1)

    ok = st.button("Check now")
    if ok:
        x = np.array([[x1, x2, x3, x4, x5]])
        x[:, 0] = le_obesity.transform(x[:, 0])
        x[:, 1] = le_appetite.transform(x[:, 1])
        x[:, 2] = le_meow.transform(x[:, 2])
        x[:, 3] = le_grooming.transform(x[:, 3])
        x[:, 4] = le_energy.transform(x[:, 4])
        x = x.astype(int)
        
        depressed = regressor.predict(x)
        if depressed == 0:
            st.subheader(f"Your cat is not depressed")
        else:
            st.subheader(f"Your cat is depressed")


def set_bg_hack_url():
    '''
    A function to unpack an image from url and set as bg.
    Returns
    -------
    The background.
    '''
        
    st.markdown(
        f""" 
        <style>
        .stApp {{
        background: url("https://images5.alphacoders.com/109/1093289.jpg");
        background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
set_bg_hack_url()