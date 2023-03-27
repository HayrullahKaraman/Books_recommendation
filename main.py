import streamlit as st
import pandas as pd
import numpy as np
import pickle
from joblib import load
import matplotlib.pyplot as plt

from Book_Recommandation import get_recommendations
from Book_Recommandation import find_similar_films

# Page Setting
st.set_page_config(
    page_title="Next Book ? ",
    page_icon="https://images.theconversation.com/files/45159/original/rptgtpxd-1396254731.jpg?ixlib=rb-1.1.0&q=45&auto=format&w=754&h=502&fit=crop&dpr=1",
    menu_items={
        "Get help": "mailto:hayrullahkaraman@gmail.com",
        "About": "For More Information\n" + "https://github.com/HayrullahKaraman"
    }
)

#Info
url="https://www.nvidia.com/en-us/glossary/data-science/recommendation-system/#:~:text=A%20recommendation%20system%20is%20an,demographic%20information%2C%20and%20other%20factors."
st.title("Book Recommendation System")
st.header("")
st.markdown("In my project, I will talk about the kiap recommendation system"
"I will use the Book-Crossing Dataset published by Cai-Nicolas Ziegler in August 2004 to use in the project. Approximately 278k users, 1.149.000 votes, 271 books were evaluated.")
st.image("https://static01.nyt.com/images/2015/10/24/opinion/24manguel/24manguel-superJumbo.jpg?quality=75&auto=webp",)
st.markdown("About Technology")
st.markdown("In our daily life, we come across recommendation systems when recommending friends on an e-commerce site, "
            "online movie, online game, or even on an Instagram, so lets briefly talk about what these recommendation systems are.")
st.markdown("About recommendation systems")
st.markdown("Basically, there are two types of recommendation systems, but now these two types of systems can be used as hybrids, one of them is content-based recommendation systems, suggesting another similar book according to the similarities of a book, suggesting another similar book to the person who bought the adventure book, and the second method is two users according to the user behavior. If he likes a book, it is for someone to recommend a different book he bought to another.")



top10=load("top10.pkl")
st.table(top10.bookTitle)

content=st.text_input('Input film name')

if st.button("Submit"):
   content_fun=get_recommendations(content)
   st.write(content_fun)
else:
   st.markdown("select one film file")

collabarity=st.text_input('Users like')
if st.button("Submit-user"):
   collabarity_fun=find_similar_films([collabarity],5)
   st.write(collabarity_fun)
else:
   st.markdown("select one film file")