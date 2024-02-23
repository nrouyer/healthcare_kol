import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import os

st.set_page_config(
        page_title="Healthcare KOL",
)

#from streamlit_extras.app_logo import add_logo
#add_logo("images/icon_accidents.png", height=10)



from st_pages import show_pages_from_config

show_pages_from_config()

"""
# Find Key Opinion Leader
"""

#st.image('images/chatgpt_accident.png', caption='Accidents', width=500)

# openai_api_key = st.secrets["OPENAI_KEY"]

from langchain_community.llms import OpenAI
from langchain_community.graphs import Neo4jGraph
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.embeddings.openai import OpenAIEmbeddings

os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_KEY"]
url = st.secrets["AAA_URI"]
username = st.secrets["AAA_USERNAME"]
password = st.secrets["AAA_PASSWORD"]
graph = Neo4jGraph(
    url=url,
    username=username,
    password=password
)


llm = OpenAI()

vectorstore = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(),
    url=url,
    username=username,
    password=password,
    index_name='ct_vector_index',
    node_label="ClinicalTrial",
    text_node_properties=['study_title', 'summary'],
    embedding_node_property='ctembedding',
)

vector_qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(), chain_type="stuff", retriever=vectorstore.as_retriever())

contextualize_query = """
match (node)<-[:PARTICIPATES_IN]->(hcp:HCP)
WITH node AS ct, hcp, score, {} as metadata limit 5
RETURN "ClinicalTrial : "+ ct.study_title + " enriched context of HCP working on clinical trials : " + coalesce(apoc.text.join(collect(custom.hcp.pubctContext(hcp)),"\n"), "") +"\n" as text, score, metadata
"""

contextualized_vectorstore = Neo4jVector.from_existing_index(
    OpenAIEmbeddings(),
    url=url,
    username=username,
    password=password,
    index_name="ct_vector_index",
    retrieval_query=contextualize_query,
)

vector_plus_context_qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(), chain_type="stuff", retriever=contextualized_vectorstore.as_retriever())

# Streamlit layout with tabs
container = st.container()
question = container.text_input("**:blue[Question:]**", "")

if question:
    tab1, tab2, tab3 = st.tabs(["No-RAG", "Basic RAG", "Augmented RAG"])
    with tab1:
        st.markdown("**:blue[No-RAG.] LLM only. AI responds to question; can lcause hallucinations:**")
        st.write(llm(question))
    with tab2:
        st.markdown("**:blue[Basic RAG.] Simple Vector Search:**")
        st.write(vector_qa.run(question))
    with tab3:
        st.markdown("**:blue[Augmented RAG.] Vector Search on clinical trials plus HCP context:**")
        st.write(vector_plus_context_qa.run(question))


