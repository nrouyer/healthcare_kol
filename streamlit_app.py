import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import st_pages as st_pages
import os
import langchain_openai as langchain_openai 

from st_pages import show_pages_from_config

st.set_page_config(
        page_title="Healthcare KOL",
)

#from streamlit_extras.app_logo import add_logo
#add_logo("images/icon_accidents.png", height=10)

show_pages_from_config()

"""
# Find Key Opinion Leader
"""

st.image('images/healthcare_kol.jpg', caption='KOL', width=400)

# openai_api_key = st.secrets["OPENAI_KEY"]

#from langchain_community.llms import OpenAI
from langchain_openai import OpenAI
from langchain_community.graphs import Neo4jGraph
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain.chains import GraphCypherQAChain

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

# Vector Search
vectorstore = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(),
    url=url,
    username=username,
    password=password,
    index_name="publications",
    node_label="Publication",
    text_node_properties=["abstract", "title"],
    embedding_node_property="pubEmbedding",
)

# from existing index
#vectorstore = Neo4jVector.from_existing_index(
#    OpenAIEmbeddings(),
#    url=url,
#    username=username,
#    password=password,
#    index_name="publications",
#)

vector_qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(), chain_type="stuff", retriever=vectorstore.as_retriever())

# Vector + context
contextualize_query = """
match (node)<-[:PARTICIPATES_IN]-(hcp:HCP)
WITH node AS pub, hcp, score, {} as metadata limit 1
WITH pub, score, metadata, hcp, custom.hcp.pubctContext(hcp) AS hcpContext
WITH pub, score, metadata, collect(hcpContext) AS hcpContexts
RETURN "Publication : "+ pub.title + " enriched context of HCP working on publication : " + coalesce(apoc.text.join(hcpContexts,"\n"), "") +"\n" as text, score, metadata
"""
contextualize_query_2 = """
match (node)<-[:PARTICIPATES_IN]-(hcp:HCP)
WITH node AS pub, hcp, score, {} as metadata limit 1
OPTIONAL MATCH (hcp)-[rPub:PARTICIPATES_IN]-(p:Publication)
WITH pub, score, metadata, hcp, type(rPub) + " " + labels(p)[0] as type, collect(p.title) as pubTitles
WITH pub, score, metadata, hcp, type+": "+reduce(s="", n IN pubTitles | s + n + ", ") as types
WITH pub, score, metadata, hcp, collect(types) as contextsPub
OPTIONAL MATCH (hcp)-[rCt:PARTICIPATES_IN]-(ct:ClinicalTrial)
WITH pub, score, metadata, hcp, contextsPub, type(rCt) + " " + labels(ct)[0] as type, collect(ct.study_title) as ctTitles
WITH pub, score, metadata, hcp, contextsPub, type+": "+reduce(s="", n IN ctTitles | s + n + ", ") as types
WITH pub, score, metadata, hcp, contextsPub, collect(types) as contextsCt
RETURN "HCP name: "+ hcp.full_name + "\n" +
reduce(s="", c in contextsPub | s + substring(c, 0, size(c)-2) +"\n") + "\n" +
reduce(s="", c in contextsCt | s + substring(c, 0, size(c)-2) +"\n") as context
"""

#contextualized_vectorstore = Neo4jVector.from_existing_index(
#    OpenAIEmbeddings(),
#    url=url,
#    username=username,
#    password=password,
#    index_name="publications",
#    retrieval_query=contextualize_query,
#)

contextualized_vectorstore = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(),
    url=url,
    username=username,
    password=password,
    index_name="publications",
    node_label="Publication",
    text_node_properties=["abstract", "title"],
    embedding_node_property="pubEmbedding",
    retrieval_query=contextualize_query_2,
)

vector_plus_context_qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(), chain_type="stuff", retriever=contextualized_vectorstore.as_retriever())

# cypher dependency context
graph.refresh_schema()

print(graph.schema)

cypher_dependency_context_qa = GraphCypherQAChain.from_llm(
    cypher_llm = ChatOpenAI(temperature=0, model_name='gpt-4'),
    qa_llm = ChatOpenAI(temperature=0), graph=graph, verbose=True,
)

# Streamlit layout with tabs
container = st.container()
question = container.text_input("**:blue[Question:]**", "")

if question:
    tab1, tab2, tab3, tab4 = st.tabs(["No-RAG", "Basic RAG", "Augmented RAG", "Dependency Questions"])
    with tab1:
        st.markdown("**:blue[No-RAG.] LLM only. AI responds to question; can cause hallucinations:**")
        st.write(llm(question))
    with tab2:
        st.markdown("**:blue[Basic RAG.] Simple Vector Search:**")
        st.write(vector_qa.run(question))
    with tab3:
        st.markdown("**:blue[Augmented RAG.] Vector Search on publications plus HCP context:**")
        st.write(vector_plus_context_qa.run(question))
    with tab4:
        st.markdown("**:blue[Dependency Questions.] LLM maps the underlying graph model:**")
        st.write(cypher_dependency_context_qa.run(question))

