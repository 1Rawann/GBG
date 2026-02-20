import streamlit as st
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from sqlalchemy import create_engine

import os
from dotenv import load_dotenv
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


DB_URL=os.getenv("DB_URL")

st.set_page_config(page_title="Postgres SQL Chatbot", page_icon=":robot_face:")
st.title("Chatbot with DB ")

# --- 1. Init Database ---
@st.cache_resource
def get_db():
    # LangChain's wrapper automatically handles connections
    return SQLDatabase.from_uri(DB_URL)

# --- 2. Init LLM ---
@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=GOOGLE_API_KEY,
        temperature=0
    )

# --- 3. SQL Generation Chain ---
def get_sql_chain(db):
    template = """You are a PostgreSQL expert.
    
    Here is the database schema:
    {schema}
    
    Task: Write a SQL query to answer the following question:
    {question}
    
    IMPORTANT NOTES:
    1. The tables were created via Pandas. If columns or table names are mixed case, use double quotes (e.g. "PassengerId").
    2. Return ONLY the SQL query. No markdown, no explanation.
    3. make sure not be case sensitive when writing the query, use lower case for tables and columns names like usa and us. 
    4. notice that usa=us and us=usa in the query because of the case sensitivity of the tables and columns names.
    """
    
    prompt = PromptTemplate.from_template(template)
    llm = get_llm()
    
    # This chain automatically:
    # 1. Takes the input question
    # 2. Fetches the schema from 'db'
    # 3. Passes both to the prompt
    # 4. Sends to LLM -> Cleans output
    sql_chain = (
        RunnablePassthrough.assign(schema=lambda _: db.get_table_info())
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return sql_chain

# --- 4. Natural Language Response Chain ---
def get_response_chain():
    template = """You are a PostgreSQL expert.
    
    Question: {question}
    SQL Query: {query}
    Data: {data}
    
    Task: Write a clear, natural language answer based on the data, the SQL query and the question.
    """
    
    prompt = PromptTemplate.from_template(template)
    llm = get_llm()
    
    return prompt | llm | StrOutputParser()

# ------------------ UI ------------------

question = st.text_input("Ask your database a question:")

if st.button("Ask"):
    if question:
        db = get_db()
        
        # 1. Generate SQL
        with st.spinner("Generating SQL..."):
            sql_chain = get_sql_chain(db)
            # We invoke the chain with just the question; the lambda function above handles the schema
            raw_response = sql_chain.invoke({"question": question})
            
            # Clean formatting just in case
            sql_query = raw_response.replace("```sql", "").replace("```", "").strip()
        
        st.subheader("Generated SQL")
        st.code(sql_query, language="sql")

        # 2. Run Query
        # We use the internal engine to run the query via Pandas for a nice dataframe
        try:
            with st.spinner("Running Query..."):
                result = pd.read_sql(sql_query, db._engine)
            
            st.subheader("Query Result")
            st.dataframe(result)

            # 3. Generate Answer
            with st.spinner("Analyzing Answer..."):
                response_chain = get_response_chain()
                answer = response_chain.invoke({
                    "question": question,
                    "query": sql_query,
                    "data": result.to_markdown()
                })

            st.subheader("Final Answer")
            st.write(answer)

        except Exception as e:
            st.error(f"Error executing query: {e}")