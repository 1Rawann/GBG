import streamlit as st
import google.generativeai as genai
import pandas as pd
from sqlalchemy import create_engine, text
import chromadb
import chromadb.utils.embedding_functions as embedding_functions

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DB_URL = os.getenv("DB_URL")

# Initialize Gemini Model
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

st.set_page_config(page_title="Postgres SQL Chatbot (RAG)", page_icon=":robot_face:")
st.title("Chatbot with DB (Powered by RAG)")

@st.cache_resource
def get_engine():
    engine = create_engine(DB_URL)
    return engine

@st.cache_resource
def init_vector_db():
    engine = get_engine()
    
    # 1. Fetch schema
    inspector_query = text("""
        SELECT table_name, column_name 
        FROM information_schema.columns
        WHERE table_schema = 'public' 
        ORDER BY table_name, ordinal_position;
    """)
    
    tables_dict = {}
    try:
        with engine.connect() as conn:
            result = conn.execute(inspector_query)
            for row in result:
                t_name = row[0]
                c_name = row[1]
                if t_name not in tables_dict:
                    tables_dict[t_name] = []
                tables_dict[t_name].append(c_name)
    except Exception as e:
        st.error(f"Error fetching schema: {e}")
        return None

    # 2. Setup ChromaDB 
    chroma_client = chromadb.Client()
   
    google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    api_key=GOOGLE_API_KEY, 
    model_name="models/gemini-embedding-001"
)
    
    # --- FIXED SECTION ---
    # We check if the collection exists by trying to get it; if not, we create it.
    # To ensure it's fresh, we delete it only if it actually exists.
    collection_name = "db_schema"
    try:
        chroma_client.delete_collection(name=collection_name)
    except Exception:
        # If it doesn't exist, we don't care, just move on
        pass
        
    collection = chroma_client.create_collection(
        name=collection_name, 
        embedding_function=google_ef
    )
    # ---------------------
    
    docs = []
    metadatas = []
    ids = []
    
    for table_name, columns in tables_dict.items():
        doc_content = f"Table: {table_name}\nColumns: {', '.join(columns)}"
        docs.append(doc_content)
        metadatas.append({"table_name": table_name})
        ids.append(table_name)
        
    if docs:
        collection.add(
            documents=docs,
            metadatas=metadatas,
            ids=ids
        )
        
    return collection

def retrieve_relevant_schema(question, collection, n_results=3):
    """
    Queries the vector database to find the top K most relevant tables
    based on the user's question.
    """
    if not collection:
        return ""
        
    results = collection.query(
        query_texts=[question],
        n_results=n_results # Adjust this depending on how many tables might be needed for joins
    )
    
    # Combine the retrieved table documents into a single schema string
    retrieved_docs = results['documents'][0]
    schema_str = "\n\n".join(retrieved_docs)
    return schema_str

def generate_sql_from_gemini(question, schema):
    prompt = f"""You are an expert PostgreSQL developer.
Here are the relevant database tables for the query:
{schema}

Your task:
1- Write a SQL query that answers the following question:
{question}
2- Important notes: The tables were created using pandas. 
If columns or tables names are mixed case, use double quotes around them in the query.
3- Only write the SQL query without any explanation or text.
"""
    response = model.generate_content(prompt)
    clean_sql = response.text.replace("```sql", "").replace("```", "").strip()
    return clean_sql

def run_query(query):
    engine = get_engine()
    with engine.connect() as conn:
        try:
            result = conn.execute(text(query))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            return df
        except Exception as e:
            return str(e)
        
def get_natural_response(question, sql, data):
    prompt = f"""You are an expert PostgreSQL developer. 
Here is the question: {question}
Here is the SQL query that was generated: {sql}
Here is the data returned by the query: {data}

Task: Write a concise and clear answer to the question in natural language based on the SQL query and the data returned.
If the data is empty or contains an error, explain that in your answer.
"""   
    response = model.generate_content(prompt)
    return response.text

# ------------------ UI ------------------

# Initialize the vector DB on startup
schema_collection = init_vector_db()

question = st.text_input("Ask your database a question:")

if st.button("Ask"):
    if question:
        if schema_collection is None:
            st.error("Vector database failed to initialize.")
        else:
            # 1. RAG Retrieval Step
            relevant_schema = retrieve_relevant_schema(question, schema_collection, n_results=3)
            
            st.subheader("Retrieved Schema context (RAG)")
            st.text(relevant_schema) # Showing the user what the LLM sees

            # 2. SQL Generation Step
            sql_query = generate_sql_from_gemini(question, relevant_schema)
            st.subheader("Generated SQL")
            st.code(sql_query, language="sql")

            # 3. Execution Step
            result = run_query(sql_query)

            if isinstance(result, pd.DataFrame):
                st.subheader("Query Result")
                st.dataframe(result)

                # 4. Natural Language Synthesis Step
                answer = get_natural_response(
                    question,
                    sql_query,
                    result.to_markdown()
                )

                st.subheader("Final Answer")
                st.write(answer)
            else:
                st.error(f"SQL Execution Error: {result}")