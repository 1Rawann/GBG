import streamlit as st
import google.generativeai as genai
import pandas as pd
from sqlalchemy import create_engine, text

import os
from dotenv import load_dotenv
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


DB_URL=os.getenv("DB_URL")

model=genai.GenerativeModel("gemini-2.5-flash")

st.set_page_config(page_title="Postgress SQL Chatbot", page_icon=":robot_face:")
st.title("Chatbot with DB")

@st.cache_resource
def get_engine():
    engine = create_engine(DB_URL)
    return engine

@st.cache_resource
def get_schema():
    engine = get_engine()
    inspector_query = text("""
        Select table_name, column_name from information_schema.columns
        where table_schema = 'public' 
        order by table_name, ordinal_position;  
                                           
                           
        """)
    schema_str = ""

    try:

        with engine.connect() as conn:
            result = conn.execute(inspector_query)
            current_table=""
            for row in result:
                table_name = row[0]
                column_name = row[1]
                if table_name != current_table:
                    schema_str += f"\nTable: {table_name}\n Columns: "
                    current_table = table_name
                schema_str += f" - {column_name},"

    except Exception as e:
        st.error(f"Error fetching schema: {e}")
        return ""            

    return schema_str

def generate_sql_from_gemeni(question, schema):
    promt=f"""you are an expert postgreSQL developer
Here is the database schema:
{schema}
your task :
1- write a SQL query that answers the following question:
{question}
2- important notes:the tables were created using via pandas  :
if columns or tables name are mixedcase use double qouts around them in the query.
3- only write the SQL query without any explanation or text.
"""
    response = model.generate_content(promt)
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
    promt=f"""you are an expert postgreSQL developer 
Here is the question:{question}
Here is the SQL query that was generated: {sql}
Here is the data returned by the query: {data}

Task: write a concise and clear answer to the question in natural language based on the SQL query and the data returned.
if the data is empty or contains an error, explain that in your answer.



             """   
    response = model.generate_content(promt)
    return response.text


# ------------------ UI ------------------

question = st.text_input("Ask your database a question:")

if st.button("Ask"):
    if question:
        schema = get_schema()

        sql_query = generate_sql_from_gemeni(question, schema)
        st.subheader("Generated SQL")
        st.code(sql_query, language="sql")

        result = run_query(sql_query)

        if isinstance(result, pd.DataFrame):
            st.subheader("Query Result")
            st.dataframe(result)

            answer = get_natural_response(
                question,
                sql_query,
                result.to_markdown()
            )

            st.subheader("Final Answer")
            st.write(answer)
        else:
            st.error(result)
