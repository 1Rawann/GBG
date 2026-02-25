import streamlit as st
import pandas as pd
import json
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DB_URL = os.getenv("DB_URL")

st.set_page_config(page_title="Postgres SQL Chatbot", page_icon=":robot_face:")
st.title("Chatbot with DB")

# ------------------ 1. Init Database ------------------

@st.cache_resource
def get_db():
    return SQLDatabase.from_uri(DB_URL)

# ------------------ 2. Init LLM ------------------

@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0
    )

# ------------------ 3. Load Few-Shot Examples ------------------

@st.cache_resource
def load_fewshots():
    with open("fewshots.json", "r", encoding="utf-8") as f:
        examples = json.load(f)
    return examples[:5]  

def get_sql_chain(db):
    examples = load_fewshots()

    few_shot_text = ""
    for ex in examples:
        few_shot_text += f"""
Question: {ex['naturalQuestion']}
SQL:
{ex['sqlQuery']}

"""

    template = f"""You are a PostgreSQL expert.

Here is the database schema:
{{schema}}

Here are example question-SQL pairs:
{few_shot_text}

Task: Write a SQL query to answer the following question:
{{question}}

IMPORTANT NOTES:
1. The tables were created via Pandas. If columns or table names are mixed case, use double quotes.
2. Return ONLY the SQL query. No markdown, no explanation.
3. Use lower case for tables and column names when possible.
4. usa = us and us = usa if needed due to case sensitivity.
"""

    prompt = PromptTemplate.from_template(template)
    llm = get_llm()

    sql_chain = (
        RunnablePassthrough.assign(schema=lambda _: db.get_table_info())
        | prompt
        | llm
        | StrOutputParser()
    )

    return sql_chain


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


question = st.text_input("Ask your database a question:")

if st.button("Ask"):
    if question:
        db = get_db()

        # 1. Generate SQL
        with st.spinner("Generating SQL..."):
            sql_chain = get_sql_chain(db)
            raw_response = sql_chain.invoke({"question": question})
            sql_query = raw_response.replace("```sql", "").replace("```", "").strip()

        st.subheader("Generated SQL")
        st.code(sql_query, language="sql")

        # 2. Execute Query
        try:
            with st.spinner("Running Query..."):
                result = pd.read_sql(sql_query, db._engine)

            st.subheader("Query Result")
            st.dataframe(result)

            # 3. Generate Natural Language Answer
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