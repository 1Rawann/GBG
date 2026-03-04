import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# --- 1. FIXED IMPORTS FOR 2026 (LangChain Classic) ---
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 2. CONFIGURATION & STATE ---
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="AI Executive Recruiter", layout="wide", page_icon="💼")

# Initialize Session States
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# --- 3. SIDEBAR: DYNAMIC UPLOADER & QUOTA-FRIENDLY INDEXING ---
with st.sidebar:
    st.title("📂 Candidate Database")
    st.info("Upload CVs here to start the conversation.")
    
    uploaded_files = st.file_uploader(
        "Upload PDF Resumes", 
        type="pdf", 
        accept_multiple_files=True
    )
    
    if st.button("Build Talent Index") and uploaded_files:
        with st.spinner("Analyzing and Chunking Documents..."):
            all_docs = []
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    loader = PyPDFLoader(tmp_file.name)
                    data = loader.load()
                    # Keep track of which PDF each chunk came from
                    for d in data:
                        d.metadata["source"] = uploaded_file.name
                    all_docs.extend(data)

            # DOCUMENT-AWARE CHUNKING (Optimized for Rate Limits)
            # We use separators to keep paragraphs and sentences together.
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150,
                separators=["\n\n", "\n", ". ", " "]
            )
            splits = text_splitter.split_documents(all_docs)
            
            # Create Embeddings
            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
            
            # Build FAISS Index in RAM
            st.session_state.vectorstore = FAISS.from_documents(splits, embeddings)
            st.success(f"Indexed {len(uploaded_files)} candidates successfully!")

# --- 4. MAIN CHAT INTERFACE ---
st.title("🤖 Pro Recruiter AI")
st.markdown("Rank your top candidates based on hands-on experience with verifiable proof.")

# Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input Logic
if user_query := st.chat_input("Ex: Find the top 3 Senior React devs with AWS experience."):
    
    # Add User message to state
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Process AI Response
    if not st.session_state.vectorstore:
        with st.chat_message("assistant"):
            st.warning("Please upload CVs in the sidebar before searching.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Screening candidates for hands-on experience..."):
                
                # Setup Gemini 2.5 Flash
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
                
                # Setup Retriever (Get k=8 chunks for a broader view of candidates)
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 8})

                # THE MASTER PROMPT
                system_prompt = (
                    "You are a Senior Executive Technical Recruiter. Your goal is to identify the "
                    "TOP 3 strongest candidates based on real-world hands-on experience.\n\n"
                    "EVALUATION CRITERIA:\n"
                    "- Prioritize candidates who built, led, or implemented projects rather than just 'mentioning' tools.\n"
                    "- Look for specific metrics or technical achievements.\n\n"
                    "STRICT OUTPUT PROTOCOL:\n"
                    "1. Provide a ranked list of the Top 3 candidates (mention Filenames).\n"
                    "2. For EACH candidate, you MUST include a 'VERIFIABLE PROOF' section. "
                    "This section must quote a specific project or sentence from their CV that confirms "
                    "the hands-on experience you are using as a reason for their rank.\n"
                    "3. If the user's input is not a legitimate job-related query, say 'unrelevant job description'.\n"
                    "4. If no one fits the criteria, say 'Need more candidates.'\n\n"
                     "do not hallucinate any information for imaginary jobs titls or skills. if the job is not clear or real do not return any candidate and return 'unrelevant job description'"
                    "Context (CV Chunks):\n{context}"
                )

                qa_prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}"),
                ])

                # Build the Chain (v1.2.x standard)
                doc_chain = create_stuff_documents_chain(llm, qa_prompt)
                rag_chain = create_retrieval_chain(retriever, doc_chain)
                
                # Invoke
                response = rag_chain.invoke({"input": user_query})
                answer = response["answer"]
                
                # Display result
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})