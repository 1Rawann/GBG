import os
import shutil
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


DB_PATH = "./my_arabic_db"
INPUT_FILE = "arabic.txt"
OUTPUT_FILE = "output_answer.txt"

if os.path.exists(DB_PATH):
    shutil.rmtree(DB_PATH)

def run_arabic_rag_free(file_path, query):

    print("--- 1. Loading Document ---")
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found!")
        return

    loader = TextLoader(file_path, encoding="utf-8")
    raw_documents = loader.load()

    print("--- 2. Creating Chunks ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "؟", "،", " ", ""]
    )
    
    chunks = text_splitter.split_documents(raw_documents)
    

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"Chunk_{i+1}"
    
    print(f"Total Chunks Created: {len(chunks)}")


    print("--- 3. Initializing Local Embeddings ---")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


    print("--- 4. Building Vector Store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings,
        persist_directory=DB_PATH
    )


    print("--- 5. Connecting to Local Llama 3.2 ---")
    llm = OllamaLLM(model="llama3.2") 
    
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question in Arabic only. "
        "If the answer is not in the context, say you don't know. "
        "\n\nContext:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])


    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(vectorstore.as_retriever(), question_answer_chain)


    print("--- 6. Generating Answer (Please wait...) ---")
    response = rag_chain.invoke({"input": query})


    print("\n" + "="*30)
    print("PROCESS COMPLETE")
    print("="*30)


    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"QUESTION: {query}\n")
        f.write("-" * 20 + "\n")
        f.write(f"ANSWER:\n{response['answer']}\n")
        f.write("-" * 20 + "\n")
        f.write("SOURCES USED:\n")
        for doc in response["context"]:
            chunk_name = doc.metadata.get('chunk_id', 'Unknown')
            f.write(f"[{chunk_name}]: {doc.page_content}\n\n")

    print(f"\n[SUCCESS] Arabic output is saved in: {OUTPUT_FILE}")
    print("Please open that file with Notepad to read the answer clearly.")

if __name__ == "__main__":
    query_text = "ماذا حدث عند دخول المنطقة تحت مظلة الحكم العثماني؟"
    run_arabic_rag_free(INPUT_FILE, query_text)