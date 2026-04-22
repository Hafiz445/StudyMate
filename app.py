import streamlit as st
import fitz  # PyMuPDF
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_cohere import ChatCohere

# Load environment variables from .env file (for local runs)
load_dotenv()

# Helper function to get API key safely without crashing
def get_api_key():
    # 1. Try to get it from environment variables first (this handles local .env)
    api_key = os.getenv("COHERE_API_KEY")
    if api_key:
        return api_key
        
    # 2. Safely try Streamlit secrets if environment variable is missing
    try:
        if "COHERE_API_KEY" in st.secrets:
            return st.secrets["COHERE_API_KEY"]
    except Exception:
        # If the secrets file is missing or broken, catch the error so the app doesn't crash
        pass
        
    return None

# --- Custom Premium Light Theme with Hover Effects & Study Background ---
def apply_custom_css():
    st.markdown("""
        <style>
        /* 1. Study Background: Subtle Dot Grid (Notebook Style) */
        .stApp {
            background-color: #f8fafc;
            background-image: radial-gradient(#cbd5e1 1.5px, transparent 1.5px);
            background-size: 25px 25px;
            color: #0f172a;
        }
        
        /* Ambient Floating Glows in the background */
        .stApp::before {
            content: "";
            position: fixed;
            top: -20%; left: -10%;
            width: 50vw; height: 50vw;
            background: radial-gradient(circle, rgba(147,197,253,0.3) 0%, rgba(255,255,255,0) 70%);
            z-index: -1;
            pointer-events: none;
        }
        .stApp::after {
            content: "";
            position: fixed;
            bottom: -20%; right: -10%;
            width: 50vw; height: 50vw;
            background: radial-gradient(circle, rgba(196,181,253,0.25) 0%, rgba(255,255,255,0) 70%);
            z-index: -1;
            pointer-events: none;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #1e3a8a !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            font-weight: 800;
        }
        
        /* 2. Interactive Text Input (Mouse Hover & Focus Effects) */
        .stTextInput > div > div > input {
            background-color: rgba(255, 255, 255, 0.9);
            color: #0f172a;
            border: 2px solid #e2e8f0;
            border-radius: 16px;
            padding: 18px;
            font-size: 16px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.02);
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275); 
        }
        .stTextInput > div > div > input:hover {
            border-color: #60a5fa;
            transform: translateY(-5px) scale(1.01);
            box-shadow: 0 15px 25px rgba(59, 130, 246, 0.15);
        }
        .stTextInput > div > div > input:focus {
            border-color: #3b82f6;
            background-color: #ffffff;
            box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.2);
        }
        
        /* Sidebar styling (Glassmorphism) */
        [data-testid="stSidebar"] {
            background-color: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(10px);
            border-right: 1px solid rgba(226, 232, 240, 0.8);
        }
        
        /* 3. Action Buttons (Smooth scaling and glowing shadow) */
        .stButton>button {
            background: linear-gradient(135deg, #4f46e5 0%, #0ea5e9 100%);
            color: white;
            border: none;
            border-radius: 14px;
            padding: 12px 24px;
            font-weight: 700;
            letter-spacing: 0.5px;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
            width: 100%;
            box-shadow: 0 4px 10px rgba(79, 70, 229, 0.2);
        }
        .stButton>button:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 20px rgba(79, 70, 229, 0.4);
            color: white;
        }
        .stButton>button:active {
            transform: translateY(2px);
        }
        
        /* 4. File Dropzone (Breathing effect on hover) */
        [data-testid="stFileUploadDropzone"] {
            background-color: rgba(255, 255, 255, 0.7);
            border: 2px dashed #94a3b8;
            border-radius: 16px;
            transition: all 0.3s ease-in-out;
        }
        [data-testid="stFileUploadDropzone"]:hover {
            border-color: #4f46e5;
            background-color: rgba(238, 242, 255, 0.9);
            transform: scale(1.03);
            box-shadow: 0 10px 25px rgba(79, 70, 229, 0.15);
        }
        
        /* 5. Answer Alert Box (Pop-out effect) */
        .stAlert {
            background-color: rgba(255, 255, 255, 0.95);
            border: none;
            border-left: 6px solid #4f46e5;
            color: #1e293b;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            transition: all 0.3s ease;
        }
        .stAlert:hover {
            transform: translateX(8px) translateY(-2px);
            box-shadow: -5px 12px 20px -5px rgba(79, 70, 229, 0.15);
        }
        
        /* 6. Expanders (Lift and glow) */
        .streamlit-expanderHeader {
            background-color: rgba(255, 255, 255, 0.9);
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            transition: all 0.3s ease;
        }
        .streamlit-expanderHeader:hover {
            background-color: #ffffff;
            border-color: #60a5fa;
            transform: translateY(-2px);
            box-shadow: 0 8px 15px rgba(59, 130, 246, 0.1);
            color: #2563eb !important;
        }
        
        /* 7. Custom Frosted Glass Footer */
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(15px);
            -webkit-backdrop-filter: blur(15px);
            color: #64748b;
            text-align: center;
            padding: 15px;
            border-top: 1px solid rgba(226, 232, 240, 0.8);
            font-size: 15px;
            z-index: 100;
            box-shadow: 0 -4px 15px rgba(0, 0, 0, 0.02);
            transition: all 0.3s ease;
        }
        .footer:hover {
            background-color: rgba(255, 255, 255, 0.9);
            padding-bottom: 22px; 
            color: #475569;
        }
        .footer span {
            background: linear-gradient(135deg, #4f46e5 0%, #0ea5e9 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 900;
            font-size: 16px;
        }
        
        /* Add padding to bottom so footer doesn't hide content */
        .main .block-container {
            padding-bottom: 100px;
        }
        </style>
        
        <div class="footer">
            StudyMate v1.0 • Engineered with ❤️ by <span>Hafiz and Team</span>
        </div>
    """, unsafe_allow_html=True)


# --- Milestone 1: PDF Parsing and Chunk Preparation ---
def extract_text_from_pdfs(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        doc = fitz.open(stream=pdf.read(), filetype="pdf")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500, 
        chunk_overlap=500, 
        length_function=len
    )
    return text_splitter.split_text(text)

# --- Milestone 2: Embedding, Indexing, and Retrieval ---
def create_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- Milestone 3: Cohere LLM Integration ---
def get_cohere_llm():
    # Fetch key using our new robust helper function
    api_key = get_api_key()
    
    llm = ChatCohere(
        model="command-r7b-12-2024", 
        temperature=0.3, 
        cohere_api_key=api_key
    )
    return llm

# --- Milestone 4: Streamlit Interface and Session Handling ---
def main():
    st.set_page_config(page_title="StudyMate", page_icon="📝", layout="wide")
    
    # Apply the custom styling and footer
    apply_custom_css()
    
    if "history" not in st.session_state:
        st.session_state.history = []

    # Custom Header
    st.markdown("<h1>📝 StudyMate <span style='font-size:24px; color:#64748b; font-weight: 500; letter-spacing: normal;'>AI Academic Assistant</span></h1>", unsafe_allow_html=True)
    st.markdown("Upload your textbooks, lecture notes, or research papers and interact with them in real-time.")
    st.divider()

    with st.sidebar:
        st.header("📂 Document Upload")
        pdf_docs = st.file_uploader("Drop PDFs Here", accept_multiple_files=True, type=['pdf'])
        
        if st.button("Process Documents 🚀"):
            # Check for API Key using the new helper
            if not get_api_key():
                st.error("Missing COHERE_API_KEY in Streamlit Secrets.")
            elif not pdf_docs:
                st.warning("Please upload at least one PDF.")
            else:
                with st.spinner("Extracting text and building semantic index..."):
                    raw_text = extract_text_from_pdfs(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vector_store = create_vector_store(text_chunks)
                    
                    st.session_state.vector_store = vector_store
                    st.success("✅ Knowledge base built! Ready for questions.")

    st.subheader("Ask your Documents")
    user_question = st.text_input("", placeholder="E.g., What are the main concepts discussed in chapter 2?")

    if user_question:
        if "vector_store" not in st.session_state:
            st.error("Please upload and process your PDFs in the sidebar first.")
        else:
            with st.spinner("Analyzing documents..."):
                llm = get_cohere_llm()
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                
                prompt_template = """
                You are StudyMate, a helpful academic assistant. Answer the question based strictly on the following context.
                If you cannot find the answer in the context, explicitly state that you don't know. Do not hallucinate or invent information.
                
                Context:
                {context}
                
                Question: {question}
                
                Answer:
                """
                PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
                
                rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | PROMPT
                    | llm
                    | StrOutputParser()
                )
                
                answer = rag_chain.invoke(user_question)
                source_docs = retriever.invoke(user_question)
                
                st.session_state.history.append({"question": user_question, "answer": answer})
                
                st.markdown("### 💡 Answer:")
                st.info(answer)
                
                with st.expander("🔍 View Source References"):
                    for i, doc in enumerate(source_docs):
                        st.markdown(f"**Source {i+1}:**")
                        st.caption(doc.page_content)
                        st.divider()

    if st.session_state.history:
        st.divider()
        st.subheader("🕒 Session History")
        
        transcript = "StudyMate Session Transcript\n" + "="*30 + "\n\n"
        
        for idx, interaction in enumerate(reversed(st.session_state.history)):
            st.markdown(f"**Q:** {interaction['question']}")
            st.markdown(f"**A:** {interaction['answer']}")
            st.markdown("---")
            
            transcript += f"Q: {interaction['question']}\nA: {interaction['answer']}\n\n"
            
        st.download_button(
            label="⬇️ Export Transcript",
            data=transcript,
            file_name="studymate_transcript.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()
