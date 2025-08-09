import streamlit as st
import os
import json
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import time

# --- Helper Functions ---

def get_pdf_text(pdf_docs):
    """Extracts text from a list of uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text
    return text

def get_text_chunks(text):
    """Splits text into manageable chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

@st.cache_resource
def get_vector_store(_text_chunks):
    """Creates a vector store from text chunks using HuggingFace embeddings."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(texts=_text_chunks, embedding=embeddings)
    return vector_store

def get_conversation_chain(vector_store):
    """Creates the main conversational retrieval chain for technical questions."""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.5)
    
    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True
    )
    
    custom_prompt_template = """
You are a highly skilled AI assistant helping technicians troubleshoot appliances using ONLY the uploaded technical manual.

**Guidelines:**
- Use only information from the manual (context below).
- Be direct, precise, and practical.
- Do NOT guess or add outside knowledge.
- If the manual doesn't contain enough information, say: "The manual does not contain information about this issue."

---

🔎 **Problem Analysis**:

Provide a detailed technical explanation of possible causes for the user’s issue. Mention key components, symptoms, and likely root causes. Include safety notes if necessary. Write naturally, but focus on clarity and depth.

---

🛠️ **Actionable Steps**:

List 4-6 concise technician instructions (no more than 1 line each). Each should start with a title and be followed by a short, plain-language instruction.

Example format:
1. **Test Gas Valve (High Priority)**: Unplug the dryer and test the gas valve coils with a multimeter.
2. **Check Thermal Fuse**: Remove the back panel and test continuity across the fuse.
3. **Inspect Exhaust Vent**: Look for any blockages or lint buildup in the external vent.

If no actions are found, say: “No actionable steps found in the manual.”

---

Chat History: {chat_history}
Context: {context}
Question: {question}

Begin response below:
"""
    
    CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_prompt_template)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": CUSTOM_QUESTION_PROMPT}
    )
    return conversation_chain

def get_user_intent(user_question):
    """
    Uses the LLM to classify the user's intent to route the conversation.
    """
    classifier_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.1)

    prompt = PromptTemplate(
        template="""
        Analyze the user's input and classify its primary intent.
        Respond with ONLY a single JSON object with a single key "intent".
        The possible values for the intent are:
        - "TECHNICAL_QUESTION" (For any question about troubleshooting, parts, steps, or appliance behavior)
        - "POSITIVE_FEEDBACK" (When the user indicates success, satisfaction, or gratitude, e.g., "it works now", "thanks", "fixed")
        - "NEGATIVE_FEEDBACK" (When the user indicates frustration, failure, or that something is not working, e.g., "still broken", "irritating")

        User Input: "{question}"
        JSON Response:
        """,
        input_variables=["question"],
    )

    chain = prompt | classifier_llm

    try:
        response = chain.invoke({"question": user_question})
        cleaned_response = response.content.strip().replace("```json", "").replace("```", "").strip()
        intent_data = json.loads(cleaned_response)
        return intent_data.get("intent", "TECHNICAL_QUESTION")
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Error classifying intent: {e}. Defaulting to technical question.")
        return "TECHNICAL_QUESTION"


def typing_effect(text, delay=0.015):
    """Yields text character by character for a typing animation."""
    output = ""
    for char in text:
        output += char
        yield output
        time.sleep(delay)

# --- Main Streamlit App ---

def main():
    st.set_page_config(
        page_title="MechBro - Technical Manual Assistant", 
        page_icon="⚙️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
        .main .block-container {padding-top: 1rem; padding-left: 2rem; max-width: 900px;}
        [data-testid="stSidebar"] {background-color: #2c3e50 !important; color: white !important;}
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, 
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {color: white !important;}
        .stButton > button {width: 100%; border-radius: 6px; border: none; background: #3498db; color: white; font-weight: 500; padding: 0.5rem 1rem;}
        .stButton > button:hover {background: #2980b9;}
        [data-testid="stChatMessage"] {padding: 1rem 1.5rem; border-radius: 8px; margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05);}
        [data-testid="stChatMessageContent"] p {margin: 0;}
    </style>
    """, unsafe_allow_html=True)

    # --- Session State Initialization ---
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "google_api_key_set" not in st.session_state:
        st.session_state.google_api_key_set = False
    if "typing" not in st.session_state:
        st.session_state.typing = False
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None

    # --- Sidebar ---
    with st.sidebar:

        st.markdown("### 1. API Configuration")
        api_key = st.text_input(
            "Gemini AI API Key:", 
            type="password", 
            key="api_key_input", 
            placeholder="Enter your API key here..."
        )
        if st.button("Set API Key"):
            if api_key:
                os.environ["GOOGLE_API_KEY"] = api_key
                st.session_state.google_api_key_set = True
                # --- MODIFIED: Use st.toast for a temporary success message ---
                st.toast("API Key configured!", icon="✅")
            else:
                st.warning("Please enter a valid API key.")
        

        st.markdown("### 2. Document Processing")
        pdf_docs = st.file_uploader(
            "Upload a PDF manual:", 
            accept_multiple_files=True, 
            type="pdf"
        )
        if st.button("Process Documents"):
            if not st.session_state.google_api_key_set:
                st.error("Please set your API Key first.")
            elif not pdf_docs:
                st.error("Please upload at least one PDF.")
            else:
                with st.spinner("Processing documents..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vector_store = get_vector_store(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vector_store)
                # Use st.toast for this success message as well
                st.toast("Documents processed successfully!", icon="📄")
        

        st.markdown("### 3. Session Management")
        if st.button("Reset Session", help="Clears chat and uploaded documents", type="primary"):
            st.session_state.conversation = None
            st.session_state.chat_history = []
            st.session_state.pending_question = None
            st.cache_resource.clear()
            st.toast("Session has been reset.", icon="🔄")
            # A small sleep helps ensure the toast has time to display before the rerun
            time.sleep(1) 
            st.rerun()

    # --- Main Page Title ---
    st.markdown("""
    <div style="display: flex; align-items: baseline; gap: 12px;">
      <h1 style="margin: 0;">⚙️ MechBro</h1>
      <p style="margin: 0; font-size: 1.1rem; color: #556677;">Your personal technical manual assistant</p>
    </div>
    """, unsafe_allow_html=True)


    # --- Main Chat Display ---
    for idx, message in enumerate(st.session_state.chat_history):
        role = message["role"]
        with st.chat_message(role):
            if role == "assistant" and idx == len(st.session_state.chat_history) - 1 and st.session_state.typing:
                response_text = message["content"]
                placeholder = st.empty()
                for partial_response in typing_effect(response_text):
                    placeholder.markdown(f"**MechBro:** {partial_response}")
                st.session_state.typing = False
            else:
                prefix = "**Technician:** " if role == "user" else "**MechBro:** "
                st.markdown(f"{prefix}{message['content']}")

    # --- Processing logic for a pending TECHNICAL question ---
    if st.session_state.pending_question:
        user_question = st.session_state.pending_question
        st.session_state.pending_question = None

        if st.session_state.conversation:
            response = st.session_state.conversation({'question': user_question})
            st.session_state.chat_history[-1] = {"role": "assistant", "content": response['answer']}
            st.session_state.typing = True
            st.rerun()
        else:
            st.session_state.chat_history[-1] = {"role": "assistant", "content": "Please process a document first."}
            st.rerun()

    # --- Chat input logic with LLM-based intent routing ---
    if user_question := st.chat_input("Describe the issue or ask about the manual..."):
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        intent = get_user_intent(user_question)
        
        if intent == "TECHNICAL_QUESTION":
            st.session_state.chat_history.append({"role": "assistant", "content": "Hold on, flipping pages..."})
            st.session_state.pending_question = user_question
            
        elif intent == "POSITIVE_FEEDBACK":
            response_text = "That's great to hear! Glad I could help. Let me know if you need anything else."
            st.session_state.chat_history.append({"role": "assistant", "content": response_text})
            st.session_state.typing = True
            
        elif intent == "NEGATIVE_FEEDBACK":
            response_text = "I understand that can be frustrating. Let's try to pinpoint the problem. Please describe the specific technical issue you are seeing."
            st.session_state.chat_history.append({"role": "assistant", "content": response_text})
            st.session_state.typing = True
        
        st.rerun()


if __name__ == '__main__':

    main()
