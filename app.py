import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os

# --- New od-parser Import ---
# This replaces PyPDF2 for much better and more advanced PDF parsing.
from od_parse.main import parse_pdf

# --- Langchain and AI Imports ---
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# --- Your Custom Project Modules ---
from data_analysis.data_analysis import (
    parse_llm_summary,
    display_metric_summary,
    predict_conditions,
    download_metrics
)
from data_diagrams.data_diagrams import (
    plot_metric_comparison,
    generate_radial_health_score,
    display_reference_table,
    create_clinical_summary_pdf
)
from data_analysis.predictive import DiseasePredictor
from data_analysis.similarity import ReportComparator
from data_analysis.trends import show_trend_analysis, detect_anomalies

# Load environment variables from .env file
load_dotenv()


# --- New Advanced PDF Processing Function using od-parser ---
def process_medical_reports(pdf_docs):
    """
    Processes uploaded medical reports using od-parser to extract both
    plain text and structured tables for the highest accuracy.
    """
    full_text_content = ""
    all_extracted_tables = []
    temp_file_path = "temp_report.pdf"

    for pdf in pdf_docs:
        # od-parser works with file paths, so we save the uploaded file temporarily
        with open(temp_file_path, "wb") as f:
            f.write(pdf.getvalue())

        try:
            st.info(f"🧠 Analyzing '{pdf.name}' with od-parser...")
            
            # Use the main parse_pdf function with the 'full' pipeline
            result = parse_pdf(
                file_path=temp_file_path,
                pipeline_type="full",
                use_deep_learning=True
            )

            # Extract the text and tables from the result dictionary
            # The key for text is 'full_text'
            if result.get("full_text"):
                full_text_content += result.get("full_text") + "\n\n---\n\n"
            
            if result.get("tables"):
                all_extracted_tables.extend(result.get("tables"))

        except Exception as e:
            st.error(f"❌ Error processing '{pdf.name}': {e}")
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    if not full_text_content and not all_extracted_tables:
        st.error("⚠️ od-parser could not extract any content from the PDFs.")
        return None

    return {"text": full_text_content, "tables": all_extracted_tables}


def summarize_text(text):
    """Generates a summary of the text using an LLM."""
    try:
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            st.error("❌ API key missing! Please set TOGETHER_API_KEY.")
            return None

        llm = ChatOpenAI(
            base_url="https://api.together.xyz/v1",
            api_key=api_key,
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        )

        summary_prompt = (
            "You are a medical expert assistant. Summarize the following medical report text. "
            "Include key findings, diagnoses, and recommendations in bullet points. "
            "If you do not find structured tables in the text, extract medical metrics into a JSON array "
            "with 'metric', 'value', 'reference_range', and 'unit' fields. "
            f"Here is the report text:\n\n{text}"
        )
        summary = llm.predict(summary_prompt)
        return summary
    except Exception as e:
        st.error(f"❌ Error generating summary: {e}")
        return None


def get_text_chunks(text):
    """Splits text into manageable chunks for vector embedding."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    if not chunks:
        st.error("⚠️ No valid text chunks found for the chat model.")
        return None
    return chunks


def get_vectorstore(text_chunks):
    """Creates a FAISS vector store from text chunks."""
    if not text_chunks:
        raise ValueError("Error: No text chunks provided for vector store creation!")
    
    model_name = "sentence-transformers/all-mpnet-base-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    """Initializes the conversational retrieval chain."""
    try:
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            st.error("❌ API key missing! Please set TOGETHER_API_KEY.")
            return None

        llm = ChatOpenAI(
            base_url="https://api.together.xyz/v1",
            api_key=api_key,
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        )
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        return conversation_chain
    except Exception as e:
        st.error(f"❌ Error initializing chat: {e}")
        return None


def handle_userinput(user_question):
    """Handles user questions and displays the conversation history."""
    if st.session_state.conversation:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            role = "You" if i % 2 == 0 else "Bot"
            with st.chat_message(role):
                st.write(message.content)
    else:
        st.warning("⚠️ Please process a document first to start the conversation.")


def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(page_title="Medical Report Analyzer", page_icon="⚕️", layout="wide")

    # Initialize session state variables
    for key in ["conversation", "chat_history", "pdf_text", "summary"]:
        if key not in st.session_state:
            st.session_state[key] = None

    st.header("⚕️ Medical Report Analyzer")

    with st.sidebar:
        st.subheader("📄 Your Medical Documents")
        pdf_docs = st.file_uploader("Upload PDF reports and click 'Process'", accept_multiple_files=True)

        if st.button("🚀 Process Documents"):
            with st.spinner("Analyzing documents... This may take a moment."):
                if not pdf_docs:
                    st.error("⚠️ Please upload at least one PDF file!")
                    return

                # --- Use the new od-parser function to get both text and tables ---
                processed_data = process_medical_reports(pdf_docs)
                if not processed_data:
                    return

                raw_text = processed_data["text"]
                tables = processed_data["tables"]
                st.session_state.pdf_text = raw_text

                metrics_df = pd.DataFrame()
                # --- New Logic: Create DataFrame directly from extracted tables ---
                if tables:
                    st.success(f"✅ Extracted {len(tables)} data table(s) directly from PDFs!")
                    try:
                        table_data = tables[0].get('data', [])
                        if len(table_data) > 1:
                            headers = table_data[0]
                            rows = table_data[1:]
                            metrics_df = pd.DataFrame(rows, columns=headers)
                        else:
                            st.warning("Found a table, but it was empty or malformed.")
                    except Exception as e:
                        st.error(f"Could not convert extracted table to a DataFrame. Error: {e}")
                else:
                    st.warning("⚠️ No data tables found. Metrics will be extracted from text via LLM.")

                # Generate summary and extract metrics via LLM as a fallback
                summary = summarize_text(raw_text)
                if summary:
                    st.session_state.summary = summary
                    if metrics_df.empty:
                        parsed_data = parse_llm_summary(summary)
                        metrics_df = pd.DataFrame(parsed_data)
                
                # Setup the chat functionality (RAG)
                text_chunks = get_text_chunks(raw_text)
                if text_chunks:
                    try:
                        vectorstore = get_vectorstore(text_chunks)
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                        st.success("✅ Processing complete! You can now ask questions.")
                    except Exception as e:
                        st.error(f"❌ Error setting up conversation: {e}")

    # Main page layout
    if st.session_state.summary:
        st.subheader("📊 Analysis Dashboard")
        
        if not metrics_df.empty:
            display_metric_summary(metrics_df.to_dict('records'))
            predict_conditions(metrics_df.to_dict('records'))

            st.subheader("📈 Interactive Visualizations")
            col1, col2 = st.columns(2)
            with col1:
                plot_metric_comparison(metrics_df)
            with col2:
                generate_radial_health_score(metrics_df)
            
            display_reference_table(metrics_df)
            
            pdf_report = create_clinical_summary_pdf(metrics_df)
            st.download_button("📄 Download PDF Report", pdf_report, "clinical_report.pdf", "application/pdf")
            download_metrics(metrics_df.to_dict('records'))
        else:
            st.info("No structured metrics were found to generate a dashboard.")
    
    # Chat interface
    st.subheader("💬 Ask Your Questions Here")
    user_question = st.text_input("E.g., 'What was my white blood cell count?'")
    if user_question:
        handle_userinput(user_question)

if __name__ == '__main__':
    main()
