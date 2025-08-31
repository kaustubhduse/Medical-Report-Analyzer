import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
import tempfile
import re # Import the regular expression module

# Use the main parse_pdf function which can handle different pipelines
from od_parse import parse_pdf, convert_to_markdown

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Assuming these are your custom modules
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

# Load environment variables
load_dotenv()

# --- NEW: Medical Data Definitions (from your React code) ---
medical_entities = {
    "diseases": ["diabetes", "cancer", "asthma", "hypertension", "stroke"],
    "medications": ["aspirin", "metformin", "ibuprofen", "insulin", "paracetamol"],
    "symptoms": ["fever", "cough", "fatigue", "headache", "dizziness"],
}

fatal_diseases = ["cancer", "stroke", "hypertension"]

# --- NEW: Function to highlight text (Python version of your JS function) ---
def highlight_entities(text):
    """
    Highlights medical entities in the summary text using HTML styling for Streamlit.
    """
    if not text:
        return ""
    
    # Use inline CSS for styling within Streamlit's markdown
    style_map = {
        "diseases": 'style="color: #F87171; font-weight: bold;"',  # Red
        "medications": 'style="color: #60A5FA; font-weight: bold;"', # Blue
        "symptoms": 'style="color: #4ADE80; font-weight: bold;"'    # Green
    }
    
    processed_text = text
    for category, words in medical_entities.items():
        for word in words:
            # Use regex to find whole words only, case-insensitive
            regex = re.compile(fr'\b({re.escape(word)})\b', re.IGNORECASE)
            # Replace with a span tag containing the style
            replacement = f'<span {style_map[category]}>\\1</span>'
            processed_text = regex.sub(replacement, processed_text)
            
    return processed_text.replace("\n", "<br>")

# --- NEW: Function to check for serious conditions ---
def check_for_fatal_diseases(summary_text):
    """
    Checks if any serious conditions are mentioned in the summary and displays a warning.
    """
    if not summary_text:
        return

    # Check if any fatal disease is mentioned (case-insensitive)
    found_diseases = [
        disease for disease in fatal_diseases 
        if disease.lower() in summary_text.lower()
    ]

    if found_diseases:
        st.warning(
            f"**Health Alert:** A potentially serious condition ({', '.join(found_diseases)}) "
            "has been detected in the summary. Please consult a medical professional immediately."
        )


# --- MODIFIED FUNCTION to accept advanced options ---
def get_pdf_text(pdf_docs, pipeline_type="default", use_deep_learning=False):
    full_text = ""
    st.info(f"Running '{pipeline_type}' pipeline...")
    if use_deep_learning:
        st.warning("Deep Learning is enabled. Processing will be slower but more accurate. 🧠")

    for pdf in pdf_docs:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf.getvalue())
            tmp_file_path = tmp_file.name
        try:
            parsed_data = parse_pdf(
                file_path=tmp_file_path,
                pipeline_type=pipeline_type,
                use_deep_learning=use_deep_learning
            )
            markdown_text = convert_to_markdown(
                parsed_data,
                include_images=False,
                include_tables=True,
                include_forms=True,
                include_handwritten=True
            )
            full_text += markdown_text + "\n\n---\n\n"
        except Exception as e:
            st.error(f"⚠️ Error parsing {pdf.name} with od-parse: {e}")
        finally:
            os.remove(tmp_file_path)

    if not full_text.strip():
        st.error("⚠️ No readable text found in uploaded PDFs! od-parse could not extract content.")
        return None
    return full_text


def summarize_text(text):
    try:
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            st.error("❌ API key missing! Please set TOGETHER_API_KEY in your environment variables.")
            return None

        llm = ChatOpenAI(
            base_url="https://api.together.xyz/v1",
            api_key=api_key,
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        )
        summary_prompt = (
            "You are a medical expert assistant. Carefully read and summarize the following medical report, which is in Markdown format. "
            "Your summary should include:\n"
            "- Patient's name (if available)\n"
            "- Date of the report (if available)\n"
            "- Relevant medical history or background (in bullet points)\n"
            "- Key findings and observations (in bullet points)\n"
            "- Diagnoses or impressions (if mentioned)\n"
            "- Recommendations for further tests, treatments, or follow-up (in bullet points)\n"
            "\n"
            "Extract medical metrics as a JSON array with the following fields:\n"
            "- metric: Test name\n"
            "- value: Numeric result\n"
            "- reference_range: X-Y format\n"
            "- unit: Measurement unit\n"
            "Return metrics only in JSON format and other information in plain text.\n"
            f"{text}"
        )
        summary = llm.predict(summary_prompt)
        return summary
    except Exception as e:
        st.error(f"❌ Error generating summary: {e}")
        return None


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    if not chunks:
        st.error("⚠️ No valid text chunks found! Ensure PDFs contain readable text.")
        return None
    return chunks


def get_vectorstore(text_chunks):
    if not text_chunks:
        raise ValueError("Error: No text chunks provided for FAISS indexing!")
    model_name = "sentence-transformers/all-mpnet-base-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        metadatas=[{}] * len(text_chunks))
    return vectorstore


def get_conversation_chain(vectorstore):
    try:
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            st.error("❌ API key missing! Please set TOGETHER_API_KEY in your environment variables.")
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
    if st.session_state.conversation:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']
        for i, message in enumerate(st.session_state.chat_history):
            role = "User" if i % 2 == 0 else "Bot"
            st.write(f"**{role}:** {message.content}")
    else:
        st.warning("⚠️ No conversation started yet! Upload PDFs and process them first.")


def main():
    st.set_page_config(page_title="Medical Chatbot", page_icon="⚕️", layout="wide")

    # Initialize session state keys
    for key in ["conversation", "chat_history", "summary", "metrics_df"]:
        if key not in st.session_state:
            st.session_state[key] = None

    with st.sidebar:
        st.header("⚕️ Chat with Medical Reports")
        st.subheader("📄 Upload Medical Reports (PDF)")
        pdf_docs = st.file_uploader("Upload PDFs and click 'Process'", accept_multiple_files=True)
        
        st.subheader("⚙️ Parsing Options")
        pipeline_type = st.selectbox(
            "Select Parsing Pipeline:",
            options=["default", "forms", "structure", "full"],
            help="Choose the type of content to focus on. 'Full' is the most comprehensive."
        )
        use_deep_learning = st.checkbox(
            "Enable Deep Learning",
            value=False,
            help="Slower but provides more accurate extraction for tables and forms."
        )

        if st.button("🚀 Process"):
            if not pdf_docs:
                st.error("⚠️ Please upload at least one PDF file!")
                return
            
            with st.spinner("⏳ Processing... This may take a moment."):
                raw_text = get_pdf_text(pdf_docs, pipeline_type, use_deep_learning)
                if not raw_text:
                    return

                # Generate and store summary
                summary = summarize_text(raw_text)
                if not summary:
                    st.warning("⚠️ Could not generate summary.")
                    return
                st.session_state.summary = summary
                
                # Extract health metrics for display
                parsed_data = parse_llm_summary(summary)
                st.session_state.metrics_df = pd.DataFrame(parsed_data)

                # Setup chat functionality
                text_chunks = get_text_chunks(raw_text)
                if text_chunks:
                    try:
                        vectorstore = get_vectorstore(text_chunks)
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                        st.success("✅ Processing complete!")
                    except ValueError as e:
                        st.error(f"❌ Error: {e}")

    # --- MAIN PAGE DISPLAY ---
    st.header("Medical Report Analysis")

    # --- MODIFIED: Display summary if it exists in session state ---
    if st.session_state.summary:
        st.subheader("📝 Medical Report Summary")

        # Check for and display a health alert if necessary
        check_for_fatal_diseases(st.session_state.summary)

        # Display color legend
        st.markdown("""
        **Legend:** <span style="color: #F87171; font-weight: bold;">■ Diseases</span>&nbsp;&nbsp;
        <span style="color: #60A5FA; font-weight: bold;">■ Medications</span>&nbsp;&nbsp;
        <span style="color: #4ADE80; font-weight: bold;">■ Symptoms</span>
        """, unsafe_allow_html=True)
        
        # Display the highlighted summary
        highlighted_summary = highlight_entities(st.session_state.summary)
        st.markdown(f'<div style="background-color: #f0f2f6; border-radius: 10px; padding: 15px; border: 1px solid #dde0e3;">{highlighted_summary}</div>', unsafe_allow_html=True)

        st.download_button(
            "📥 Download Full Summary",
            st.session_state.summary.encode('utf-8'),
            file_name="medical_summary.txt",
            mime="text/plain"
        )

        st.markdown("---") # Visual separator

        # Display charts and other data if metrics were parsed
        if st.session_state.metrics_df is not None and not st.session_state.metrics_df.empty:
            st.subheader("📈 Interactive Visual Analysis")
            col1, col2 = st.columns(2)
            with col1:
                plot_metric_comparison(st.session_state.metrics_df)
            with col2:
                generate_radial_health_score(st.session_state.metrics_df)

            display_reference_table(st.session_state.metrics_df)
            
            # Add download buttons for reports
            pdf_report = create_clinical_summary_pdf(st.session_state.metrics_df)
            st.download_button(
                "📄 Download PDF Report",
                pdf_report,
                "clinical_report.pdf",
                "application/pdf"
            )
            download_metrics(st.session_state.metrics_df.to_dict('records'))

    else:
        st.info("Upload a medical report and click 'Process' to see the analysis.")

    st.markdown("---")
    
    # Chat interface
    st.subheader("💬 Chat with your Report")
    user_question = st.text_input("Ask a question about your medical report:")
    if user_question:
        handle_userinput(user_question)

if __name__ == '__main__':
    main()