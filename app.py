import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import os
import json
import re

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Assuming your custom modules are in the correct path
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
from data_analysis.trends import show_trend_analysis

# Load environment variables
load_dotenv()

def get_pdf_text(pdf_docs):
    """Extracts text from a list of PDF documents."""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {e}")
    if not text.strip():
        st.warning("No readable text found in the uploaded PDF(s).")
        return None
    return text

def summarize_text(text, filename):
    """Generates a structured summary for the text of a single PDF."""
    st.info(f"Generating summary for {filename}...")
    try:
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            st.error("API key for 'Together AI' is missing.")
            return None

        llm = ChatOpenAI(
            base_url="https://api.together.xyz/v1",
            api_key=api_key,
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        )

        summary_prompt = (
             "You are a medical expert assistant. Carefully read and summarize the following medical report. "
            "Your summary should include:\n"
            "- Patient's name (if available)\n"
            "- Date of the report (if available, in YYYY-MM-DD format)\n"
            "- Relevant medical history or background (in bullet points)\n"
            "- Key findings and observations (in bullet points)\n"
            "- Diagnoses or impressions (if mentioned)\n"
            "- Recommendations for further tests, treatments, or follow-up (in bullet points)\n"
            "\n"
            "After the summary, extract medical metrics as a JSON array with these fields:\n"
            "- metric: The name of the test (e.g., 'Hemoglobin').\n"
            "- value: The numeric result of the test.\n"
            "- reference_range: The normal range in 'X-Y' format (e.g., '13.5-17.5').\n"
            "- unit: The unit of measurement (e.g., 'g/dL').\n"
            "IMPORTANT: The JSON array must be the very last part of your response, with no text after it.\n"
            f"Here is the report:\n{text}"
        )
        summary = llm.predict(summary_prompt)
        return summary
    except Exception as e:
        st.error(f"Error generating summary for {filename}: {e}")
        return None

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)
    if not chunks:
        st.error("Failed to create text chunks.")
        return None
    return chunks

def get_vectorstore(text_chunks):
    if not text_chunks:
        raise ValueError("Cannot create vector store: No text chunks provided.")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    try:
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            st.error("API key for 'Together AI' is missing.")
            return None
        llm = ChatOpenAI(
            base_url="https://api.together.xyz/v1",
            api_key=api_key,
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        )
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        return ConversationalRetrievalChain.from_llm(
            llm=llm, retriever=vectorstore.as_retriever(), memory=memory
        )
    except Exception as e:
        st.error(f"Error initializing chat: {e}")
        return None

def handle_userinput(user_question):
    if st.session_state.conversation:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']
        for i, message in enumerate(st.session_state.chat_history):
            role = "user" if i % 2 == 0 else "assistant"
            with st.chat_message(role):
                st.markdown(message.content)
    else:
        st.warning("Please upload and process reports first.")

def main():
    st.set_page_config(page_title="Medical Report Analyzer", page_icon="⚕️", layout="wide")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("⚕️ Chat with Your Medical Reports")
    st.write("Upload reports to get an AI summary, data analysis, and an interactive chatbot.")

    if st.session_state.chat_history:
        for i, message in enumerate(st.session_state.chat_history):
            role = "user" if i % 2 == 0 else "assistant"
            with st.chat_message(role):
                st.markdown(message.content)

    user_question = st.chat_input("Ask a question about your reports...")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("📄 Upload Your Reports")
        pdf_docs = st.file_uploader("Upload PDFs and click 'Process'", accept_multiple_files=True, type="pdf")

        if st.button("🚀 Process Reports"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file.")
                return

            with st.spinner("Analyzing reports..."):
                all_texts, all_summaries, all_parsed_data = [], [], []

                for i, pdf in enumerate(pdf_docs):
                    text = get_pdf_text([pdf])
                    if text:
                        all_texts.append(text)
                        summary = summarize_text(text, pdf.name)
                        if summary:
                            # Attempt to extract a date from the summary for trend analysis
                            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', summary)
                            date = date_match.group(1) if date_match else f"Report_{i+1}"
                            
                            parsed_data = parse_llm_summary(summary)
                            if parsed_data:
                                # Add date to each metric entry
                                for item in parsed_data:
                                    item['date'] = date
                                all_parsed_data.append({'filename': pdf.name, 'data': parsed_data, 'date': date})

                if not all_texts:
                    st.error("Processing failed. No readable text found.")
                    return

                # Setup chatbot
                combined_text = "\n\n--- END OF REPORT ---\n\n".join(all_texts)
                text_chunks = get_text_chunks(combined_text)
                if text_chunks:
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.success("Processing complete! You can now ask questions.")

                # Display Analysis
                st.session_state.analysis_placeholder = st.empty()
                with st.session_state.analysis_placeholder.container():
                    st.markdown("---")
                    st.header("📊 Report Analysis")
                    for report in all_parsed_data:
                        with st.expander(f"**Analysis for {report['filename']}**"):
                            metrics_df = pd.DataFrame(report['data'])
                            if not metrics_df.empty:
                                metrics_df['value'] = pd.to_numeric(metrics_df['value'], errors='coerce').fillna(0)
                                metrics_df['status'] = metrics_df['status'].astype(str).fillna('Unknown')
                                
                                plot_metric_comparison(metrics_df)
                                generate_radial_health_score(metrics_df)
                                display_reference_table(metrics_df)
                                
                                pdf_report = create_clinical_summary_pdf(metrics_df)
                                st.download_button(
                                    f"📄 Download PDF Report for {report['filename']}",
                                    pdf_report, f"clinical_report_{report['filename']}.pdf", "application/pdf"
                                )
                    
                    # --- THIS IS THE FIX ---
                    # Trend analysis for multiple reports
                    if len(all_parsed_data) > 1:
                        st.header("📈 Trend Analysis")
                        
                        # 1. Combine all data into a single long-form DataFrame
                        flat_list = [item for report in all_parsed_data for item in report['data']]
                        historical_df_long = pd.DataFrame(flat_list)
                        
                        # 2. Convert to wide format for the trend analysis function
                        if 'date' in historical_df_long.columns and 'metric' in historical_df_long.columns:
                            historical_df_wide = historical_df_long.pivot_table(
                                index='date', columns='metric', values='value'
                            ).reset_index()
                            
                            # 3. Get the list of metrics to plot
                            metrics_to_plot = [col for col in historical_df_wide.columns if col != 'date']
                            
                            # 4. Call the function with the correct arguments
                            show_trend_analysis(historical_df_wide, metrics_to_plot)
                        else:
                            st.warning("Could not generate trend analysis due to missing date or metric data.")

if __name__ == '__main__':
    main()
