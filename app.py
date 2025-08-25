import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


import os
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


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    if not text.strip():
        st.error("⚠️ No readable text found in uploaded PDFs! Please ensure they contain selectable text.")
        return None
    return text


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
            "You are a medical expert assistant. Carefully read and summarize the following medical report. "
            "Your summary should include:\n"
            "- Patient's name (if available)\n"
            "- Date of the report (if available)\n"
            "- Relevant medical history or background (in bullet points)\n"
            "- Key findings and observations (in bullet points)\n"
            "- Diagnoses or impressions (if mentioned)\n"
            "- Recommendations for further tests, treatments, or follow-up (in bullet points)\n"
            "\n"
            "Extract medical metrics as JSON array with the following fields:\n"
            "- metric: Test name\n"
            "- value: Numeric result\n"
            "- reference_range: X-Y format\n"
            "-unit: Measurement unit \n"
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
        metadatas=[{}]*len(text_chunks))
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
    st.set_page_config(page_title="Medical Chatbot", page_icon="⚕️")

    for key in ["conversation", "chat_history", "pdf_text", "text_chunks", "vectorstore", "summary"]:
        if key not in st.session_state:
            st.session_state[key] = None

    st.header("⚕️ Chat with Medical Reports")

    user_question = st.text_input("Ask a question about your medical report:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("📄 Upload Medical Reports (PDF)")
        pdf_docs = st.file_uploader("Upload PDFs and click 'Process'", accept_multiple_files=True)

        if st.button("🚀 Process"):
            with st.spinner("⏳ Processing..."):
                if not pdf_docs:
                    st.error("⚠️ Please upload at least one PDF file!")
                    return

                raw_text = get_pdf_text(pdf_docs)
                if not raw_text:
                    return

                st.session_state.pdf_text = raw_text

                # Generate summary but DO NOT display it
                summary = summarize_text(raw_text)
                if summary:
                    st.session_state.summary = summary
                    st.download_button(
                        "📥 Download Medical Summary", 
                        summary.encode('utf-8'), 
                        file_name="medical_summary.txt", 
                        mime="text/plain"
                    )
                else:
                    st.warning("⚠️ Could not generate summary.")
                    return
                
                # Extract health metrics and display
                parsed_data = parse_llm_summary(summary)
                 # Convert to DataFrame for diagrams
                metrics_df = pd.DataFrame(parsed_data)
                
                # Text chunking + vectorstore + conversation setup
                text_chunks = get_text_chunks(raw_text)
                if not text_chunks:
                    return

                st.session_state.text_chunks = text_chunks

                try:
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.vectorstore = vectorstore
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.success("✅ Processing complete! You can now ask questions.")
                except ValueError as e:
                    st.error(f"❌ Error: {e}")
                
                # 1. Disease risk prediction
                predictor = DiseasePredictor()
                metrics_dict = {item['metric']: item['value'] for item in parsed_data}
                risk_assessment = predictor.predict_risk(metrics_dict)
                
                st.subheader("🩺 Disease Risk Assessment")
                if 'anemia' in risk_assessment:
                    st.progress(risk_assessment['anemia']['probability'])
                    st.markdown(risk_assessment['anemia']['advice'])
                    
                # 2. Similar Report Detection
                comparator = ReportComparator(st.session_state.vectorstore)
                # Compute embedding for the current report text
                embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
                text_embedding = embedding_model.embed_documents([raw_text])[0]
                similar_reports = comparator.find_similar_reports(text_embedding)
                
                if similar_reports:
                    st.subheader("🔍 Similar Reports Found")
                    for report, similarity in similar_reports:
                        st.write(f"**{similarity:.1%} match**: {report['diagnosis']}")

                
                # 3. Time Series Analysis
                if len(pdf_docs) > 1:  # Only show trends if multiple reports
                    # Process multiple reports to extract historical data
                    def process_multiple_reports(pdf_docs):
                        historical_data = []
                        for pdf in pdf_docs:
                            text = get_pdf_text([pdf])
                            if text:
                                summary = summarize_text(text)
                                if summary:
                                    parsed = parse_llm_summary(summary)
                                    report_date = pdf.name
                                    for item in parsed:
                                        item["date"] = report_date
                                    historical_data.extend(parsed)
                        return historical_data

                    historical_data = process_multiple_reports(pdf_docs)
                    if historical_data:
                        # Convert to DataFrame
                        historical_df = pd.DataFrame(historical_data)
                        # Force numeric conversion for the "value" column if it exists
                        if "value" in historical_df.columns:
                            historical_df["value"] = pd.to_numeric(historical_df["value"], errors="coerce")    
                        if "date" in historical_df.columns:
                            historical_df["date"] = historical_df["date"].astype(str)

                
                display_metric_summary(parsed_data)
                predict_conditions(parsed_data)
                
                # New visualizations
                st.subheader("📈 Interactive Visual Analysis")
                col1, col2 = st.columns(2)
                with col1:
                    plot_metric_comparison(metrics_df)
                with col2:
                    generate_radial_health_score(metrics_df)
                    
                # Interactive table
                display_reference_table(metrics_df)
                
                # PDF Report
                pdf_report = create_clinical_summary_pdf(metrics_df)
                st.download_button(
                   "📄 Download Full PDF Report",
                   pdf_report,
                   "clinical_report.pdf",
                   "application/pdf"
                )
                
                # Existing CSV download
                download_metrics(parsed_data)
if __name__ == '__main__':
    main()
