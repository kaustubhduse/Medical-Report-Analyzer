import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
import tempfile

# Use the main parse_pdf function which can handle different pipelines
from od_parse import parse_pdf, convert_to_markdown

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

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


# --- MODIFIED FUNCTION to accept advanced options ---
def get_pdf_text(pdf_docs, pipeline_type="default", use_deep_learning=False):
    """
    Parses uploaded PDF documents using od-parse with user-selected options
    and returns their content as a single Markdown formatted string.
    """
    full_text = ""
    st.info(f"Running '{pipeline_type}' pipeline...")
    if use_deep_learning:
        st.warning("Deep Learning is enabled. Processing will be slower but more accurate. 🧠")

    for pdf in pdf_docs:
        # od-parse needs a file path, so we save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf.getvalue())
            tmp_file_path = tmp_file.name

        try:
            # Step 1: Parse the PDF using od-parse with selected advanced options
            parsed_data = parse_pdf(
                file_path=tmp_file_path,
                pipeline_type=pipeline_type,
                use_deep_learning=use_deep_learning
            )

            # Step 2: Convert the parsed data to a Markdown string
            markdown_text = convert_to_markdown(
                parsed_data,
                include_images=False,
                include_tables=True,
                include_forms=True,
                include_handwritten=True
            )
            full_text += markdown_text + "\n\n---\n\n"  # Add separator between docs
        except Exception as e:
            st.error(f"⚠️ Error parsing {pdf.name} with od-parse: {e}")
        finally:
            # Clean up the temporary file
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
        summary = ll.predict(summary_prompt)
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
        
        # --- NEW: Advanced Parsing Options ---
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
        # --- END NEW ---

        if st.button("🚀 Process"):
            with st.spinner("⏳ Processing... This may take a moment."):
                if not pdf_docs:
                    st.error("⚠️ Please upload at least one PDF file!")
                    return

                # --- UPDATED to pass options to the parsing function ---
                raw_text = get_pdf_text(pdf_docs, pipeline_type, use_deep_learning)
                
                if not raw_text:
                    return

                st.session_state.pdf_text = raw_text

                # Generate summary
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

                # The rest of the app logic remains the same...
                predictor = DiseasePredictor()
                metrics_dict = {item['metric']: item['value'] for item in parsed_data}
                risk_assessment = predictor.predict_risk(metrics_dict)

                st.subheader("🩺 Disease Risk Assessment")
                if 'anemia' in risk_assessment:
                    st.progress(risk_assessment['anemia']['probability'])
                    st.markdown(risk_assessment['anemia']['advice'])

                comparator = ReportComparator(st.session_state.vectorstore)
                embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
                text_embedding = embedding_model.embed_documents([raw_text])[0]
                similar_reports = comparator.find_similar_reports(text_embedding)

                if similar_reports:
                    st.subheader("🔍 Similar Reports Found")
                    for report, similarity in similar_reports:
                        st.write(f"**{similarity:.1%} match**: {report['diagnosis']}")

                if len(pdf_docs) > 1:
                    def process_multiple_reports(pdf_docs):
                        historical_data = []
                        for pdf in pdf_docs:
                            text = get_pdf_text([pdf], pipeline_type, use_deep_learning)
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
                        historical_df = pd.DataFrame(historical_data)
                        if "value" in historical_df.columns:
                            historical_df["value"] = pd.to_numeric(historical_df["value"], errors="coerce")
                        if "date" in historical_df.columns:
                            historical_df["date"] = historical_df["date"].astype(str)

                display_metric_summary(parsed_data)
                predict_conditions(parsed_data)

                st.subheader("📈 Interactive Visual Analysis")
                col1, col2 = st.columns(2)
                with col1:
                    plot_metric_comparison(metrics_df)
                with col2:
                    generate_radial_health_score(metrics_df)

                display_reference_table(metrics_df)

                pdf_report = create_clinical_summary_pdf(metrics_df)
                st.download_button(
                   "📄 Download Full PDF Report",
                   pdf_report,
                   "clinical_report.pdf",
                   "application/pdf"
                )
                download_metrics(parsed_data)

if __name__ == '__main__':
    main()