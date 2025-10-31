import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
import tempfile

from od_parse import parse_pdf, convert_to_markdown

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain_core.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, AIMessage

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

load_dotenv()


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
    
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
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
    """
    Processes user question, gets response, and updates chat history in session state.
    """
    if "conversation" in st.session_state and st.session_state.conversation:
        with st.spinner("Thinking..."):
            response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']
    else:
        st.warning("⚠️ No conversation started yet! Upload and process PDFs first.")


def main():
    st.set_page_config(page_title="Medical Report Analyzer", page_icon="⚕️", layout="wide")

    session_keys = [
        "conversation", "chat_history", "summary", "metrics_df", 
        "risk_assessment", "similar_reports", "pdf_report_bytes"
    ]
    for key in session_keys:
        if key not in st.session_state:
            st.session_state[key] = None

    with st.sidebar:
        st.image("https://i.imgur.com/g0Dr1w1.png", width=100) # Placeholder logo
        st.title("👨‍⚕️ Report Analyzer")
        st.markdown("---")
        
        st.subheader("📄 Upload Your Reports")
        pdf_docs = st.file_uploader(
            "Upload your PDF medical reports and click 'Process'", 
            accept_multiple_files=True,
            type="pdf"
        )
        
        st.markdown("---")
        st.subheader("⚙️ Advanced Options")
        pipeline_type = st.selectbox(
            "Parsing Pipeline:",
            options=["default", "forms", "structure", "full"],
            index=3, # Default to 'full' for best results
            help="Choose the parsing method. 'Full' is the most comprehensive and recommended."
        )
        use_deep_learning = st.checkbox(
            "Enable Deep Learning",
            value=False,
            help="Slower, but more accurate for complex tables and layouts. Requires powerful hardware."
        )
        st.markdown("---")

        if st.button("🚀 Process Reports", type="primary"):
            if not pdf_docs:
                st.error("⚠️ Please upload at least one PDF file!")
            else:
                with st.spinner("Analyzing documents... This might take a moment."):
                    # Step 1: Extract text from PDFs
                    raw_text = get_pdf_text(pdf_docs, pipeline_type, use_deep_learning)
                    if not raw_text:
                        return

                    # Step 2: Summarize and extract metrics with LLM
                    summary = summarize_text(raw_text)
                    if not summary:
                        return
                    st.session_state.summary = summary
                    
                    # Save summary for external use
                    summary_path = os.path.join("client", "client-side", "public", "summary.txt")
                    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
                    with open(summary_path, "w", encoding="utf-8") as f:
                        f.write(str(summary))
                        
                    # Step 3: Parse summary to get structured data
                    parsed_data = parse_llm_summary(summary)
                    st.session_state.metrics_df = pd.DataFrame(parsed_data)

                    # Step 4: Create vector store and conversation chain for chat
                    text_chunks = get_text_chunks(raw_text)
                    if not text_chunks:
                        return
                        
                    try:
                        vectorstore = get_vectorstore(text_chunks)
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                        st.session_state.chat_history = []
                    except ValueError as e:
                        st.error(f"❌ Error creating vector store: {e}")
                        return

                    # Step 5: Run advanced analysis
                    predictor = DiseasePredictor()
                    metrics_dict = {item['metric']: item['value'] for item in parsed_data}
                    st.session_state.risk_assessment = predictor.predict_risk(metrics_dict)
                    
                    comparator = ReportComparator(vectorstore)
                    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
                    text_embedding = embedding_model.embed_documents([raw_text])[0]
                    st.session_state.similar_reports = comparator.find_similar_reports(text_embedding)

                    # Step 6: Generate downloadable PDF report
                    pdf_report_bytes = create_clinical_summary_pdf(st.session_state.metrics_df)
                    st.session_state.pdf_report_bytes = pdf_report_bytes
                    
                    st.success("✅ Analysis complete! View the results in the tabs.")

    st.title("⚕️ Interactive Medical Report Dashboard")
    st.markdown("Welcome! Upload your medical reports via the sidebar to unlock insights about your health data.")

    if not st.session_state.conversation:
        st.info("⬆️ Upload your reports and click the **'Process Reports'** button in the sidebar to begin.")
        st.image("https://i.imgur.com/A6f5p6H.png", use_column_width=True) # Placeholder image
    else:
        tab_chat, tab_summary, tab_visuals, tab_advanced = st.tabs([
            "💬 **Chat with Report**", 
            "📄 **AI Summary & Metrics**", 
            "📊 **Visual Analysis**", 
            "🔬 **Advanced Insights**"
        ])

        # --- Chat Tab ---
        with tab_chat:
            st.header("Ask Questions About Your Report")
            st.markdown("Use this chat to ask specific questions about the content of your uploaded document(s).")
            
            if st.session_state.chat_history:
                for message in st.session_state.chat_history:
                    role = "user" if isinstance(message, HumanMessage) else "assistant"
                    with st.chat_message(role):
                        st.markdown(message.content)

            if user_question := st.chat_input("e.g., What was my white blood cell count?"):
                handle_userinput(user_question)
                st.rerun() # Rerun to display the latest chat message

        # --- Summary & Metrics Tab ---
        with tab_summary:
            st.header("AI-Generated Summary")
            if st.session_state.summary:
                full_summary_string = str(st.session_state.summary)

                json_metrics = parse_llm_summary(full_summary_string)

                summary_text = full_summary_string
                json_start_index = full_summary_string.find('[')
                if json_start_index != -1:
                    summary_text = full_summary_string[:json_start_index].strip()
                
                st.markdown(summary_text)

                st.download_button(
                    "📥 Download Text Summary",
                    data=full_summary_string.encode('utf-8'),
                    file_name="medical_summary.txt",
                    mime="text/plain"
                )
                
                st.header("Extracted Health Metrics")
                st.dataframe(st.session_state.metrics_df) 
                
                download_metrics(json_metrics)
            else:
                st.warning("No summary available.")

        with tab_visuals:
            st.header("Visual Analysis of Key Metrics")

            if st.session_state.metrics_df is not None and not st.session_state.metrics_df.empty:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Metric Comparison")
                    plot_metric_comparison(st.session_state.metrics_df)
                with col2:
                    st.subheader("Overall Health Score")
                    generate_radial_health_score(st.session_state.metrics_df)

                st.subheader("Reference Ranges")
                display_reference_table(st.session_state.metrics_df)
                
                if st.session_state.pdf_report_bytes:
                    st.download_button(
                        "📄 Download Full PDF Report",
                        data=st.session_state.pdf_report_bytes,
                        file_name="clinical_summary_report.pdf",
                        mime="application/pdf"
                    )
            else:
                st.warning("No metrics data available to generate visuals.")
                
        with tab_advanced:
            st.header("Advanced Health Insights")

            st.subheader("🩺 Disease Risk Assessment")
            if st.session_state.risk_assessment:
                 if 'anemia' in st.session_state.risk_assessment:
                    risk_info = st.session_state.risk_assessment['anemia']
                    st.write(f"**Anemia Risk Probability:** {risk_info['probability']:.0%}")
                    st.progress(risk_info['probability'])
                    with st.expander("Show Advice"):
                        st.markdown(risk_info['advice'])
            else:
                st.info("No risk assessment data available.")
                
            st.markdown("---")
            
            st.subheader("🔍 Similar Reports")
            if st.session_state.similar_reports:
                st.write("Found reports with similar profiles in our anonymized database:")
                for report, similarity in st.session_state.similar_reports:
                    st.success(f"**{similarity:.1%} match**: Related to '{report['diagnosis']}'")
            else:
                st.info("No similar reports were found.")

if __name__ == '__main__':
    main()
