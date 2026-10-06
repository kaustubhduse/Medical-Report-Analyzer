import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
import tempfile

from od_parse import parse_pdf, convert_to_markdown

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.messages import HumanMessage, AIMessage

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


def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
    .stApp                { background: #0d1117 !important; color: #e2e8f0 !important; }
    .main .block-container{ background: #0d1117 !important; }

    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }

    /* Header transparent so toggle button stays visible */
    header[data-testid="stHeader"] {
        background: rgba(13,17,23,0) !important;
        box-shadow: none !important;
    }
    header[data-testid="stHeader"] .stDeployButton { display: none !important; }

    /* Sidebar open — collapse arrow */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapseButton"] > button {
        visibility: visible !important; opacity: 1 !important; display: flex !important;
        background: rgba(255,255,255,0.1) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255,255,255,0.18) !important;
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebarCollapseButton"] svg { fill: #e2e8f0 !important; }

    /* Sidebar closed — expand tab on left edge */
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebarCollapsedControl"] > button,
    [data-testid="collapsedControl"],
    [data-testid="collapsedControl"] > button {
        visibility: visible !important; opacity: 1 !important; display: flex !important;
        background: #1d4ed8 !important;
        border-radius: 0 10px 10px 0 !important;
        border: none !important;
        box-shadow: 3px 0 14px rgba(29,78,216,0.5) !important;
        z-index: 9999 !important;
    }
    [data-testid="stSidebarCollapsedControl"] svg,
    [data-testid="collapsedControl"] svg { fill: white !important; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1437 0%, #0e2050 60%, #102766 100%) !important;
        border-right: 1px solid rgba(99,179,237,0.15) !important;
    }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #e2e8f0 !important; }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background: rgba(255,255,255,0.05) !important;
        border: 2px dashed rgba(99,179,237,0.4) !important;
        border-radius: 12px !important;
    }
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background: rgba(255,255,255,0.07) !important;
        border: 1px solid rgba(99,179,237,0.3) !important;
        color: #e2e8f0 !important;
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] .stCheckbox label { color: #cbd5e1 !important; }
    [data-testid="stSidebar"] hr { border-color: rgba(99,179,237,0.2) !important; }

    /* Process button */
    [data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
        color: white !important; border: none !important;
        border-radius: 12px !important; font-weight: 700 !important;
        font-size: 15px !important; width: 100% !important;
        padding: 0.65rem 1.2rem !important;
        box-shadow: 0 4px 15px rgba(59,130,246,0.4) !important;
        transition: all 0.25s ease !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(59,130,246,0.6) !important;
    }

    /* Sidebar brand block */
    .sidebar-brand {
        display: flex; flex-direction: column; align-items: center;
        padding: 1.2rem 0 0.8rem;
        border-bottom: 1px solid rgba(99,179,237,0.2);
        margin-bottom: 1rem;
    }
    .sidebar-brand .brand-title { font-size: 1.1rem; font-weight: 700; color: #e2e8f0; margin-top: 0.5rem; }
    .sidebar-brand .brand-sub   { font-size: 0.72rem; color: #94a3b8; margin-top: 2px; }

    /* Page header banner */
    .page-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #1d4ed8 50%, #0ea5e9 100%);
        border-radius: 20px; padding: 2rem 2.5rem; margin-bottom: 1.5rem;
    }
    .page-header h1 { color: white !important; font-size: 2rem !important; font-weight: 800 !important; margin: 0 !important; }
    .page-header p  { color: rgba(255,255,255,0.85) !important; margin: 0.4rem 0 0 !important; font-size: 1rem; }

    /* Feature cards */
    .feature-grid {
        display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: 1.5rem 0;
    }
    @media (max-width: 900px) { .feature-grid { grid-template-columns: repeat(2, 1fr); } }
    .feature-card {
        background: #161b27; border-radius: 16px; padding: 1.4rem 1.2rem; text-align: center;
        box-shadow: 0 2px 16px rgba(0,0,0,0.4);
        border: 1px solid rgba(255,255,255,0.06); border-top-width: 3px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .feature-card:hover { transform: translateY(-3px); box-shadow: 0 8px 28px rgba(0,0,0,0.5); }
    .feature-card.blue  { border-top-color: #3b82f6; }
    .feature-card.teal  { border-top-color: #0ea5e9; }
    .feature-card.green { border-top-color: #10b981; }
    .feature-card.purple{ border-top-color: #8b5cf6; }
    .feature-card .icon { font-size: 2.2rem; margin-bottom: 0.6rem; }
    .feature-card h4 { font-size: 0.95rem; font-weight: 700; color: #e2e8f0; margin: 0 0 0.3rem; }
    .feature-card p  { font-size: 0.8rem; color: #94a3b8; margin: 0; line-height: 1.4; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #161b27 !important; border-radius: 12px 12px 0 0 !important;
        padding: 0.3rem 0.3rem 0 !important; gap: 4px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0 !important; font-weight: 500 !important;
        font-size: 0.9rem !important; padding: 0.6rem 1.2rem !important;
        color: #94a3b8 !important; background: transparent !important;
    }
    .stTabs [aria-selected="true"] {
        background: #1e293b !important; color: #60a5fa !important;
        font-weight: 700 !important; border-bottom: 3px solid #3b82f6 !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        background: #1e293b !important; border-radius: 0 0 16px 16px !important;
        padding: 1.5rem !important; box-shadow: 0 4px 16px rgba(0,0,0,0.3) !important;
    }

    /* Section headers */
    .section-header {
        display: flex; align-items: center; gap: 10px;
        padding-bottom: 0.75rem; border-bottom: 2px solid rgba(99,179,237,0.2); margin-bottom: 1.2rem;
    }
    .section-header h2 { font-size: 1.25rem !important; font-weight: 700 !important; color: #93c5fd !important; margin: 0 !important; }

    /* Chat */
    [data-testid="stChatMessage"] {
        border-radius: 14px !important; padding: 0.8rem 1rem !important;
        margin-bottom: 0.5rem !important; background: #0f172a !important;
        border: 1px solid rgba(99,179,237,0.15) !important;
    }
    .stChatInput textarea {
        background: #161b27 !important; color: #e2e8f0 !important;
        border-radius: 12px !important; border: 2px solid rgba(59,130,246,0.3) !important;
    }
    .stChatInput textarea:focus { border-color: #3b82f6 !important; box-shadow: 0 0 0 3px rgba(59,130,246,0.2) !important; }

    /* Download buttons */
    .stDownloadButton > button {
        border-radius: 10px !important; font-weight: 600 !important;
        border: 2px solid #3b82f6 !important; color: #60a5fa !important;
        background: rgba(59,130,246,0.1) !important; transition: all 0.2s !important;
    }
    .stDownloadButton > button:hover { background: rgba(59,130,246,0.2) !important; }

    /* DataFrames */
    [data-testid="stDataFrame"] {
        border-radius: 12px !important; overflow: hidden !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.4) !important;
    }

    /* Progress */
    .stProgress > div { background: rgba(255,255,255,0.08) !important; border-radius: 99px !important; }
    .stProgress > div > div > div { border-radius: 99px !important; }

    /* Alerts */
    [data-testid="stAlert"] { border-radius: 12px !important; }

    /* Expander */
    .streamlit-expanderHeader {
        border-radius: 10px !important; font-weight: 600 !important;
        background: #161b27 !important; color: #e2e8f0 !important;
    }
    .streamlit-expanderContent { background: #0f172a !important; }

    /* Text */
    .stMarkdown p, .stMarkdown li { color: #cbd5e1 !important; }
    h1, h2, h3, h4 { color: #e2e8f0 !important; }
    .stCaption { color: #64748b !important; }
    </style>
    """, unsafe_allow_html=True)


def get_pdf_text(pdf_docs, pipeline_type="default", use_deep_learning=False):
    full_text = ""
    st.info(f"Running '{pipeline_type}' pipeline...")
    if use_deep_learning:
        st.warning("Deep Learning enabled — processing will be slower. 🧠")

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
            st.error(f"⚠️ Error parsing {pdf.name}: {e}")
        finally:
            os.remove(tmp_file_path)

    if not full_text.strip():
        st.error("⚠️ No readable text found in uploaded PDFs.")
        return None
    return full_text


def summarize_text(text):
    try:
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            st.error("❌ TOGETHER_API_KEY missing. Add it to Streamlit Secrets or .env.")
            return None

        llm = ChatOpenAI(
            base_url="https://api.together.xyz/v1",
            api_key=api_key,
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        )
        summary_prompt = (
            "You are a medical expert assistant. Carefully read and summarize the following medical report in Markdown format. "
            "Include: patient name, date, medical history, key findings, diagnoses, and recommendations as bullet points.\n"
            "Also extract medical metrics as a JSON array with fields: metric, value, reference_range, unit.\n"
            "Return metrics only in JSON format and other info in plain text.\n"
            f"{text}"
        )
        return llm.invoke(summary_prompt).content
    except Exception as e:
        st.error(f"❌ Error generating summary: {e}")
        return None


def get_text_chunks(text):
    chunks = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    ).split_text(text)
    if not chunks:
        st.error("⚠️ No valid text chunks found.")
        return None
    return chunks


def get_vectorstore(text_chunks):
    if not text_chunks:
        raise ValueError("No text chunks provided for FAISS indexing.")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)


def get_conversation_chain(vectorstore):
    try:
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            st.error("❌ TOGETHER_API_KEY missing.")
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
        st.error(f"❌ Error initializing chat: {e}")
        return None


def handle_userinput(user_question):
    if "conversation" in st.session_state and st.session_state.conversation:
        with st.spinner("Thinking..."):
            response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']
    else:
        st.warning("⚠️ Upload and process a report first.")


def render_landing():
    st.markdown("""
    <div class="page-header">
        <h1>⚕️ Medical Report Analyzer</h1>
        <p>AI-powered analysis of your medical reports — summaries, metrics, charts, and risk insights in seconds.</p>
    </div>
    <div class="feature-grid">
        <div class="feature-card blue">
            <div class="icon">🤖</div>
            <h4>AI Summary</h4>
            <p>LLM-generated plain-language summaries of complex reports</p>
        </div>
        <div class="feature-card teal">
            <div class="icon">📊</div>
            <h4>Visual Analytics</h4>
            <p>Interactive charts showing your metrics vs. reference ranges</p>
        </div>
        <div class="feature-card green">
            <div class="icon">💬</div>
            <h4>Chat with Report</h4>
            <p>Ask any question and get instant answers from your report</p>
        </div>
        <div class="feature-card purple">
            <div class="icon">🔬</div>
            <h4>Risk Assessment</h4>
            <p>ML-powered disease risk scores based on your lab values</p>
        </div>
    </div>
    <div style="background:#161b27;border-radius:16px;padding:1.5rem 2rem;
                box-shadow:0 2px 16px rgba(0,0,0,0.4);border-left:4px solid #3b82f6;margin-top:0.5rem;">
        <b style="color:#60a5fa;">👈 Get started:</b>
        <span style="color:#94a3b8;"> Upload a PDF medical report in the sidebar, then click
        <b style="color:#e2e8f0;">Process Reports</b> to unlock all features.</span>
    </div>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="Medical Report Analyzer",
        page_icon="⚕️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    inject_css()

    for key in ["conversation", "chat_history", "summary", "metrics_df",
                "risk_assessment", "similar_reports", "pdf_report_bytes"]:
        if key not in st.session_state:
            st.session_state[key] = None

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand">
            <div style="font-size:2.4rem;">⚕️</div>
            <div class="brand-title">MedReport AI</div>
            <div class="brand-sub">Powered by Mixtral + LangChain</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p style="color:#94a3b8;font-size:0.78rem;font-weight:600;letter-spacing:0.8px;margin-bottom:6px;">STEP 1 — UPLOAD</p>', unsafe_allow_html=True)
        pdf_docs = st.file_uploader(
            "Drop PDFs here", accept_multiple_files=True,
            type="pdf", label_visibility="collapsed"
        )
        if pdf_docs:
            st.markdown(f'<p style="color:#34d399;font-size:0.82rem;">✓ {len(pdf_docs)} file(s) ready</p>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p style="color:#94a3b8;font-size:0.78rem;font-weight:600;letter-spacing:0.8px;margin-bottom:6px;">STEP 2 — OPTIONS</p>', unsafe_allow_html=True)
        pipeline_type = st.selectbox(
            "Parsing Pipeline", options=["default", "forms", "structure", "full"],
            index=3, help="'Full' is the most comprehensive and recommended."
        )
        use_deep_learning = st.checkbox(
            "Enable Deep Learning (slower, more accurate)", value=False,
            help="Not recommended on Streamlit free tier."
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p style="color:#94a3b8;font-size:0.78rem;font-weight:600;letter-spacing:0.8px;margin-bottom:6px;">STEP 3 — ANALYZE</p>', unsafe_allow_html=True)

        if st.button("🚀 Process Reports", type="primary"):
            if not pdf_docs:
                st.error("Please upload at least one PDF first.")
            else:
                with st.spinner("Analyzing... this may take a minute."):
                    raw_text = get_pdf_text(pdf_docs, pipeline_type, use_deep_learning)
                    if not raw_text:
                        st.stop()

                    summary = summarize_text(raw_text)
                    if not summary:
                        st.stop()
                    st.session_state.summary = summary

                    try:
                        summary_path = os.path.join("client", "client-side", "public", "summary.txt")
                        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
                        with open(summary_path, "w", encoding="utf-8") as f:
                            f.write(str(summary))
                    except Exception:
                        pass

                    parsed_data = parse_llm_summary(summary)
                    st.session_state.metrics_df = pd.DataFrame(parsed_data)

                    text_chunks = get_text_chunks(raw_text)
                    if not text_chunks:
                        st.stop()

                    try:
                        vectorstore = get_vectorstore(text_chunks)
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                        st.session_state.chat_history = []
                    except ValueError as e:
                        st.error(f"❌ Vector store error: {e}")
                        st.stop()

                    predictor = DiseasePredictor()
                    metrics_dict = {item['metric']: item['value'] for item in parsed_data}
                    st.session_state.risk_assessment = predictor.predict_risk(metrics_dict)

                    comparator = ReportComparator(vectorstore)
                    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
                    text_embedding = embedding_model.embed_documents([raw_text])[0]
                    st.session_state.similar_reports = comparator.find_similar_reports(text_embedding)

                    st.session_state.pdf_report_bytes = create_clinical_summary_pdf(st.session_state.metrics_df)
                    st.success("✅ Analysis complete!")

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="border-top:1px solid rgba(99,179,237,0.2);padding-top:1rem;">
            <p style="color:#475569;font-size:0.72rem;text-align:center;margin:0;">
                🔒 Your data is processed locally.<br>Reports are never stored.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── Main Content ─────────────────────────────────────────────────────────
    if not st.session_state.conversation:
        render_landing()
    else:
        st.markdown("""
        <div class="page-header">
            <h1>⚕️ Medical Report Dashboard</h1>
            <p>Analysis complete — explore your results across the tabs below.</p>
        </div>
        """, unsafe_allow_html=True)

        tab_chat, tab_summary, tab_visuals, tab_advanced = st.tabs([
            "💬  Chat with Report",
            "📄  AI Summary & Metrics",
            "📊  Visual Analysis",
            "🔬  Advanced Insights"
        ])

        with tab_chat:
            st.markdown("""<div class="section-header"><span style="font-size:1.4rem;">💬</span><h2>Ask Questions About Your Report</h2></div>""", unsafe_allow_html=True)
            st.caption("Ask anything — lab values, diagnoses, recommendations.")
            if st.session_state.chat_history:
                for message in st.session_state.chat_history:
                    role = "user" if isinstance(message, HumanMessage) else "assistant"
                    with st.chat_message(role):
                        st.markdown(message.content)
            if user_question := st.chat_input("e.g., What was my hemoglobin level?"):
                handle_userinput(user_question)
                st.rerun()

        with tab_summary:
            st.markdown("""<div class="section-header"><span style="font-size:1.4rem;">📄</span><h2>AI-Generated Summary</h2></div>""", unsafe_allow_html=True)
            if st.session_state.summary:
                full_summary_string = str(st.session_state.summary)
                json_metrics = parse_llm_summary(full_summary_string)
                summary_text = full_summary_string
                idx = full_summary_string.find('[')
                if idx != -1:
                    summary_text = full_summary_string[:idx].strip()
                st.markdown(summary_text)
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button("📥 Download Text Summary", data=full_summary_string.encode('utf-8'),
                                       file_name="medical_summary.txt", mime="text/plain")
                with col2:
                    download_metrics(json_metrics)
                st.markdown("---")
                st.markdown("""<div class="section-header"><span style="font-size:1.4rem;">🧪</span><h2>Extracted Health Metrics</h2></div>""", unsafe_allow_html=True)
                st.dataframe(st.session_state.metrics_df, use_container_width=True, hide_index=True)
            else:
                st.info("No summary available yet.")

        with tab_visuals:
            st.markdown("""<div class="section-header"><span style="font-size:1.4rem;">📊</span><h2>Visual Analysis</h2></div>""", unsafe_allow_html=True)
            if st.session_state.metrics_df is not None and not st.session_state.metrics_df.empty:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Metric Comparison**")
                    plot_metric_comparison(st.session_state.metrics_df)
                with col2:
                    st.markdown("**Overall Health Score**")
                    generate_radial_health_score(st.session_state.metrics_df)
                st.markdown("---")
                display_reference_table(st.session_state.metrics_df)
                if st.session_state.pdf_report_bytes:
                    st.download_button("📄 Download Full PDF Report",
                                       data=st.session_state.pdf_report_bytes,
                                       file_name="clinical_summary_report.pdf",
                                       mime="application/pdf")
            else:
                st.info("No metrics data available — process a report first.")

        with tab_advanced:
            st.markdown("""<div class="section-header"><span style="font-size:1.4rem;">🔬</span><h2>Advanced Health Insights</h2></div>""", unsafe_allow_html=True)
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("#### 🩺 Disease Risk Assessment")
                if st.session_state.risk_assessment and 'anemia' in st.session_state.risk_assessment:
                    risk_info = st.session_state.risk_assessment['anemia']
                    prob = risk_info['probability']
                    color = "#ef4444" if prob > 0.7 else ("#f59e0b" if prob > 0.4 else "#10b981")
                    st.markdown(f"""
                    <div style="background:#0f172a;border-radius:14px;padding:1.2rem;
                                box-shadow:0 2px 10px rgba(0,0,0,0.4);border-left:4px solid {color};">
                        <div style="font-size:0.85rem;color:#64748b;font-weight:600;">ANEMIA RISK</div>
                        <div style="font-size:2rem;font-weight:800;color:{color};">{prob:.0%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(prob)
                    with st.expander("View Clinical Advice"):
                        st.markdown(risk_info['advice'])
                else:
                    st.info("No risk data available.")
            with col_b:
                st.markdown("#### 🔍 Similar Reports")
                if st.session_state.similar_reports:
                    for report, similarity in st.session_state.similar_reports:
                        st.success(f"**{similarity:.1%} match** — Related to '{report['diagnosis']}'")
                else:
                    st.info("No similar reports found.")


if __name__ == '__main__':
    main()
