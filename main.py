# streamlit run app.py
# Optional deps (recommended): sentence-transformers, spacy, pymupdf
# pip install streamlit sentence-transformers spacy pymupdf && python -m spacy download en_core_web_sm
'''
To run your own Streamlit app, first save your Python script (e.g., my_app.py). Then, run the following 
command in your terminal from the same directory: 
                            streamlit run my_app.py --> will have to name the file 'my_app.py'

Run the activation script. Assuming your virtual environment is named venv and is in your project's root 
folder, run this command:
                            source venv/bin/activate --> env name is 'venv'
'''
import os
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from modules import llm_registry
from modules.functions import load_spacy_model, read_pdf_text
from modules.magent import ManualOnlyAgent
from modules.message_db_functions import save_conversation, get_session_list, load_conversations

# -------------------------
# Initialization / Globals
# -------------------------
# Load cloud API keys from env file (OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY)
load_dotenv("files/keys.env")

# Embedding + NLP components (loaded once)
EMBEDDER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
NLP = load_spacy_model("en_core_web_sm")

# ------- STREAMLIT --------
st.set_page_config(page_title="Manual QA Chatbot", page_icon="💬", layout="wide")
st.title("📘 Manual QA Chatbot (Multi-Provider: OpenAI · Claude · Gemini)")

# Initialize session ID
if "session_id" not in st.session_state:
    st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
if "chat" not in st.session_state:
    st.session_state.chat = []
if "manual_text" not in st.session_state:
    st.session_state.manual_text = ""
if "agent_key" not in st.session_state:
    st.session_state.agent_key = 0

# Sidebar
with st.sidebar:
    st.title("Manual Source")
    uploaded = st.file_uploader("Upload manual (.txt/.md/.pdf)", type=["txt", "md", "pdf"])
    manual_text = ""

    # Source input - handle upload separately
    if uploaded is not None:
        if uploaded.type == "application/pdf":
            with st.spinner("Extracting text from PDF..."):
                extracted_text = read_pdf_text(uploaded) or ""
                if extracted_text:
                    st.session_state.manual_text = extracted_text
                    st.success(f"✅ PDF loaded ({len(extracted_text)} characters)")
        else:
            extracted_text = uploaded.read().decode("utf-8", errors="ignore")
            if extracted_text:
                st.session_state.manual_text = extracted_text
                st.success(f"✅ File loaded ({len(extracted_text)} characters)")

    st.write("**Or paste manual text:**")
    pasted_text = st.text_area(
        "Paste here (Markdown or plain text)",
        value="" if uploaded is not None else st.session_state.manual_text,
        height=200,
        help="Paste text here OR upload a file above"
    )

    # Only update from text area if no file is uploaded
    if uploaded is None and pasted_text:
        st.session_state.manual_text = pasted_text

    manual_text = st.session_state.manual_text

    # -------------------------
    # Sidebar: QA Settings
    # -------------------------
    st.subheader("Settings")
    low_conf = st.slider(
        "Clarify if confidence below", 0.0, 1.0, 0.18, 0.01, key="slider_conf"
    )
    word_cap = st.slider(
        "Answer word cap", 40, 600, 150, 10, key="slider_wordcap"
    )
    use_refiner = st.checkbox(
        "Use factual LLM refinement", value=True, key="checkbox_refiner"
    )

    # -------------------------
    # Sidebar: LLM Settings (Cloud providers)
    # -------------------------
    st.sidebar.subheader("LLM Settings")

    # Build / refresh agent when manual changes
    if "agent_key" not in st.session_state:
        st.session_state.agent_key = 0

    providers = llm_registry.available_providers()  # ["OpenAI", "Anthropic", "Gemini"]
    # Default to OpenAI for the demo if present
    default_provider_index = 0 if "OpenAI" in providers else 0
    provider = st.selectbox("Provider", providers, index=default_provider_index)

    model_names = llm_registry.models_for(provider)
    default_index = 0  # registry is ordered with sensible defaults first
    model_name = st.selectbox("Model", model_names, index=default_index)

    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    # Keep a generous upper bound; llm_registry will clip per model
    max_tokens = st.number_input(
        "Max output tokens", min_value=64, max_value=8192, value=512, step=64
    )

    # Persist current LLM settings for downstream use (agent reads from session)
    st.session_state["llm_settings"] = {
        "provider": provider,
        "model": model_name,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "use_refiner": bool(use_refiner),
    }

    agent = ManualOnlyAgent(
        st.session_state.manual_text,
        low_conf=low_conf,
        word_cap=word_cap,
        embedder=EMBEDDER,
        nlp=NLP,
    )

    # Manual refresh button (forces agent re-init if you key the cache by agent_key)
    if st.sidebar.button("🔄 Load / Refresh Manual", type="primary"):
        st.session_state.agent_key += 1
        st.sidebar.success("✅ Manual refreshed!")

    if not st.session_state.manual_text.strip():
        st.info("👈 Upload or paste your manual on the left, then click **Load / Refresh Manual**.")
        st.stop()

    st.divider()

    # Conversation history section
    st.subheader("💬 Conversation History")

    sessions = get_session_list()
    if sessions:
        st.caption(f"Found {len(sessions)} previous session(s)")

        # Show sessions
        for sid, first_msg, count in sessions[:5]:  # Show last 5 sessions
            is_current = sid == st.session_state.session_id
            prefix = "▶️ " if is_current else ""
            if st.button(f"{prefix}{first_msg}... ({count} msgs)", key=f"load_{sid}"):
                # Load this session
                if not is_current:
                    st.session_state.session_id = sid
                    st.session_state.chat = []

                    # Load conversations into chat
                    convs = load_conversations(sid)
                    for conv in convs:
                        st.session_state.chat.append(("user", conv["user_message"]))

                        # Recreate assistant message
                        if conv.get("takeaways"):
                            takeaway_text = "\n".join([f"{i}. {t}" for i, t in enumerate(conv["takeaways"], 1)])
                            assistant_msg = f"🎯 **Key Takeaways:**\n\n{takeaway_text}"
                        else:
                            assistant_msg = conv.get("answer", "")[:200] + "..."

                        st.session_state.chat.append(("assistant", assistant_msg))

                    st.rerun()

        if len(sessions) > 5:
            st.caption(f"...and {len(sessions) - 5} more")
    else:
        st.caption("No previous conversations yet")

    # TODO: Add remove/load convo button and allow loading of new/old conversations
    # New conversation button
    if st.button("🆕 New Conversation"):
        st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.session_state.chat = []
        st.rerun()

    st.divider()
    st.caption("💡 **Tips:**")
    st.caption("• Be specific in your questions")
    st.caption("• Check Key Takeaways for quick info")
    st.caption("• Expand Full Answer for details")

    if st.session_state.chat:
        if st.button("🗑️ Clear Current Chat"):
            st.session_state.chat = []
            st.rerun()

agent = ManualOnlyAgent(st.session_state.manual_text, low_conf=low_conf, word_cap=word_cap, embedder=EMBEDDER, nlp=NLP)

# Chat history (UI)
for role, msg in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(msg)

user_msg = st.chat_input("Ask a question about the manual…")
if user_msg:
    st.session_state.chat.append(("user", user_msg))
    with st.chat_message("user"):
        st.markdown(user_msg)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                resp = agent.respond(user_msg)
            except Exception as e:
                st.error(f"Error while generating answer: {e}")
                st.stop()

        # Save to persistent storage
        save_conversation(user_msg, resp, st.session_state.session_id)

        # Safely read fields with defaults
        intent = resp.get("intent", "unknown")
        confidence = float(resp.get("confidence", 0.0) or 0.0)
        conf_label = resp.get("confidence_label", "Confidence")
        answer = resp.get("answer", "(No answer)")
        sources = resp.get("sources", [])
        takeaways = resp.get("takeaways")
        follow_up = resp.get("follow_up_question", None)

        provider_label = st.session_state["llm_settings"]["provider"]
        model_label = st.session_state["llm_settings"]["model"]

        # Header line with intent + confidence
        col1, col2 = st.columns([2, 1])
        with col1:
            intent_icons = {
                "eligibility": "🎯", "procedure": "📋", "coverage": "🛡️",
                "documentation": "📄", "contact": "📞", "definition": "📖", "other": "❓"
            }

            icon = intent_icons.get(intent, "💬")
            st.markdown(f"*Model:* `{provider_label} / {model_label}` · ")
            st.markdown(f"{icon} **Intent:** `{intent}` · **{conf_label}**")

        with col2:
            conf = confidence
            conf_color = "🟢" if conf >= 0.7 else "🟡" if conf >= 0.4 else "🔴"
            st.markdown(f"{conf_color} **Confidence:** `{conf:.2f}`")

        st.divider()

        # KEY TAKEAWAYS - The star of the show!
        if takeaways and len(takeaways) > 0:
            st.markdown("### 🎯 Key Takeaways")
            for i, takeaway in enumerate(takeaways, 1):
                st.markdown(f"**{i}.** {takeaway}")
            st.divider()

        # Full answer - collapsed by default
        with st.expander("📄 Full Answer", expanded=False):
            st.markdown(answer)

        # Sources expander
        with st.expander(f"📖 View {len(sources)} source(s)", expanded=False):
            for i, s in enumerate(sources, 1):
                st.markdown(f"**{i}. {s['section']}**")
                st.markdown(f"> {s['excerpt']}")
                score = s['score']
                if score <= 1.0:
                    bar_width = int(score * 20)
                    st.caption(f"Relevance: {'█' * bar_width}{'░' * (20 - bar_width)} {score:.0%}")
                else:
                    st.caption(f"Keyword matches: {int(score)}")
                if i < len(resp["sources"]):
                    st.divider()

        # Log in UI history - show takeaways in chat
        icon = intent_icons.get(intent, "💬")
        if resp.get("takeaways"):
            takeaway_text = "\n".join([f"{i}. {t}" for i, t in enumerate(takeaways, 1)])
            combined = f"{icon} **Key Takeaways:**\n\n{takeaway_text}"
        else:
            combined = f"{icon} {answer[:200]}..."

        # Follow-up suggestion
        if follow_up:
            st.info(follow_up)

        st.session_state.chat.append(("assistant", combined))
