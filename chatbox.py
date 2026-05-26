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

    Marcelle's version
'''
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st

# -------------------- Optional backends --------------------
@st.cache_resource(show_spinner=False)
def load_embedder():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def load_spacy():
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except Exception:
        return None

EMBEDDER = load_embedder()
NLP = load_spacy()

# -------------------- Small utils --------------------
def _clean(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()

def _split_sents(txt: str) -> List[str]:
    out, cur = [], []
    for part in re.split(r"([.?!])", txt):
        if part is None:
            continue
        cur.append(part)
        if part in (".", "?", "!"):
            s = _clean("".join(cur))
            if s:
                out.append(s)
            cur = []
    tail = _clean("".join(cur))
    if tail:
        out.append(tail)
    return out

# -------------------- Data classes --------------------
@dataclass
class Turn:
    user: str
    intent: str
    entities: Dict[str, str]
    confidence: float
    answer: str
    sources: List[Dict[str, Any]]

@dataclass
class MessageHistory:
    turns: List[Turn] = field(default_factory=list)
    def add(self, **kw): self.turns.append(Turn(**kw))

@dataclass
class Chunk:
    section: str
    text: str
    start: int
    end: int

# -------------------- Interpreter --------------------
class NLPInterpreter:
    INTENTS = ["eligibility","procedure","coverage","documentation","contact","definition","other"]

    def __init__(self, embedder=None, nlp=None):
        self.embedder = embedder
        self.nlp = nlp
        if embedder:
            from sentence_transformers import util  # noqa: F401
            self.label_emb = embedder.encode(self.INTENTS, convert_to_tensor=True)
        else:
            self.label_emb = None

    def _route(self, q: str) -> str:
        ql = q.lower()
        if any(w in ql for w in ["eligib","qualif"]): return "eligibility"
        if any(w in ql for w in ["how to","how do","steps","procedure","schedule","apply"]): return "procedure"
        if any(w in ql for w in ["cover","covered","benefit","pay for","not covered"]): return "coverage"
        if any(w in ql for w in ["document","proof","form","deadline"]): return "documentation"
        if any(w in ql for w in ["call","phone","contact","office","broker"]): return "contact"
        if ql.startswith("what is") or " meaning" in ql: return "definition"
        if self.embedder and self.label_emb is not None:
            from sentence_transformers import util
            qv = self.embedder.encode([q], convert_to_tensor=True)
            sims = util.cos_sim(qv, self.label_emb)[0].tolist()
            return self.INTENTS[int(max(range(len(sims)), key=lambda i: sims[i]))]
        return "other"

    def _entities(self, q: str) -> Dict[str, str]:
        ents = {}
        if self.nlp:
            doc = self.nlp(q)
            for e in doc.ents:
                ents[e.label_.lower()] = e.text
            for tok in doc:
                if tok.text.lower() in {"medicaid","nemt","logisticare","voucher","county","broker"}:
                    ents.setdefault("program", tok.text)
        else:
            if re.search(r"\bmedicaid\b", q, re.I): ents["program"] = "Medicaid"
            if re.search(r"\bnemt\b", q, re.I): ents["program"] = "NEMT"
            if re.search(r"logisticare", q, re.I): ents["program"] = "LogistiCare"
            if re.search(r"\bvoucher\b", q, re.I): ents["program"] = "Voucher"
            if re.search(r"\bcounty\b", q, re.I): ents["program"] = "County"
        return ents

    def interpret(self, query: str) -> Tuple[str, Dict[str, str]]:
        return self._route(query), self._entities(query)

# -------------------- Manual DB (chunk + retrieve + answer) --------------------
class ManualDB:
    def __init__(self, manual_text: str, max_chars=1200, overlap=160, embedder=None):
        self.text = manual_text
        self.chunks = self._chunk(manual_text, max_chars, overlap)
        self.embedder = embedder
        self.emb = (
            embedder.encode([c.text for c in self.chunks], convert_to_tensor=True)
            if (embedder and self.chunks) else None
        )

    def _chunk(self, manual: str, max_chars: int, overlap: int) -> List[Chunk]:
        lines = manual.splitlines()
        sections, cur, title = [], [], "General"
        for ln in lines:
            if re.match(r"^\s*(#{1,6}\s+.+|[0-9]+\.\s+.+|[A-Z][A-Z0-9 \-]{6,})\s*$", ln):
                if cur: sections.append((title, "\n".join(cur).strip())); cur = []
                title = _clean(re.sub(r"^\s*#{1,6}\s*", "", ln))
            else:
                cur.append(ln)
        if cur: sections.append((title, "\n".join(cur).strip()))
        chunks = []
        for title, body in sections:
            if not body: continue
            i = 0
            while i < len(body):
                j = min(i + max_chars, len(body))
                window = body[i:j]
                if j < len(body):
                    lastp = window.rfind(". ")
                    if lastp > max_chars * 0.5:
                        j = i + lastp + 1
                        window = body[i:j]
                abs_start = self.text.find(window)
                chunks.append(Chunk(title, window, abs_start, abs_start + len(window)))
                if j >= len(body): break
                i = max(0, j - overlap)
        return chunks

    def retrieve(self, query: str, k=5) -> List[Tuple[Chunk, float]]:
        if not self.chunks: return []
        if not (self.embedder and self.emb is not None):
            qtok = set(re.findall(r"\w+", query.lower()))
            scored = [(c, float(len(qtok & set(re.findall(r"\w+", c.text.lower()))))) for c in self.chunks]
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:k]
        from sentence_transformers import util
        qv = self.embedder.encode([query], convert_to_tensor=True)
        cos = util.cos_sim(qv, self.emb)[0]
        vals, idxs = cos.topk(k=min(k, len(self.chunks)))
        return [(self.chunks[i], float(s)) for s, i in zip(vals.tolist(), idxs.tolist())]

    def answer(self, query: str, word_cap=90, k=5) -> Tuple[str, List[Dict[str, Any]]]:
        hits = self.retrieve(query, k=k)
        if not hits:
            return "I couldn't find relevant guidance in the manual.", []
        passages=[]
        for c,_ in hits:
            lines=[l.strip() for l in c.text.splitlines() if l.strip()]
            ruleish=[l for l in lines if re.search(r"\b(eligib|require|must|should|steps?|procedure|covered|not covered|verification|proof|deadline)\b", l, re.I)]
            passages.append("\n".join(ruleish[:8]) if ruleish else c.text)
        sents=[]
        for p in passages:
            sents += [s for s in _split_sents(p) if len(s.split())>=3]
        if not sents:
            sents=[_clean(hits[0][0].text)]
        # rank sentences
        if EMBEDDER:
            from sentence_transformers import util
            qv = EMBEDDER.encode([query], convert_to_tensor=True)
            sv = EMBEDDER.encode(sents, convert_to_tensor=True)
            sims = util.cos_sim(qv, sv)[0].tolist()
            ranked = [x for _,x in sorted(zip(sims, sents), key=lambda t:t[0], reverse=True)]
        else:
            qtok=set(re.findall(r"\w+",query.lower()))
            ranked=[x for _,x in sorted([(len(qtok & set(re.findall(r"\w+",s.lower()))), s) for s in sents], key=lambda t:t[0], reverse=True)]
        # dedupe + cap
        seen, picked = set(), []
        for s in ranked:
            key = re.sub(r"[^a-z0-9]+","", s.lower())[:80]
            if key in seen: continue
            seen.add(key); picked.append(s)
            if len(" ".join(picked).split()) >= word_cap: break
        answer = " ".join(" ".join(picked).split()[:word_cap]).strip()
        sources = [{
            "section": c.section,
            "excerpt": _clean(c.text)[:220] + ("…" if len(c.text) > 220 else ""),
            "location": [c.start, c.end],
            "score": round(s, 4) if s <= 1.0 else round(s, 2)
        } for c, s in hits]
        return answer, sources

# -------------------- Actions (no tree) --------------------
class Actions:
    @staticmethod
    def format_steps(answer_text: str) -> str:
        lines = [l.strip() for l in re.split(r"[\n;]+", answer_text)]
        bullets = [f"- {l}" for l in lines if re.match(r"^(?:[0-9]+\.|Verify|Call|Provide|Schedule|Submit|Check|Confirm)\b", l)]
        return "\n".join(bullets[:6]) if bullets else ""

# -------------------- Agent (manual-only) --------------------
class ManualOnlyAgent:
    def __init__(self, manual_text: str, low_conf=0.18, word_cap=90, embedder=None, nlp=None):
        self.history = MessageHistory()
        self.interpreter = NLPInterpreter(embedder=embedder, nlp=nlp)
        self.db = ManualDB(manual_text, embedder=embedder)
        self.low_conf = low_conf
        self.word_cap = word_cap

    def _conf(self, sources: List[Dict[str, Any]]) -> float:
        if not sources: return 0.0
        mx = max(s.get("score", 0.0) for s in sources)
        return float(mx if mx <= 1.0 else min(1.0, mx/(mx+5.0)))

    def _clarifier(self, intent: str) -> str:
        qs = {
            "eligibility":"Which program should I check (e.g., Medicaid NEMT, county vouchers, travel training)?",
            "procedure":"Is this about scheduling a ride, applying for benefits, or arranging training?",
            "coverage":"Do you mean which trips are covered, or what exclusions apply?",
            "documentation":"Do you need required proofs (ID/residence/medical need) or deadlines?",
            "contact":"Do you need the broker’s phone number, office hours, or escalation steps?",
            "other":"What specific program/step should I focus on?"
        }
        return qs.get(intent, qs["other"])

    def respond(self, user_msg: str) -> Dict[str, Any]:
        intent, ents = self.interpreter.interpret(user_msg)
        answer, sources = self.db.answer(user_msg, word_cap=self.word_cap, k=5)
        conf = self._conf(sources)
        payload = {
            "intent": intent,
            "entities": ents,
            "confidence": round(conf, 3),
            "answer": (f"Steps (summary): {answer}" if intent=="procedure"
                       else f"Eligibility (summary): {answer}" if intent=="eligibility"
                       else f"Coverage (summary): {answer}" if intent=="coverage"
                       else answer),
            "sources": sources
        }
        if conf < self.low_conf:
            payload["follow_up_question"] = self._clarifier(intent)
        action_view = Actions.format_steps(answer) if intent=="procedure" else ""
        if action_view:
            payload["action_snippet"] = action_view

        self.history.add(user=user_msg,intent=intent,entities=ents,confidence=conf,
                         answer=payload["answer"],sources=sources)
        return payload

# -------------------- PDF ingestion (optional) --------------------
def read_pdf_text(file) -> str:
    try:
        import fitz  # PyMuPDF
    except Exception:
        return ""
    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        txt = ""
        for page in doc:
            txt += page.get_text()
        return txt
    except Exception:
        return ""

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Manual QA Agent", page_icon=" ", layout="wide")

st.sidebar.title("Manual Source")
uploaded = st.sidebar.file_uploader("Upload manual (.txt/.md/.pdf)", type=["txt","md","pdf","docx"])
manual_text = ""

if "manual_text" not in st.session_state:
    st.session_state.manual_text = ""

# Source input
if uploaded is not None:
    if uploaded.type == "application/pdf":
        with st.spinner("Extracting text from PDF..."):
            manual_text = read_pdf_text(uploaded) or ""
    else:
        manual_text = uploaded.read().decode("utf-8", errors="ignore")
    st.session_state.manual_text = manual_text

st.sidebar.write("or paste manual text:")
manual_text = st.sidebar.text_area(
    "Paste here (Markdown or plain text). This overrides uploaded content for this session.",
    value=st.session_state.manual_text,
    height=200
)
st.session_state.manual_text = manual_text

# Agent options
st.sidebar.subheader("Settings")
low_conf = st.sidebar.slider("Clarify if confidence below", 0.0, 1.0, 0.18, 0.01)
word_cap = st.sidebar.slider("Answer word cap", 40, 600, 150, 10)


# Build / refresh agent when manual changes
if "agent_key" not in st.session_state:
    st.session_state.agent_key = 0

if st.sidebar.button("Load / Refresh manual"):
    st.session_state.agent_key += 1

if not st.session_state.manual_text.strip():
    st.info("Upload or paste your manual on the left, then click **Load / Refresh manual**.")
    st.stop()

@st.cache_resource(show_spinner=False)
def build_agent(manual_text: str, low_conf: float, word_cap: int, key: int):
    return ManualOnlyAgent(manual_text, low_conf=low_conf, word_cap=word_cap, embedder=EMBEDDER, nlp=NLP)

agent = build_agent(st.session_state.manual_text, low_conf, word_cap, st.session_state.agent_key)

st.title("Manual QA Chat")
#st.caption("Ask concise questions; answers come only from your manual. Optional clarifier appears if confidence is low.")

# Chat history (UI)
if "chat" not in st.session_state:
    st.session_state.chat = []

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
            resp = agent.respond(user_msg)

        # Header line with intent + confidence
        st.markdown(f"**Intent:** `{resp['intent']}` · **Confidence:** `{resp['confidence']}`")
        st.markdown(resp["answer"])

        # Optional actionization
        if "action_snippet" in resp and resp["action_snippet"]:
            st.markdown("**Actionable steps**")
            st.code(resp["action_snippet"])

        # Sources expander
        with st.expander("Show sources"):
            for s in resp["sources"]:
                st.markdown(f"- **{s['section']}** — {s['excerpt']}  \n  _offsets: {s['location'][0]}–{s['location'][1]} · score: {s['score']}_")

        # Clarifier if low confidence
        if "follow_up_question" in resp:
            st.info(resp["follow_up_question"])

        # Log in UI history
        combined_answer = f"**Intent:** {resp['intent']} | **Conf:** {resp['confidence']}\n\n{resp['answer']}"
        st.session_state.chat.append(("assistant", combined_answer))
 