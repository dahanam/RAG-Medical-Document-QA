"""
Improved Manual QA Agent with better formatted, readable responses.
Run with: streamlit run chat.py

I think so far this one has had the best responses
"""
import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from functools import lru_cache

import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== Configuration ====================
class Config:
    """Centralized configuration management"""
    MAX_CHUNK_CHARS = 1200
    CHUNK_OVERLAP = 160
    MAX_HISTORY_TURNS = 50
    DEFAULT_LOW_CONF = 0.18
    DEFAULT_WORD_CAP = 150
    MAX_FILE_SIZE_MB = 50
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    SPACY_MODEL = "en_core_web_sm"

# ==================== Resource Loading ====================
@st.cache_resource(show_spinner=False)
def load_embedder():
    """Load sentence transformer with error handling"""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(Config.EMBEDDING_MODEL)
    except Exception as e:
        logger.warning(f"Failed to load embedder: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_spacy():
    """Load spaCy model with error handling"""
    try:
        import spacy
        return spacy.load(Config.SPACY_MODEL)
    except Exception as e:
        logger.warning(f"Failed to load spaCy: {e}")
        return None

EMBEDDER = load_embedder()
NLP = load_spacy()

# ==================== Utility Functions ====================
def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()

@lru_cache(maxsize=1000)
def split_sentences(text: str) -> Tuple[str, ...]:
    """Split text into sentences with caching"""
    if not text:
        return tuple()
    
    out, cur = [], []
    for part in re.split(r"([.?!])", text):
        cur.append(part)
        if part in (".", "?", "!"):
            s = clean_text("".join(cur))
            if s:
                out.append(s)
            cur = []
    
    tail = clean_text("".join(cur))
    if tail:
        out.append(tail)
    return tuple(out)

def validate_query(query: str) -> Tuple[bool, str]:
    """Validate user query"""
    if not query or not query.strip():
        return False, "Please enter a question."
    if len(query) > 500:
        return False, "Query too long (max 500 characters)."
    return True, ""

def format_answer_for_display(answer: str, intent: str) -> str:
    """Format answer text to be more readable with proper structure"""
    
    # Clean up any existing formatting issues
    answer = clean_text(answer)
    
    # Split into sentences
    sentences = [s.strip() for s in split_sentences(answer) if s.strip()]
    
    if not sentences:
        return answer
    
    # For procedure/steps, try to extract numbered or bulleted items
    if intent == "procedure":
        # Look for numbered steps or action verbs
        steps = []
        for sent in sentences:
            # Check if it's already numbered or has action verbs
            if re.match(r'^\d+\.', sent) or any(sent.lower().startswith(verb) for verb in 
                ['call', 'ensure', 'verify', 'check', 'complete', 'contact', 'review', 'submit', 'provide', 'schedule']):
                steps.append(sent)
            else:
                steps.append(sent)
        
        if steps:
            formatted = "**Steps to follow:**\n\n"
            for i, step in enumerate(steps, 1):
                # Remove existing numbering if present
                step = re.sub(r'^\d+\.\s*', '', step)
                formatted += f"{i}. {step}\n\n"
            return formatted.strip()
    
    # For eligibility, break into requirements
    elif intent == "eligibility":
        formatted = "**Eligibility Requirements:**\n\n"
        
        # Try to identify separate requirements
        requirements = []
        current_req = []
        
        for sent in sentences:
            # If sentence starts with keywords, it's likely a new requirement
            if any(sent.lower().startswith(word) for word in ['must', 'need', 'require', 'should', 'client']):
                if current_req:
                    requirements.append(' '.join(current_req))
                    current_req = []
            current_req.append(sent)
        
        if current_req:
            requirements.append(' '.join(current_req))
        
        if requirements:
            for i, req in enumerate(requirements, 1):
                formatted += f"• {req}\n\n"
            return formatted.strip()
        else:
            # Fallback: just bullet the sentences
            for sent in sentences:
                formatted += f"• {sent}\n\n"
            return formatted.strip()
    
    # For coverage, separate what's covered vs not covered
    elif intent == "coverage":
        formatted = "**Coverage Information:**\n\n"
        
        covered = []
        not_covered = []
        general = []
        
        for sent in sentences:
            sent_lower = sent.lower()
            if 'not covered' in sent_lower or 'exclude' in sent_lower or 'does not cover' in sent_lower:
                not_covered.append(sent)
            elif 'covered' in sent_lower or 'include' in sent_lower:
                covered.append(sent)
            else:
                general.append(sent)
        
        if covered:
            formatted += "**✓ What's Covered:**\n"
            for item in covered:
                formatted += f"• {item}\n"
            formatted += "\n"
        
        if not_covered:
            formatted += "**✗ What's NOT Covered:**\n"
            for item in not_covered:
                formatted += f"• {item}\n"
            formatted += "\n"
        
        if general:
            formatted += "**Additional Information:**\n"
            for item in general:
                formatted += f"• {item}\n"
        
        return formatted.strip()
    
    # For documentation, list required docs
    elif intent == "documentation":
        formatted = "**Required Documentation:**\n\n"
        
        docs = []
        deadlines = []
        other = []
        
        for sent in sentences:
            sent_lower = sent.lower()
            if 'deadline' in sent_lower or 'due' in sent_lower or 'within' in sent_lower:
                deadlines.append(sent)
            elif any(word in sent_lower for word in ['proof', 'form', 'document', 'id', 'certificate', 'verification']):
                docs.append(sent)
            else:
                other.append(sent)
        
        if docs:
            formatted += "**Documents Needed:**\n"
            for doc in docs:
                formatted += f"• {doc}\n"
            formatted += "\n"
        
        if deadlines:
            formatted += "**⏰ Important Deadlines:**\n"
            for deadline in deadlines:
                formatted += f"• {deadline}\n"
            formatted += "\n"
        
        if other:
            for item in other:
                formatted += f"{item}\n\n"
        
        return formatted.strip()
    
    # For contact info, format cleanly
    elif intent == "contact":
        formatted = "**Contact Information:**\n\n"
        
        for sent in sentences:
            # Try to identify phone numbers, emails, addresses
            if re.search(r'\d{3}[-.]?\d{3}[-.]?\d{4}', sent):
                formatted += f"📞 {sent}\n\n"
            elif '@' in sent:
                formatted += f"📧 {sent}\n\n"
            elif any(word in sent.lower() for word in ['address', 'office', 'location']):
                formatted += f"📍 {sent}\n\n"
            elif any(word in sent.lower() for word in ['hours', 'open', 'available']):
                formatted += f"🕐 {sent}\n\n"
            else:
                formatted += f"• {sent}\n\n"
        
        return formatted.strip()
    
    # For definitions, keep it simple
    elif intent == "definition":
        formatted = f"**Definition:**\n\n{sentences[0]}\n\n"
        if len(sentences) > 1:
            formatted += "**Additional Information:**\n\n"
            for sent in sentences[1:]:
                formatted += f"• {sent}\n"
        return formatted.strip()
    
    # Default: just paragraph with line breaks between sentences
    else:
        formatted = ""
        for sent in sentences:
            formatted += f"{sent}\n\n"
        return formatted.strip()

def extract_key_points(text: str, max_points: int = 5) -> List[str]:
    """Extract key action points or important information from text"""
    sentences = [s.strip() for s in split_sentences(text) if s.strip()]
    
    # Keywords that indicate important information
    important_keywords = [
        'must', 'require', 'need', 'should', 'call', 'contact',
        'deadline', 'ensure', 'verify', 'complete', 'submit',
        'important', 'note', 'warning', 'critical'
    ]
    
    # Score sentences by importance
    scored = []
    for sent in sentences:
        score = 0
        sent_lower = sent.lower()
        
        # Check for important keywords
        for keyword in important_keywords:
            if keyword in sent_lower:
                score += 2
        
        # Shorter sentences with action verbs are often more important
        if len(sent.split()) < 20 and any(sent_lower.startswith(verb) for verb in 
            ['call', 'ensure', 'verify', 'check', 'complete', 'contact']):
            score += 3
        
        # Numbers and specific details are important
        if re.search(r'\d+', sent):
            score += 1
        
        scored.append((score, sent))
    
    # Sort by score and return top points
    scored.sort(reverse=True, key=lambda x: x[0])
    return [sent for score, sent in scored[:max_points] if score > 0]

# ==================== Data Classes ====================
@dataclass
class Turn:
    """Represents a conversation turn"""
    user: str
    intent: str
    entities: Dict[str, str]
    confidence: float
    answer: str
    sources: List[Dict[str, Any]]
    timestamp: Optional[str] = None

@dataclass
class MessageHistory:
    """Manages conversation history with limits"""
    turns: List[Turn] = field(default_factory=list)
    max_turns: int = Config.MAX_HISTORY_TURNS
    
    def add(self, **kwargs):
        """Add turn with automatic pruning"""
        self.turns.append(Turn(**kwargs))
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]
    
    def get_context(self, n: int = 3) -> List[Turn]:
        """Get last n turns for context"""
        return self.turns[-n:] if self.turns else []

@dataclass
class Chunk:
    """Represents a text chunk with metadata"""
    section: str
    text: str
    start: int
    end: int
    
    def __hash__(self):
        return hash((self.start, self.end))

# ==================== NLP Interpreter ====================
class NLPInterpreter:
    """Enhanced intent classification and entity extraction"""
    
    INTENTS = ["eligibility", "procedure", "coverage", "documentation", 
               "contact", "definition", "other"]
    
    INTENT_KEYWORDS = {
        "eligibility": ["eligib", "qualif", "who can", "am i", "requirements"],
        "procedure": ["how to", "how do", "steps", "procedure", "schedule", 
                     "apply", "process", "enroll"],
        "coverage": ["cover", "covered", "benefit", "pay for", "not covered",
                    "include", "exclude"],
        "documentation": ["document", "proof", "form", "deadline", "paper",
                         "certificate", "verification"],
        "contact": ["call", "phone", "contact", "office", "broker", "reach",
                   "email", "address"],
        "definition": ["what is", "define", "meaning", "explain", "mean"]
    }
    
    def __init__(self, embedder=None, nlp=None):
        self.embedder = embedder
        self.nlp = nlp
        self.label_emb = None
        
        if embedder:
            try:
                self.label_emb = embedder.encode(self.INTENTS, convert_to_tensor=True)
            except Exception as e:
                logger.warning(f"Failed to encode intent labels: {e}")
    
    def classify_intent(self, query: str) -> str:
        """Classify query intent using keywords and embeddings"""
        query_lower = query.lower()
        
        # Try keyword matching first (faster)
        for intent, keywords in self.INTENT_KEYWORDS.items():
            if any(kw in query_lower for kw in keywords):
                return intent
        
        # Fall back to semantic similarity if available
        if self.embedder and self.label_emb is not None:
            try:
                from sentence_transformers import util
                qv = self.embedder.encode([query], convert_to_tensor=True)
                sims = util.cos_sim(qv, self.label_emb)[0].tolist()
                best_idx = max(range(len(sims)), key=lambda i: sims[i])
                return self.INTENTS[best_idx]
            except Exception as e:
                logger.warning(f"Intent classification error: {e}")
        
        return "other"
    
    def extract_entities(self, query: str) -> Dict[str, str]:
        """Extract entities using spaCy or regex fallback"""
        entities = {}
        
        if self.nlp:
            try:
                doc = self.nlp(query)
                for ent in doc.ents:
                    entities[ent.label_.lower()] = ent.text
                
                # Domain-specific entity detection
                for token in doc:
                    if token.text.lower() in {"medicaid", "nemt", "logisticare", 
                                             "voucher", "county", "broker"}:
                        entities.setdefault("program", token.text)
            except Exception as e:
                logger.warning(f"Entity extraction error: {e}")
        else:
            # Regex fallback
            patterns = {
                "medicaid": r"\bmedicaid\b",
                "nemt": r"\bnemt\b",
                "logisticare": r"\blogisticare\b",
                "voucher": r"\bvoucher\b",
                "county": r"\bcounty\b"
            }
            for name, pattern in patterns.items():
                if re.search(pattern, query, re.I):
                    entities["program"] = name.title()
        
        return entities
    
    def interpret(self, query: str) -> Tuple[str, Dict[str, str]]:
        """Main interpretation method"""
        intent = self.classify_intent(query)
        entities = self.extract_entities(query)
        return intent, entities

# ==================== Manual Database ====================
class ManualDB:
    """Enhanced retrieval system with hybrid search"""
    
    def __init__(self, manual_text: str, max_chars: int = None, 
                 overlap: int = None, embedder=None):
        self.text = manual_text
        self.max_chars = max_chars or Config.MAX_CHUNK_CHARS
        self.overlap = overlap or Config.CHUNK_OVERLAP
        self.embedder = embedder
        
        # Chunk the manual
        self.chunks = self._create_chunks()
        
        # Create embeddings if available
        self.chunk_embeddings = None
        if embedder and self.chunks:
            try:
                texts = [c.text for c in self.chunks]
                self.chunk_embeddings = embedder.encode(texts, convert_to_tensor=True)
            except Exception as e:
                logger.warning(f"Failed to create embeddings: {e}")
    
    def _create_chunks(self) -> List[Chunk]:
        """Create overlapping chunks from manual"""
        if not self.text:
            return []
        
        lines = self.text.splitlines()
        sections = []
        current_section = []
        current_title = "General"
        
        # Identify sections by headers
        header_pattern = r"^\s*(#{1,6}\s+.+|[0-9]+\.\s+.+|[A-Z][A-Z0-9 \-]{6,})\s*$"
        
        for line in lines:
            if re.match(header_pattern, line):
                if current_section:
                    sections.append((current_title, "\n".join(current_section).strip()))
                    current_section = []
                current_title = clean_text(re.sub(r"^\s*#{1,6}\s*", "", line))
            else:
                current_section.append(line)
        
        if current_section:
            sections.append((current_title, "\n".join(current_section).strip()))
        
        # Create chunks with overlap
        chunks = []
        for title, body in sections:
            if not body:
                continue
            
            i = 0
            while i < len(body):
                end_pos = min(i + self.max_chars, len(body))
                window = body[i:end_pos]
                
                # Try to break at sentence boundary
                if end_pos < len(body):
                    last_period = window.rfind(". ")
                    if last_period > self.max_chars * 0.5:
                        end_pos = i + last_period + 1
                        window = body[i:end_pos]
                
                # Find absolute position in original text
                abs_start = self.text.find(window)
                if abs_start == -1:
                    abs_start = 0  # Fallback
                
                chunks.append(Chunk(
                    section=title,
                    text=window,
                    start=abs_start,
                    end=abs_start + len(window)
                ))
                
                if end_pos >= len(body):
                    break
                
                i = max(i + 1, end_pos - self.overlap)
        
        return chunks
    
    def _keyword_search(self, query: str, k: int) -> List[Tuple[Chunk, float]]:
        """Keyword-based search fallback"""
        query_tokens = set(re.findall(r"\w+", query.lower()))
        
        scored = []
        for chunk in self.chunks:
            chunk_tokens = set(re.findall(r"\w+", chunk.text.lower()))
            score = len(query_tokens & chunk_tokens)
            scored.append((chunk, float(score)))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
    
    def retrieve(self, query: str, k: int = 5) -> List[Tuple[Chunk, float]]:
        """Retrieve relevant chunks using semantic or keyword search"""
        if not self.chunks:
            return []
        
        # Use semantic search if available
        if self.embedder and self.chunk_embeddings is not None:
            try:
                from sentence_transformers import util
                query_emb = self.embedder.encode([query], convert_to_tensor=True)
                similarities = util.cos_sim(query_emb, self.chunk_embeddings)[0]
                
                top_k = min(k, len(self.chunks))
                top_scores, top_indices = similarities.topk(k=top_k)
                
                return [(self.chunks[idx], float(score)) 
                        for score, idx in zip(top_scores.tolist(), top_indices.tolist())]
            except Exception as e:
                logger.warning(f"Semantic search failed: {e}")
        
        # Fall back to keyword search
        return self._keyword_search(query, k)
    
    def generate_answer(self, query: str, word_limit: int = 150, 
                       k: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
        """Generate answer from retrieved chunks"""
        retrieved = self.retrieve(query, k=k)
        
        if not retrieved:
            return "I couldn't find relevant information in the manual.", []
        
        # Extract relevant sentences
        all_sentences = []
        for chunk, _ in retrieved:
            sentences = split_sentences(chunk.text)
            # Filter for substantive sentences
            relevant = [s for s in sentences if len(s.split()) >= 5]
            all_sentences.extend(relevant)
        
        if not all_sentences:
            all_sentences = [clean_text(retrieved[0][0].text)]
        
        # Rank sentences by relevance
        if EMBEDDER:
            try:
                from sentence_transformers import util
                query_emb = EMBEDDER.encode([query], convert_to_tensor=True)
                sent_embs = EMBEDDER.encode(list(all_sentences), convert_to_tensor=True)
                sims = util.cos_sim(query_emb, sent_embs)[0].tolist()
                ranked = [s for _, s in sorted(zip(sims, all_sentences), 
                                              key=lambda x: x[0], reverse=True)]
            except Exception as e:
                logger.warning(f"Sentence ranking failed: {e}")
                ranked = all_sentences
        else:
            # Keyword-based ranking
            query_tokens = set(re.findall(r"\w+", query.lower()))
            scored = [(len(query_tokens & set(re.findall(r"\w+", s.lower()))), s) 
                     for s in all_sentences]
            ranked = [s for _, s in sorted(scored, key=lambda x: x[0], reverse=True)]
        
        # Deduplicate and build answer
        seen = set()
        selected = []
        word_count = 0
        
        for sent in ranked:
            # Simple deduplication key
            key = re.sub(r"[^a-z0-9]+", "", sent.lower())[:60]
            if key in seen:
                continue
            
            seen.add(key)
            selected.append(sent)
            word_count += len(sent.split())
            
            if word_count >= word_limit:
                break
        
        answer = " ".join(selected).strip()
        
        # Format sources
        sources = [{
            "section": chunk.section,
            "excerpt": clean_text(chunk.text)[:250] + ("…" if len(chunk.text) > 250 else ""),
            "location": [chunk.start, chunk.end],
            "score": round(score, 4) if score <= 1.0 else round(score, 2)
        } for chunk, score in retrieved]
        
        return answer, sources

# ==================== Agent ====================
class ManualQAAgent:
    """Main agent orchestrating QA process"""
    
    def __init__(self, manual_text: str, low_confidence: float = None,
                 word_limit: int = None, embedder=None, nlp=None):
        self.history = MessageHistory()
        self.interpreter = NLPInterpreter(embedder=embedder, nlp=nlp)
        self.db = ManualDB(manual_text, embedder=embedder)
        self.low_confidence = low_confidence or Config.DEFAULT_LOW_CONF
        self.word_limit = word_limit or Config.DEFAULT_WORD_CAP
    
    def _calculate_confidence(self, sources: List[Dict[str, Any]]) -> float:
        """Calculate confidence from source scores"""
        if not sources:
            return 0.0
        
        max_score = max(s.get("score", 0.0) for s in sources)
        
        # Normalize scores > 1.0 (keyword match counts)
        if max_score > 1.0:
            return min(1.0, max_score / (max_score + 5.0))
        
        return float(max_score)
    
    def _get_clarification(self, intent: str) -> str:
        """Get clarification question for low-confidence responses"""
        clarifications = {
            "eligibility": "Which program are you asking about (e.g., Medicaid NEMT, vouchers)?",
            "procedure": "Are you asking about scheduling, applying, or enrollment?",
            "coverage": "Do you want to know what's covered or what's excluded?",
            "documentation": "Do you need information about required documents or deadlines?",
            "contact": "Do you need a phone number, address, or email?",
            "other": "Could you provide more details about what you're looking for?"
        }
        return clarifications.get(intent, clarifications["other"])
    
    def respond(self, user_query: str) -> Dict[str, Any]:
        """Generate response to user query"""
        # Validate query
        valid, error_msg = validate_query(user_query)
        if not valid:
            return {
                "intent": "invalid",
                "entities": {},
                "confidence": 0.0,
                "answer": error_msg,
                "sources": []
            }
        
        try:
            # Interpret query
            intent, entities = self.interpreter.interpret(user_query)
            
            # Retrieve and generate answer
            answer, sources = self.db.generate_answer(
                user_query, 
                word_limit=self.word_limit,
                k=5
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(sources)
            
            # Format answer for better readability
            formatted_answer = format_answer_for_display(answer, intent)
            
            # Extract key points for quick reference
            key_points = extract_key_points(answer, max_points=3)
            
            # Build response
            response = {
                "intent": intent,
                "entities": entities,
                "confidence": round(confidence, 3),
                "answer": formatted_answer,
                "raw_answer": answer,  # Keep original for reference
                "key_points": key_points,
                "sources": sources
            }
            
            # Add clarification if confidence is low
            if confidence < self.low_confidence:
                response["clarification"] = self._get_clarification(intent)
            
            # Add to history
            self.history.add(
                user=user_query,
                intent=intent,
                entities=entities,
                confidence=confidence,
                answer=answer,
                sources=sources
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "intent": "error",
                "entities": {},
                "confidence": 0.0,
                "answer": "I encountered an error processing your question. Please try rephrasing.",
                "sources": []
            }

# ==================== File Processing ====================
def read_pdf_file(uploaded_file) -> str:
    """Extract text from PDF with error handling"""
    try:
        import fitz  # PyMuPDF
        
        # Check file size
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > Config.MAX_FILE_SIZE_MB:
            st.error(f"File too large ({file_size_mb:.1f}MB). Max size: {Config.MAX_FILE_SIZE_MB}MB")
            return ""
        
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
        
    except ImportError:
        st.error("PyMuPDF not installed. Install with: pip install pymupdf")
        return ""
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

# ==================== Streamlit UI ====================
def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Manual QA Agent",
        page_icon="📚",
        layout="wide"
    )
    
    # Initialize session state
    if "manual_text" not in st.session_state:
        st.session_state.manual_text = ""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "agent_version" not in st.session_state:
        st.session_state.agent_version = 0
    
    # Sidebar configuration
    st.sidebar.title("📚 Manual Source")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload manual",
        type=["txt", "md", "pdf"],
        help="Upload a text, markdown, or PDF file"
    )
    
    # Process uploaded file
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            with st.spinner("Extracting text from PDF..."):
                manual_text = read_pdf_file(uploaded_file)
        else:
            try:
                manual_text = uploaded_file.read().decode("utf-8", errors="ignore")
            except Exception as e:
                st.sidebar.error(f"Error reading file: {e}")
                manual_text = ""
        
        if manual_text:
            st.session_state.manual_text = manual_text
    
    # Manual text input
    st.sidebar.write("**Or paste text directly:**")
    pasted_text = st.sidebar.text_area(
        "Manual text",
        value=st.session_state.manual_text,
        height=200,
        help="Paste your manual text here"
    )
    
    if pasted_text != st.session_state.manual_text:
        st.session_state.manual_text = pasted_text
    
    # Settings
    st.sidebar.subheader("⚙️ Settings")
    low_conf = st.sidebar.slider(
        "Clarification threshold",
        0.0, 1.0, Config.DEFAULT_LOW_CONF, 0.01,
        help="Ask for clarification if confidence is below this"
    )
    word_limit = st.sidebar.slider(
        "Answer length (words)",
        50, 400, Config.DEFAULT_WORD_CAP, 10
    )
    
    # Load/refresh button
    if st.sidebar.button("🔄 Load / Refresh Manual", type="primary"):
        if st.session_state.manual_text.strip():
            st.session_state.agent_version += 1
            st.session_state.chat_history = []
            st.success("Manual loaded!")
        else:
            st.error("Please provide manual text first")
    
    # Check if manual is loaded
    if not st.session_state.manual_text.strip():
        st.info("👈 Upload or paste your manual in the sidebar, then click **Load / Refresh Manual**")
        st.stop()
    
    # Build agent (cached)
    @st.cache_resource(show_spinner=False)
    def get_agent(manual_text: str, low_conf: float, word_limit: int, version: int):
        return ManualQAAgent(
            manual_text,
            low_confidence=low_conf,
            word_limit=word_limit,
            embedder=EMBEDDER,
            nlp=NLP
        )
    
    agent = get_agent(
        st.session_state.manual_text,
        low_conf,
        word_limit,
        st.session_state.agent_version
    )
    
    # Main chat interface
    st.title("💬 Manual QA Chat")
    st.caption(f"Ask questions about your manual. Using {'semantic search ✨' if EMBEDDER else 'keyword search'}.")
    
    # Display chat history
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message, unsafe_allow_html=False)
    
    # Chat input
    user_input = st.chat_input("Ask a question about the manual...")
    
    if user_input:
        # Add user message
        st.session_state.chat_history.append(("user", user_input))
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = agent.respond(user_input)
            
            # Display metadata bar
            col1, col2 = st.columns([2, 1])
            with col1:
                # Intent badge with color
                intent_colors = {
                    "eligibility": "🎯",
                    "procedure": "📋",
                    "coverage": "🛡️",
                    "documentation": "📄",
                    "contact": "📞",
                    "definition": "📖",
                    "other": "❓"
                }
                icon = intent_colors.get(response['intent'], "💬")
                st.markdown(f"{icon} **Intent:** `{response['intent']}`")
            
            with col2:
                # Confidence indicator with color
                conf = response['confidence']
                if conf >= 0.7:
                    conf_color = "🟢"
                elif conf >= 0.4:
                    conf_color = "🟡"
                else:
                    conf_color = "🔴"
                st.markdown(f"{conf_color} **Confidence:** `{conf:.0%}`")
            
            st.divider()
            
            # Display formatted answer
            st.markdown(response["answer"])
            
            # Show key points if available
            if response.get("key_points") and len(response["key_points"]) > 0:
                with st.expander("🎯 Key Takeaways", expanded=False):
                    for i, point in enumerate(response["key_points"], 1):
                        st.markdown(f"{i}. {point}")
            
            # Show sources
            if response["sources"]:
                with st.expander(f"📖 View {len(response['sources'])} source(s)", expanded=False):
                    for i, source in enumerate(response["sources"], 1):
                        st.markdown(f"**Source {i}: {source['section']}**")
                        st.markdown(f"> {source['excerpt']}")
                        
                        # Show score with visual indicator
                        score = source['score']
                        if score <= 1.0:
                            bar_width = int(score * 20)
                            st.caption(f"Relevance: {'█' * bar_width}{'░' * (20-bar_width)} {score:.2%}")
                        else:
                            st.caption(f"Keyword matches: {int(score)}")
                        
                        if i < len(response["sources"]):
                            st.divider()
            
            # Show clarification if needed
            if "clarification" in response:
                st.info(f"💡 **Need more details?** {response['clarification']}")
            
            # Add to chat history (simplified version)
            intent_emoji = intent_colors.get(response['intent'], "💬")
            assistant_msg = f"{intent_emoji} **{response['intent'].title()}** ({response['confidence']:.0%})\n\n{response['answer']}"
            st.session_state.chat_history.append(("assistant", assistant_msg))
    
    # Footer with helpful info
    with st.sidebar:
        st.divider()
        st.caption("💡 **Tips for better answers:**")
        st.caption("• Be specific in your questions")
        st.caption("• Mention relevant program names")
        st.caption("• Ask one thing at a time")
        
        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()

if __name__ == "__main__":
    main()