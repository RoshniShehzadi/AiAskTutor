import streamlit as st
import os
import json
from gtts import gTTS
from io import BytesIO
import difflib
import numpy as np
from typing import Optional
from dotenv import load_dotenv
import base64
import tempfile

# Load environment variables from .env file
load_dotenv()

# Deployment switch:
# - False (default): lightweight mode, no sentence-transformer runtime load
# - True: full semantic matching (heavier)
USE_SEMANTIC_SEARCH = os.getenv("USE_SEMANTIC_SEARCH", "false").lower() in ("1", "true", "yes")

# Voice Input Import
from streamlit_mic_recorder import mic_recorder

# --- CONFIGURATION ---
st.set_page_config(page_title="AskAI Tutor", page_icon="🎓", layout="wide")

# Styling
st.markdown("""
<style>
    .chat-message { padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex }
    .chat-message.user { background-color: #2b313e }
    .chat-message.bot { background-color: #475063 }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS (must be defined before use) ---

def _norm(text: str) -> str:
    """Normalize text by lowercasing and collapsing whitespace."""
    return " ".join((text or "").lower().split())

# --- EMBEDDING-BASED SEMANTIC SEARCH ---
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    """Load and cache the sentence transformer model for semantic search."""
    try:
        from sentence_transformers import SentenceTransformer
        # Using a lightweight, fast model that works well for semantic similarity
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ [DEBUG] Semantic embedding model loaded successfully")
        return model
    except Exception as e:
        print(f"⚠️ [DEBUG] Failed to load embedding model: {e}")
        return None

@st.cache_data(show_spinner=False)
def generate_dataset_embeddings(dataset: list[dict]) -> Optional[np.ndarray]:
    """Generate embeddings for all questions in the dataset.
    Returns a numpy array of shape (len(dataset), embedding_dim).
    Note: Model is fetched inside this function to avoid caching issues."""
    if not dataset:
        return None
    
    # Get model inside the function to avoid caching the model object
    model = get_embedding_model()
    if model is None:
        return None
    
    try:
        # Extract questions, handling empty or missing questions
        questions = []
        valid_indices = []
        for idx, entry in enumerate(dataset):
            question = entry.get("question", "").strip()
            if question:  # Only include non-empty questions
                questions.append(question)
                valid_indices.append(idx)
        
        if not questions:
            print("⚠️ [DEBUG] No valid questions found in dataset")
            return None
        
        # Generate embeddings
        embeddings = model.encode(questions, show_progress_bar=False, convert_to_numpy=True)
        
        # Create full array matching dataset size (fill missing with zeros)
        if len(valid_indices) < len(dataset):
            full_embeddings = np.zeros((len(dataset), embeddings.shape[1]), dtype=np.float32)
            for i, orig_idx in enumerate(valid_indices):
                full_embeddings[orig_idx] = embeddings[i]
            embeddings = full_embeddings
        
        print(f"✅ [DEBUG] Generated embeddings for {len(questions)} questions (dataset size: {len(dataset)})")
        return embeddings
    except Exception as e:
        print(f"⚠️ [DEBUG] Failed to generate embeddings: {e}")
        import traceback
        traceback.print_exc()
        return None

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors.
    Returns a value between -1 and 1, where 1 means identical vectors."""
    try:
        # Validate inputs
        if vec1 is None or vec2 is None:
            return 0.0
        
        # Ensure vectors are numpy arrays and have same shape
        vec1 = np.asarray(vec1, dtype=np.float32)
        vec2 = np.asarray(vec2, dtype=np.float32)
        
        if vec1.shape != vec2.shape:
            print(f"⚠️ [DEBUG] Vector shape mismatch: {vec1.shape} vs {vec2.shape}")
            return 0.0
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        # Clamp to [-1, 1] range (should already be, but safety check)
        return float(np.clip(similarity, -1.0, 1.0))
    except Exception as e:
        print(f"⚠️ [DEBUG] Error in cosine_similarity: {e}")
        return 0.0

def text_to_speech(text: str):
    """Convert text to speech using gTTS."""
    if not text or not text.strip():
        return None
    try:
        tts = gTTS(text=text, lang='en', tld='co.in', slow=False)
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp
    except Exception as e:
        st.error(f"Error generating audio: {e}")
        return None

def detect_audio_mime(audio_bytes: bytes) -> str:
    """Detect MIME type from audio bytes."""
    if not audio_bytes:
        return "audio/wav"
    if audio_bytes.startswith(b"RIFF") and audio_bytes[8:12] == b"WAVE":
        return "audio/wav"
    if audio_bytes.startswith(b"OggS"):
        return "audio/ogg"
    if audio_bytes.startswith(b"\x1a\x45\xdf\xa3"):
        return "audio/webm"
    if audio_bytes.startswith(b"ID3") or audio_bytes[:2] == b"\xff\xfb":
        return "audio/mpeg"
    if audio_bytes.startswith(b"fLaC"):
        return "audio/flac"
    return "application/octet-stream"

def audio_to_text(audio_bytes: bytes) -> Optional[str]:
    """Convert audio bytes to text using speech recognition."""
    if not audio_bytes:
        print("⚠️ [DEBUG] audio_to_text: No audio bytes provided")
        return None
    
    print(f"🔵 [DEBUG] audio_to_text: Processing {len(audio_bytes)} bytes")
    tmp_file_path = None
    converted_path = None
    
    try:
        import speech_recognition as sr
        from pydub import AudioSegment
        
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = True
        
        print(f"🔵 [DEBUG] Checking audio format...")
        audio_header = audio_bytes[:12] if len(audio_bytes) >= 12 else audio_bytes
        print(f"🔵 [DEBUG] Audio header (first 12 bytes): {audio_header.hex() if isinstance(audio_header, bytes) else 'N/A'}")
        
        is_raw_pcm = False
        if len(audio_bytes) > 4:
            header_hex = audio_bytes[:4].hex()
            if header_hex not in ['52494646', '4f676753', '1a45dfa3']:
                is_raw_pcm = True
                print(f"🔵 [DEBUG] Detected raw PCM audio data")
        
        print(f"🔵 [DEBUG] Writing audio to temp file...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav" if not is_raw_pcm else ".raw") as tmp_file:
            if is_raw_pcm:
                import struct
                sample_rate = 16000
                channels = 1
                sample_width = 2
                num_samples = len(audio_bytes) // (channels * sample_width)
                
                wav_header = struct.pack('<4sI4s4sIHHIIHH4sI',
                    b'RIFF',
                    36 + num_samples * channels * sample_width,
                    b'WAVE',
                    b'fmt ',
                    16,
                    1,
                    channels,
                    sample_rate,
                    sample_rate * channels * sample_width,
                    channels * sample_width,
                    sample_width * 8,
                    b'data',
                    num_samples * channels * sample_width
                )
                tmp_file.write(wav_header)
                tmp_file.write(audio_bytes)
            else:
                tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name
            print(f"🔵 [DEBUG] Temp file created: {tmp_file_path}")
        
        try:
            print(f"🔵 [DEBUG] Loading audio file...")
            try:
                if is_raw_pcm:
                    audio = AudioSegment.from_raw(tmp_file_path, sample_width=2, frame_rate=16000, channels=1)
                else:
                    audio = AudioSegment.from_file(tmp_file_path)
            except Exception as e:
                print(f"⚠️ [DEBUG] Error loading audio, trying with explicit format... Error: {e}")
                try:
                    audio = AudioSegment.from_file(tmp_file_path, format="webm")
                except Exception as e2:
                    print(f"⚠️ [DEBUG] WebM failed, trying OGG... Error: {e2}")
                    try:
                        audio = AudioSegment.from_file(tmp_file_path, format="ogg")
                    except Exception as e3:
                        print(f"⚠️ [DEBUG] OGG failed, trying MP3... Error: {e3}")
                        try:
                            audio = AudioSegment.from_file(tmp_file_path, format="mp3")
                        except Exception as e4:
                            print(f"⚠️ [DEBUG] MP3 failed, trying raw PCM... Error: {e4}")
                            try:
                                audio = AudioSegment.from_raw(tmp_file_path, sample_width=2, frame_rate=16000, channels=1)
                            except Exception as e5:
                                print(f"❌ [DEBUG] All formats failed. Last error: {e5}")
                                raise e5
            
            print(f"🔵 [DEBUG] Original audio: {len(audio)}ms, {audio.channels} channels, {audio.frame_rate}Hz")
            
            if len(audio) < 100:
                print(f"⚠️ [DEBUG] Audio too short ({len(audio)}ms), might be empty")
                return None
            
            audio = audio.set_channels(1)
            audio = audio.set_frame_rate(16000)
            audio = audio.normalize()
            print(f"🔵 [DEBUG] Converted audio: {len(audio)}ms, {audio.channels} channels, {audio.frame_rate}Hz")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as converted_file:
                audio.export(converted_file.name, format="wav", parameters=["-ac", "1", "-ar", "16000"])
                converted_path = converted_file.name
                print(f"🔵 [DEBUG] Converted file created: {converted_path}, size: {os.path.getsize(converted_path)} bytes")
            
            print(f"🔵 [DEBUG] Starting speech recognition...")
            with sr.AudioFile(converted_path) as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.2)
                audio_data = recognizer.record(source)
                file_size = os.path.getsize(converted_path)
                print(f"🔵 [DEBUG] Audio recorded (file size: {file_size} bytes), calling Google Speech Recognition...")
                try:
                    text = recognizer.recognize_google(audio_data, language="en-US")
                    print(f"✅ [DEBUG] Recognition successful: '{text}'")
                    return text
                except sr.UnknownValueError as e:
                    print(f"❌ [DEBUG] Google could not understand audio: {e}")
                    return None
                except sr.RequestError as e:
                    print(f"❌ [DEBUG] Google API error: {e}")
                    return None
        finally:
            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    os.unlink(tmp_file_path)
                    print(f"🔵 [DEBUG] Cleaned up temp file: {tmp_file_path}")
                except Exception as e:
                    print(f"⚠️ [DEBUG] Error cleaning temp file: {e}")
            if converted_path and os.path.exists(converted_path):
                try:
                    os.unlink(converted_path)
                    print(f"🔵 [DEBUG] Cleaned up converted file: {converted_path}")
                except Exception as e:
                    print(f"⚠️ [DEBUG] Error cleaning converted file: {e}")
    except sr.UnknownValueError as e:
        print(f"❌ [DEBUG] Speech recognition could not understand audio: {e}")
        st.warning("Could not understand the audio. Please speak more clearly.")
        return None
    except sr.RequestError as e:
        print(f"❌ [DEBUG] Speech recognition service error: {e}")
        st.error(f"Speech recognition service error. Please check your internet connection.")
        return None
    except Exception as e:
        print(f"❌ [DEBUG] Unexpected error in audio_to_text: {e}")
        import traceback
        traceback.print_exc()
        st.error(f"Error processing audio: {str(e)}")
        return None
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        if converted_path and os.path.exists(converted_path):
            try:
                os.unlink(converted_path)
            except:
                pass

@st.cache_data(show_spinner=False)
def get_available_text_models() -> list[str]:
    # Define Hugging Face models here
    available_models = ["gpt2", "EleutherAI/gpt-neo-2.7B", "facebook/bart-large", "t5-base"]
    return available_models

def pick_default_model(available: list[str]) -> str:
    preferred = [
        "gpt2",
        "EleutherAI/gpt-neo-2.7B",
        "facebook/bart-large",
        "t5-base",
    ]
    # Picking default model from available models
    for p in preferred:
        if p in available:
            return p
    return available[0] if available else "gpt2"

def validate_model_supports_text_generation(model_name: str) -> bool:
    """Validate that the selected model supports text generation."""
    available_models = get_available_text_models()
    return model_name in available_models

# --- 1. SETUP SIDEBAR ---
print("🔵 [DEBUG] Step 1: Initializing sidebar")
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/student-male--v1.png", width=100)
    st.title("AskAI Tutor 🎓")
    st.markdown("### Human Engineering Mentor")
    
    # API key from environment variable (optional - not shown in UI)
    HF_API_KEY = os.getenv("HF_API_KEY", "")
    MODEL_NAME = "gpt2"
    
    st.info("💡 Ask me anything about Human Engineering Skills, balanced schedules, mentorship, or character development!")
    if USE_SEMANTIC_SEARCH:
        st.caption("Mode: Full semantic search")
    else:
        st.caption("Mode: Lightweight deploy (fast startup)")
    
# --- 2. DATA LOADING & LOCAL RETRIEVAL (Cached) ---
@st.cache_data(show_spinner=False)
def load_dataset():
    print("🔵 [DEBUG] Step 2: Loading dataset from dataset.json")
    try:
        with open("dataset.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        st.error("dataset.json not found!")
        return []
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return []

@st.cache_data(show_spinner=False)
def preprocess_dataset(dataset: list[dict]) -> list[dict]:
    """Precompute normalized text fields used during retrieval."""
    preprocessed = []
    for entry in dataset:
        question_norm = _norm(entry.get("question", ""))
        answer_norm = _norm(entry.get("answer", ""))
        preprocessed.append(
            {
                "question_norm": question_norm,
                "answer_norm": answer_norm,
                "question_words": set(question_norm.split()) if question_norm else set(),
            }
        )
    return preprocessed

def retrieve_context(
    query: str,
    dataset: list[dict],
    k: int = 3,
    use_semantic: bool = True,
    preprocessed_dataset: Optional[list[dict]] = None,
    embedding_model=None,
    dataset_embeddings: Optional[np.ndarray] = None,
) -> tuple[str, str]:
    """Retrieve relevant context from dataset using hybrid similarity matching.
    Combines semantic embeddings with string-based matching for best results.
    Returns: (context_string, best_match_answer)"""
    q = _norm(query)
    if not q or not dataset:
        return "", ""
    
    # Validate k parameter
    k = max(1, min(k, len(dataset)))  # Ensure k is between 1 and dataset size
    
    # Use precomputed/cached resources whenever available
    if preprocessed_dataset is None:
        preprocessed_dataset = preprocess_dataset(dataset)
    if use_semantic and embedding_model is None:
        embedding_model = get_embedding_model()
    if use_semantic and dataset_embeddings is None:
        dataset_embeddings = generate_dataset_embeddings(dataset)
    
    # Validate embeddings match dataset size
    if dataset_embeddings is not None and len(dataset_embeddings) != len(dataset):
        print(f"⚠️ [DEBUG] Embedding size mismatch: {len(dataset_embeddings)} vs {len(dataset)}. Disabling semantic search.")
        dataset_embeddings = None
        embedding_model = None
    
    # Generate query embedding if semantic search is available
    query_embedding = None
    if embedding_model and dataset_embeddings is not None:
        try:
            query_embedding = embedding_model.encode([query], convert_to_numpy=True)[0]
        except Exception as e:
            print(f"⚠️ [DEBUG] Failed to encode query: {e}")
            query_embedding = None
    
    scored = []
    for idx, entry in enumerate(dataset):
        cached_entry = preprocessed_dataset[idx] if idx < len(preprocessed_dataset) else {}
        question = cached_entry.get("question_norm", _norm(entry.get("question", "")))
        answer = cached_entry.get("answer_norm", _norm(entry.get("answer", "")))
        
        # Skip empty entries
        if not question and not answer:
            continue
        
        # 1. Semantic similarity score (if available)
        semantic_score = 0.0
        if query_embedding is not None and dataset_embeddings is not None and idx < len(dataset_embeddings):
            try:
                semantic_score = cosine_similarity(query_embedding, dataset_embeddings[idx])
                # Ensure semantic score is valid (between 0 and 1)
                semantic_score = max(0.0, min(1.0, semantic_score))
            except (IndexError, ValueError, Exception) as e:
                print(f"⚠️ [DEBUG] Error calculating semantic similarity at index {idx}: {e}")
                semantic_score = 0.0
        
        # 2. String-based similarity scores (fallback/backup)
        question_score = difflib.SequenceMatcher(None, q, question).ratio() if question else 0.0
        q_words = set(q.split()) if q else set()
        question_words = cached_entry.get("question_words", set(question.split()) if question else set())
        word_overlap = len(q_words & question_words) / max(len(q_words), 1) if q_words else 0.0
        answer_score = difflib.SequenceMatcher(None, q, answer).ratio() if answer else 0.0
        
        # 3. Hybrid scoring: Prioritize semantic if available, combine with string matching
        if semantic_score > 0:
            # Use semantic as primary (60%), string matching as secondary (40%)
            combined_score = (semantic_score * 0.6) + (question_score * 0.25) + (word_overlap * 0.10) + (answer_score * 0.05)
        else:
            # Fallback to original string-based scoring
            combined_score = (question_score * 0.7) + (word_overlap * 0.2) + (answer_score * 0.1)
        
        # Ensure combined score is valid
        combined_score = max(0.0, min(1.0, combined_score))
        
        scored.append((combined_score, semantic_score, question_score, entry))

    # Sort by combined score (descending)
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [e for s, sem, qs, e in scored[:k] if s > 0]
    if not top:
        return "", ""
    
    # Get the best match answer
    best_match_answer = top[0].get('answer', '') if top else ''
    best_score = scored[0][0] if scored else 0.0
    best_semantic_score = scored[0][1] if scored else 0.0
    best_question_score = scored[0][2] if scored else 0.0
    
    print(f"🔵 [DEBUG] Best match - Combined: {best_score:.3f}, Semantic: {best_semantic_score:.3f}, String: {best_question_score:.3f}")
    if top:
        print(f"🔵 [DEBUG] Best match question: '{top[0].get('question', '')[:80]}...'")
    
    # Use dataset answer if match is good
    # For semantic: threshold 0.6, for string: threshold 0.4
    use_dataset_answer = (best_semantic_score >= 0.6) or (best_question_score >= 0.4) or (best_score >= 0.5)
    
    # Build context string safely
    context_parts = []
    for i, e in enumerate(top):
        q_text = e.get('question', '').strip()
        a_text = e.get('answer', '').strip()
        if q_text or a_text:
            context_parts.append(f"[{i+1}] Question: {q_text}\nAnswer: {a_text}")
    
    context = "\n\n".join(context_parts) if context_parts else ""
    return context, best_match_answer if use_dataset_answer else ""

# --- 3. PROMPT ENGINEERING & LOGIC ---
system_prompt = """You are a wise Human Engineering Mentor helping students understand life balance, discipline, and mentorship.

IMPORTANT INSTRUCTIONS:
1. Answer the student's question directly and clearly
2. If the context contains relevant information, use it to help answer
3. If the context doesn't contain relevant information, politely say you don't have specific information about that topic
4. Keep your response focused, coherent, and helpful
5. End with a brief reflective question if appropriate

Context: {context}

Student's Question: {question}

Mentor's Response:"""

def build_prompt(user_input: str, context: str, history: list[dict]) -> str:
    # Build a cleaner, more structured prompt
    context_text = context if context and len(context.strip()) > 20 else "No specific context available for this question."
    
    # Format the prompt with clear structure
    prompt = system_prompt.format(
        context=context_text,
        question=user_input
    )
    
    # Add conversation history if available (limit to last 3 exchanges to keep prompt focused)
    if history:
        history_lines = []
        for msg in history[-6:]:  # Last 3 exchanges (user + assistant pairs)
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                history_lines.append(f"Previous Student Question: {content}")
            elif role == "assistant":
                history_lines.append(f"Previous Mentor Answer: {content}")
        
        if history_lines:
            prompt += "\n\nPrevious Conversation:\n" + "\n".join(history_lines)
    
    return prompt

# --- 4. GENERATING ANSWERS ---
@st.cache_resource(show_spinner=False)
def get_generator(model_name: str, api_key: str):
    """Get or create generator pipeline for the specified model."""
    try:
        # Lazy import to avoid Streamlit hot-reload timing issues
        from transformers import pipeline
        return pipeline("text-generation", model=model_name, token=api_key)
    except Exception as e:
        st.error(f"Failed to load model '{model_name}': {e}")
        return None

def generate_answer(user_input: str, context: str, history: list[dict], model_name: str, api_key: str, dataset_answer: str = "") -> str:
    """Generate answer using Hugging Face model."""
    # Validate inputs
    if not user_input or not user_input.strip():
        return ""
    
    if dataset_answer and len(dataset_answer.strip()) > 10:
        print(f"✅ [DEBUG] Using dataset answer (length: {len(dataset_answer)} chars)")
        return dataset_answer.strip()
    else:
        print(f"⚠️ [DEBUG] No dataset answer found. Question is outside dataset. Returning no information message.")
        return "I don't have specific information about that topic in my knowledge base. Could you rephrase your question, or ask about something related to Human Engineering Skills, balanced schedules, mentorship, or character development? I'm here to help with questions about life balance, discipline, and personal growth."

# --- 5. DATA LOADING ---
dataset = load_dataset()
preprocessed_dataset = preprocess_dataset(dataset)
embedding_model = get_embedding_model() if USE_SEMANTIC_SEARCH else None
dataset_embeddings = generate_dataset_embeddings(dataset) if (USE_SEMANTIC_SEARCH and embedding_model) else None

# --- 6. SESSION STATE MANAGEMENT ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "audio_cache" not in st.session_state:
    st.session_state.audio_cache = {}

@st.cache_data(show_spinner=False)
def get_cached_audio_bytes(text: str) -> Optional[bytes]:
    """Generate and cache audio for text. Returns bytes."""
    if not text or not text.strip():
        return None
    audio_fp = text_to_speech(text)
    if audio_fp:
        return audio_fp.getvalue()
    return None

# --- 7. DISPLAY CHAT HISTORY ---
for i, msg in enumerate(st.session_state.chat_history):
    role = msg.get("role", "assistant")
    content = msg.get("content", "")
    with st.chat_message(role):
        st.write(content)
        if role == "assistant":
            msg_key = f"msg_{i}"
            if msg_key not in st.session_state.audio_cache:
                audio_bytes = get_cached_audio_bytes(content)
                if audio_bytes:
                    st.session_state.audio_cache[msg_key] = audio_bytes
            
            if msg_key in st.session_state.audio_cache:
                audio_bytes = st.session_state.audio_cache[msg_key]
                st.audio(audio_bytes, format="audio/mp3")

# --- 8. USER INPUT & GENERATION ---
st.markdown("---")
st.markdown("**💬 Ask your question (type or use microphone):**")

if "text_input_value" not in st.session_state:
    st.session_state.text_input_value = ""
if "voice_recording" not in st.session_state:
    st.session_state.voice_recording = False
if "pending_transcript" not in st.session_state:
    st.session_state.pending_transcript = ""
if "pending_clear_input" not in st.session_state:
    st.session_state.pending_clear_input = False
if "queued_question" not in st.session_state:
    st.session_state.queued_question = ""
if "processing_question" not in st.session_state:
    st.session_state.processing_question = False

# Apply transcript before creating the text_input widget to avoid
# StreamlitAPIException about mutating widget state after instantiation.
if st.session_state.pending_transcript:
    st.session_state.text_input_value = st.session_state.pending_transcript
    st.session_state.pending_transcript = ""

# Clear input safely before widget creation.
if st.session_state.pending_clear_input:
    st.session_state.text_input_value = ""
    st.session_state.pending_clear_input = False

col1, col2, col3 = st.columns([3.5, 1, 0.5])

user_input = ""
form_submit = False
with col1:
    if st.session_state.processing_question:
        st.text_input(
            "Type your question:",
            key="processing_input_dummy",
            value="",
            label_visibility="collapsed",
            placeholder="Processing... please wait",
            disabled=True
        )
        st.button(
            "Send",
            type="primary",
            use_container_width=False,
            key="send_question_btn_disabled",
            disabled=True
        )
    else:
        user_input = st.text_input(
            "Type your question:",
            key="text_input_value",
            label_visibility="collapsed",
            placeholder="Ask a question or click 🎤 to speak..."
        )
        form_submit = st.button(
            "Send",
            type="primary",
            use_container_width=False,
            key="send_question_btn"
        )

with col2:
    st.write("")  # Spacing
    if st.session_state.processing_question:
        st.button("🎤 Record", key="mic_recorder_disabled", disabled=True, use_container_width=True)
    else:
        audio = mic_recorder(
            start_prompt="🎤 Record",
            stop_prompt="⏹️ Stop",
            just_once=True,
            use_container_width=True,
            key="mic_recorder_input",
            format="wav"
        )
        if audio is not None:
            st.session_state.voice_recording = True
            audio_bytes = audio.get("bytes") if isinstance(audio, dict) else None
            if audio_bytes:
                with st.spinner("🎙️ Got your voice. Converting it into text..."):
                    transcript = audio_to_text(audio_bytes)
                if transcript:
                    st.session_state.pending_transcript = transcript.strip()
                    st.info("✅ Voice captured successfully. You can edit the text if needed, then press Send.")
                    st.session_state.voice_recording = False
                    st.rerun()
                else:
                    st.warning("I couldn't catch that clearly. Please try recording again.")
            st.session_state.voice_recording = False

with col3:
    st.write("")  # Spacing for alignment

# Queue submit first so input clears immediately and UI does not look duplicated.
if user_input and user_input.strip() and form_submit and not st.session_state.processing_question:
    st.session_state.queued_question = user_input.strip()
    st.session_state.processing_question = True
    st.session_state.pending_clear_input = True
    st.session_state.voice_recording = False
    st.rerun()

# Process queued question in a clean rerun.
if st.session_state.processing_question and st.session_state.queued_question:
    question_text = st.session_state.queued_question
    with st.spinner("🤖 Mentor is reviewing your question and preparing the best answer..."):
        print(f"✅ [FLOW] Processing question: '{question_text}'")
        st.session_state.chat_history.append({"role": "user", "content": question_text})

        context, dataset_answer = retrieve_context(
            question_text,
            dataset,
            k=3,
            use_semantic=USE_SEMANTIC_SEARCH,
            preprocessed_dataset=preprocessed_dataset,
            embedding_model=embedding_model,
            dataset_embeddings=dataset_embeddings,
        )
        print(f"✅ [FLOW] Context retrieved, dataset_answer length: {len(dataset_answer) if dataset_answer else 0}")

        answer = generate_answer(
            user_input=question_text,
            context=context,
            history=st.session_state.chat_history,
            model_name=MODEL_NAME,
            api_key=HF_API_KEY,
            dataset_answer=dataset_answer
        )

        if answer:
            print(f"✅ [FLOW] Answer generated: '{answer[:50]}...'")
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
        else:
            print(f"⚠️ [FLOW] No answer generated")

    st.session_state.queued_question = ""
    st.session_state.processing_question = False
    st.rerun()