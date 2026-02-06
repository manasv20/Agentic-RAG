import streamlit as st
import os
import logging
from datetime import datetime
from Utilities import recursive_chunking, initialize_chroma_db, CHROMA_PATH, fixed_size_chunking, get_agentic_sample_chars, get_agentic_chunk_params
import tempfile
import time
import wave
from rag_diagram import advanced_rag_diagram_html, ui_node_header, ui_arrow

# Configure logging
logger = logging.getLogger(__name__)

# Audio processing dependencies
try:
    import speech_recognition as sr
except ImportError:
    sr = None

try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None


# Video extensions supported for audio extraction (requires ffmpeg)
VIDEO_EXTENSIONS = {"mp4", "webm", "mov", "mkv", "avi"}


def extract_audio_from_video(video_file, video_format: str) -> str:
    """
    Extract audio track from video file and save as WAV.
    Uses pydub (which uses ffmpeg) to load the video and export audio only.
    
    Args:
        video_file: uploaded video file object
        video_format: original format (mp4, webm, mov, etc.)
    
    Returns:
        path to temporary WAV file
    """
    if AudioSegment is None:
        raise ImportError("pydub is required for video. Install: pip install pydub. Also need ffmpeg.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{video_format}") as tmp_in:
        tmp_in.write(video_file.read())
        input_path = tmp_in.name
    try:
        audio = AudioSegment.from_file(input_path, format=video_format)
        output_path = tempfile.mktemp(suffix=".wav")
        audio.export(output_path, format="wav")
        os.unlink(input_path)
        return output_path
    except Exception as e:
        if os.path.exists(input_path):
            os.unlink(input_path)
        raise e


def convert_audio_to_wav(audio_file, audio_format):
    """
    Convert audio file to WAV format for speech recognition.
    
    Args:
        audio_file: uploaded audio file object
        audio_format: original format (mp3, m4a, ogg, etc.)
    
    Returns:
        path to temporary WAV file
    """
    if AudioSegment is None:
        raise ImportError("pydub is required for audio format conversion. Install with: pip install pydub")
    
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_format}") as tmp_input:
        tmp_input.write(audio_file.read())
        input_path = tmp_input.name
    
    try:
        # Load audio file
        if audio_format == "mp3":
            audio = AudioSegment.from_mp3(input_path)
        elif audio_format == "m4a":
            audio = AudioSegment.from_file(input_path, format="m4a")
        elif audio_format == "ogg":
            audio = AudioSegment.from_ogg(input_path)
        elif audio_format == "flac":
            audio = AudioSegment.from_file(input_path, format="flac")
        else:
            audio = AudioSegment.from_file(input_path)
        
        # Convert to WAV
        output_path = tempfile.mktemp(suffix=".wav")
        audio.export(output_path, format="wav")
        
        # Clean up input file
        os.unlink(input_path)
        
        return output_path
    except Exception as e:
        # Clean up on error
        if os.path.exists(input_path):
            os.unlink(input_path)
        raise e


def transcribe_audio(audio_file_path, language="en-US"):
    """
    Transcribe audio file to text using speech recognition.
    
    Args:
        audio_file_path: path to WAV audio file
        language: language code for recognition (default: en-US)
    
    Returns:
        transcribed text string
    """
    if sr is None:
        raise ImportError("speech_recognition is required. Install with: pip install SpeechRecognition")
    
    recognizer = sr.Recognizer()
    
    with sr.AudioFile(audio_file_path) as source:
        # Adjust for ambient noise
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        
        # Record the audio
        audio_data = recognizer.record(source)
        
        try:
            # Use Google Speech Recognition (free)
            text = recognizer.recognize_google(audio_data, language=language)
            return text
        except sr.UnknownValueError:
            raise ValueError("Speech recognition could not understand the audio")
        except sr.RequestError as e:
            raise ConnectionError(f"Could not request results from speech recognition service: {e}")


def split_audio_by_duration(audio_file_path, max_duration_seconds=60):
    """
    Split long audio file into smaller chunks for better transcription.
    
    Args:
        audio_file_path: path to audio file
        max_duration_seconds: maximum duration for each chunk in seconds
    
    Returns:
        list of paths to audio chunk files
    """
    if AudioSegment is None:
        raise ImportError("pydub is required for audio splitting. Install with: pip install pydub")
    
    audio = AudioSegment.from_wav(audio_file_path)
    duration_ms = len(audio)
    max_duration_ms = max_duration_seconds * 1000
    
    if duration_ms <= max_duration_ms:
        return [audio_file_path]
    
    chunks = []
    for i in range(0, duration_ms, max_duration_ms):
        chunk = audio[i:i + max_duration_ms]
        chunk_path = tempfile.mktemp(suffix=f"_chunk_{i // max_duration_ms}.wav")
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
    
    return chunks


def audio_processing_page(video_mode: bool = False, llm_callback=None):
    """Main page for audio processing and storage in ChromaDB. llm_callback is used for agentic chunking (Agentic RAG)."""
    if video_mode:
        st.title("🎬 Video Processing")
        upload_label = "Upload Video File"
        upload_help = "MP4, WebM, MOV, MKV, AVI. We extract the audio track, transcribe it, and add to the knowledge base."
    else:
        st.title("🎤 Audio Processing")
        upload_label = "Upload Audio or Video File"
        upload_help = "Audio: WAV, MP3, M4A, OGG, FLAC. Video: MP4, WebM, MOV, MKV, AVI (audio track will be transcribed)."
    
    # RAG dependency clarity: what’s required to see RAG in Chat
    st.markdown(
        '<p class="rag-deps-intro">Agentic RAG: the agent chooses how to chunk the transcription for better retrieval. '
        'Fill <strong>Documents</strong> (file) and <strong>Vector Store</strong>, then click Process.</p>',
        unsafe_allow_html=True
    )
    st.divider()
    
    # Check dependencies
    if sr is None:
        st.error("❌ SpeechRecognition library not installed. Please install it to use audio processing.")
        st.code("pip install SpeechRecognition", language="bash")
        return
    
    if AudioSegment is None:
        st.warning("⚠️ PyDub not installed. Only WAV files will be supported. For MP3/M4A/OGG support, install pydub.")
        st.code("pip install pydub", language="bash")
    
    # —— Node: Documents (Audio/Video) ——
    doc_node_label = "Video" if video_mode else "Documents (Audio)"
    st.markdown(ui_node_header(doc_node_label, "Required for RAG"), unsafe_allow_html=True)
    st.caption("Upload file and optional metadata")
    supported_formats = ["wav"]
    if AudioSegment is not None:
        supported_formats.extend(["mp3", "m4a", "ogg", "flac"])
        supported_formats.extend(list(VIDEO_EXTENSIONS))
    if video_mode:
        supported_formats = [f for f in supported_formats if f in VIDEO_EXTENSIONS or f == "wav"]
    uploaded_audio = st.file_uploader(
        upload_label,
        type=supported_formats,
        key='uploaded_video' if video_mode else 'uploaded_audio',
        help=upload_help
    )
    audio_source = st.text_input("Audio Source", value="", key="audio_src", placeholder="Optional")
    audio_category = st.text_input("Audio Category", value="", key="audio_cat", placeholder="Optional")
    audio_speaker = st.text_input("Speaker/Host", value="", key="audio_speaker", placeholder="Optional")
    
    st.markdown(ui_arrow(), unsafe_allow_html=True)
    
    # —— Node: Chunking ——
    st.markdown(ui_node_header("Chunking", "Agentic — agent chooses size & separators"), unsafe_allow_html=True)
    st.caption("Optional — improves quality")
    col1, col2 = st.columns(2)
    
    with col1:
        language_options = {
            "English (US)": "en-US",
            "English (UK)": "en-GB",
            "Spanish": "es-ES",
            "French": "fr-FR",
            "German": "de-DE",
            "Italian": "it-IT",
            "Portuguese": "pt-BR",
            "Chinese (Mandarin)": "zh-CN",
            "Japanese": "ja-JP",
            "Korean": "ko-KR"
        }
        selected_language = st.selectbox("Language", list(language_options.keys()), key="audio_lang")
        language_code = language_options[selected_language]
    with col2:
        chunk_audio = st.checkbox("Split long audio files", value=True, key="chunk_audio", help="Split audio longer than 60s for better transcription")
    
    st.markdown(ui_arrow(), unsafe_allow_html=True)
    
    # —— Node: Chunks ——
    st.markdown(ui_node_header("Chunks", "Result after processing"), unsafe_allow_html=True)
    st.caption("Text chunks from transcription (1500 chars). Count appears after Process.")
    last_audio_chunks = st.session_state.get("last_audio_chunk_count")
    if last_audio_chunks is not None:
        st.success(f"✅ Last run: **{last_audio_chunks}** text chunks created.")
    else:
        st.info("👆 Upload a file and run Process below to see chunks.")
    
    st.markdown(ui_arrow(), unsafe_allow_html=True)
    
    # —— Node: Embedding ——
    st.markdown(ui_node_header("Embedding", "Ollama"), unsafe_allow_html=True)
    st.caption("Same embedding model as in Utilities (used automatically).")
    
    st.markdown(ui_arrow(), unsafe_allow_html=True)
    
    # —— Node: Vector Store ——
    st.markdown(ui_node_header("Vector Store", "Required for RAG"), unsafe_allow_html=True)
    st.caption("Choose existing collection or create new")
    collection_names = []
    if chromadb is not None:
        try:
            client = chromadb.PersistentClient(
                path=CHROMA_PATH,
                settings=Settings(anonymized_telemetry=False)
            )
            cols = client.list_collections()
            for c in cols:
                try:
                    name = c.name if hasattr(c, 'name') else str(c)
                except Exception:
                    name = str(c)
                collection_names.append(name)
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
    
    new_collection_label = "-- Create new collection --"
    options = collection_names.copy()
    options.insert(0, new_collection_label)
    
    # Collection selection with refresh button
    sel_col, ref_col = st.columns([8, 1])
    with sel_col:
        selected_collection = st.selectbox(
            "Choose collection (existing or create new):",
            options,
            index=0
        )
    with ref_col:
        if st.button("🔄", help="Refresh collections list", key="refresh_collections_audio_page"):
            try:
                st.rerun()
            except AttributeError:
                try:
                    st.experimental_rerun()
                except Exception:
                    raise RuntimeError('Streamlit rerun API not available. Please upgrade Streamlit.')
    
    # Handle new collection creation
    new_collection_name = None
    chosen_collection_name = None
    is_new = selected_collection == new_collection_label
    
    if is_new:
        new_collection_name = st.text_input("New collection name:", value="")
        if new_collection_name:
            chosen_collection_name = new_collection_name.strip()
        else:
            chosen_collection_name = None
    else:
        chosen_collection_name = selected_collection
    
    # Validate collection name
    def _validate_collection_name(name: str, existing: list) -> tuple:
        if not name:
            return False, "Collection name cannot be empty"
        name = name.strip()
        if len(name) < 1 or len(name) > 64:
            return False, "Collection name must be between 1 and 64 characters"
        import re
        # Allow letters, numbers, underscore, hyphen only (explicit)
        if not re.match(r"^[a-zA-Z0-9_-]+$", name):
            return False, "Use only letters (a–z, A–Z), numbers, underscores (_), and hyphens (-). No spaces."
        if name in existing:
            return False, "A collection with that name already exists"
        return True, ""
    
    collection_name_valid = True
    collection_name_msg = ""
    if is_new:
        ok, msg = _validate_collection_name(chosen_collection_name or "", collection_names)
        collection_name_valid = ok
        collection_name_msg = msg
        if not ok and chosen_collection_name:
            st.error(f"Invalid collection name: {msg}")
        elif not chosen_collection_name:
            st.info("Enter a name for the new collection before processing")
    
    # Process audio button
    if uploaded_audio is not None:
        # RAG readiness: show what’s required
        collection_ok = (chosen_collection_name and collection_name_valid) if is_new else bool(chosen_collection_name)
        st.markdown(
            f'<div class="rag-readiness">'
            f'<span class="rag-check">✓</span> File uploaded &nbsp;|&nbsp; '
            f'<span class="rag-check">{"✓" if collection_ok else "○"}</span> Collection chosen'
            f'</div>',
            unsafe_allow_html=True
        )
        file_size_mb = uploaded_audio.size / (1024 * 1024)
        st.info(f"📁 **{uploaded_audio.name}** ({file_size_mb:.2f} MB)")
        
        # Play audio preview if browser supports it (skip for video to avoid large payload)
        if not video_mode or file_size_mb < 50:
            st.audio(uploaded_audio)
        
        disable_process = False
        if is_new and (not chosen_collection_name or not collection_name_valid):
            disable_process = True
        
        btn_label = "Transcribe and process video" if video_mode else "Transcribe and process audio"
        if st.button(btn_label, disabled=disable_process, type="primary"):
            temp_files = []  # Track temporary files for cleanup
            
            try:
                uploaded_audio.seek(0)
                logger.info(f"Starting audio processing for file: {uploaded_audio.name}")
                file_format = uploaded_audio.name.split('.')[-1].lower()
                is_video = file_format in VIDEO_EXTENSIONS
                
                # SciFi-style pipeline: one status block, steps update in place
                pipeline_log = st.empty()
                diagram_placeholder = st.empty()
                
                def pipeline_step(step_num: int, label: str, done: bool = False):
                    return f'<span class="pipeline-line pipeline-{"done" if done else "running"}">[{step_num}] {"✓" if done else "►"} {label}</span>'
                
                with st.status("**Backend pipeline** — what’s happening now", state="running", expanded=True) as status:
                    steps_html = []
                    diagram_placeholder.markdown(advanced_rag_diagram_html("indexing", 0), unsafe_allow_html=True)
                    time.sleep(0.4)  # let client render so diagram is visible in real time
                    # Step 1: Extract / convert to WAV
                    steps_html.append(pipeline_step(1, "Extracting audio from source...", False))
                    pipeline_log.markdown('<div class="pipeline-log">' + "".join(steps_html) + "</div>", unsafe_allow_html=True)
                    if is_video:
                        wav_path = extract_audio_from_video(uploaded_audio, file_format)
                    elif file_format != "wav":
                        uploaded_audio.seek(0)
                        wav_path = convert_audio_to_wav(uploaded_audio, file_format)
                    else:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
                            tmp_wav.write(uploaded_audio.read())
                            wav_path = tmp_wav.name
                    temp_files.append(wav_path)
                    diagram_placeholder.markdown(advanced_rag_diagram_html("indexing", 1), unsafe_allow_html=True)
                    time.sleep(0.4)
                    steps_html[0] = pipeline_step(1, "Audio extracted to WAV", True)
                    pipeline_log.markdown('<div class="pipeline-log">' + "".join(steps_html) + "</div>", unsafe_allow_html=True)
                    
                    # Step 2: Split (optional)
                    audio_chunks = []
                    if chunk_audio:
                        steps_html.append(pipeline_step(2, "Splitting into time chunks...", False))
                        pipeline_log.markdown('<div class="pipeline-log">' + "".join(steps_html) + "</div>", unsafe_allow_html=True)
                        audio_chunks = split_audio_by_duration(wav_path, max_duration_seconds=60)
                        temp_files.extend([c for c in audio_chunks if c != wav_path])
                        steps_html[-1] = pipeline_step(2, f"Split into {len(audio_chunks)} chunk(s)", True)
                    else:
                        audio_chunks = [wav_path]
                        steps_html.append(pipeline_step(2, "Single segment (no split)", True))
                    pipeline_log.markdown('<div class="pipeline-log">' + "".join(steps_html) + "</div>", unsafe_allow_html=True)
                    
                    # Step 3: Transcribe — real-time: per-segment progress
                    transcriptions = []
                    live_transcript = st.empty()
                    steps_html.append(pipeline_step(3, f"Transcribing segment 1/{len(audio_chunks)}...", False))
                    for i, chunk_path in enumerate(audio_chunks):
                        steps_html[-1] = pipeline_step(3, f"Transcribing segment {i + 1}/{len(audio_chunks)}...", False)
                        pipeline_log.markdown('<div class="pipeline-log">' + "".join(steps_html) + "</div>", unsafe_allow_html=True)
                        diagram_placeholder.markdown(advanced_rag_diagram_html("indexing", 2), unsafe_allow_html=True)
                        time.sleep(0.12)
                        try:
                            text = transcribe_audio(chunk_path, language=language_code)
                            transcriptions.append(text)
                            steps_html[-1] = pipeline_step(3, f"Segment {i + 1}/{len(audio_chunks)}: {len(text)} chars", True)
                            seg_previews = "".join(
                                f'<span class="pipeline-line pipeline-done">Segment {j+1}: {t[:80].replace(chr(10), " ")}...</span>'
                                for j, t in enumerate(transcriptions[-4:])
                            )
                            live_transcript.markdown(
                                f'<div class="pipeline-log" style="margin-top:0.35rem; font-size:0.8rem;">{seg_previews}</div>',
                                unsafe_allow_html=True
                            )
                            pipeline_log.markdown('<div class="pipeline-log">' + "".join(steps_html) + "</div>", unsafe_allow_html=True)
                        except Exception as e:
                            logger.error(f"Failed to transcribe chunk {i+1}: {e}")
                    if not transcriptions:
                        status.update(label="Pipeline failed", state="error")
                        st.error("No audio could be transcribed. Check quality and try again.")
                        return
                    full_transcription = " ".join(transcriptions)
                    steps_html[-1] = pipeline_step(3, f"Transcribed {len(full_transcription)} chars total", True)
                    pipeline_log.markdown('<div class="pipeline-log">' + "".join(steps_html) + "</div>", unsafe_allow_html=True)
                    diagram_placeholder.markdown(advanced_rag_diagram_html("indexing", 2), unsafe_allow_html=True)
                    time.sleep(0.2)
                    
                    # Step 4: Agent decides how much transcript to sample, then chunking
                    total_chars = len(full_transcription)
                    steps_html.append(pipeline_step(4, "🤖 Agent is thinking… (choosing how much transcript to sample)", False))
                    pipeline_log.markdown('<div class="pipeline-log">' + "".join(steps_html) + "</div>", unsafe_allow_html=True)
                    with st.spinner("🤖 Agent is thinking… (choosing how much transcript to sample)"):
                        n_sample_chars, raw_sample_response = get_agentic_sample_chars(total_chars, llm_callback=llm_callback)
                    steps_html[-1] = pipeline_step(4, f"Agent chose to sample {n_sample_chars} chars (of {total_chars})", True)
                    pipeline_log.markdown('<div class="pipeline-log">' + "".join(steps_html) + "</div>", unsafe_allow_html=True)
                    steps_html.append(pipeline_step(5, "🤖 Agent is thinking… (choosing chunk size and separators)", False))
                    pipeline_log.markdown('<div class="pipeline-log">' + "".join(steps_html) + "</div>", unsafe_allow_html=True)
                    with st.spinner("🤖 Agent is thinking… (choosing chunk size and separators)"):
                        max_chunk_size, chunk_separators, priority_label, agent_sample_preview, agent_raw_response, agent_reasoning, _chunking_style = get_agentic_chunk_params(full_transcription[:n_sample_chars], llm_callback=llm_callback)
                    steps_html[-1] = pipeline_step(5, f"Agentic chunking: max_size={max_chunk_size}, priority={priority_label}", True)
                    pipeline_log.markdown('<div class="pipeline-log">' + "".join(steps_html) + "</div>", unsafe_allow_html=True)
                    with st.expander("🧠 Agent brain: how chunking was chosen", expanded=True):
                        st.caption("The agent chose how much transcript to sample, then chose chunking (Agentic RAG).")
                        st.markdown("**Sample size (agent chose):**")
                        st.info(f"**{n_sample_chars}** characters (of **{total_chars}** total).")
                        if raw_sample_response:
                            st.caption("Sample-size decision:")
                            st.code(raw_sample_response[:400] + ("…" if len(raw_sample_response) > 400 else ""), language=None)
                        st.markdown("**What the agent saw (sample):**")
                        st.text_area("Sample", agent_sample_preview or "(empty)", height=120, disabled=True, key="agent_brain_sample_audio")
                        st.markdown("**Agent's answer:**")
                        st.code(agent_raw_response, language=None)
                        if agent_reasoning:
                            st.markdown("**Agent's reasoning:**")
                            st.info(agent_reasoning)
                        st.markdown("**What we're using:**")
                        st.info(f"max_size={max_chunk_size} chars, priority={priority_label}" + (" (fixed-size chunks)" if chunk_separators is None else ""))
                    if chunk_separators is not None:
                        text_chunks = recursive_chunking(full_transcription, max_chunk_size, chunk_separators)
                    else:
                        text_chunks = fixed_size_chunking(full_transcription, max_chunk_size)
                    st.session_state["last_audio_chunk_count"] = len(text_chunks)
                    # Spinner UX: show progress, not a train of chunk lines
                    chunk_progress = st.progress(0, text="Chunking…")
                    with st.spinner("Chunking in progress…"):
                        total_c = len(text_chunks)
                        for ci in range(total_c):
                            chunk_progress.progress((ci + 1) / total_c if total_c else 1.0, text=f"Chunking… {ci + 1} of {total_c} chunks")
                            time.sleep(0.04)
                        chunk_progress.progress(1.0, text=f"✓ {total_c} chunks")
                    diagram_placeholder.markdown(advanced_rag_diagram_html("indexing", 3), unsafe_allow_html=True)
                    time.sleep(0.2)
                    steps_html[-1] = pipeline_step(5, f"Agentic: max_size={max_chunk_size}, priority={priority_label} → {len(text_chunks)} chunks", True)
                    pipeline_log.markdown('<div class="pipeline-log">' + "".join(steps_html) + "</div>", unsafe_allow_html=True)
                    
                    # Step 6: Embed & store
                    audio_meta = {}
                    if audio_source:
                        audio_meta['source'] = audio_source
                    if audio_category:
                        audio_meta['category'] = audio_category
                    if audio_speaker:
                        audio_meta['speaker'] = audio_speaker
                    audio_meta['content_type'] = 'video_transcription' if is_video else 'audio_transcription'
                    audio_meta['original_filename'] = uploaded_audio.name
                    steps_html.append(pipeline_step(6, "Embedding & storing in ChromaDB...", False))
                    diagram_placeholder.markdown(advanced_rag_diagram_html("indexing", 4), unsafe_allow_html=True)
                    time.sleep(0.4)
                    pipeline_log.markdown('<div class="pipeline-log">' + "".join(steps_html) + "</div>", unsafe_allow_html=True)
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    def update_progress(current: int, total: int):
                        progress_bar.progress(float(current) / float(total) if total > 0 else 0)
                        status_text.caption(f"Chunks: {current}/{total}")
                    collection = initialize_chroma_db(
                        documents=text_chunks,
                        batch_size=10,
                        progress_callback=update_progress,
                        document_metadata=audio_meta if audio_meta else None,
                        collection_name=chosen_collection_name
                    )
                    progress_bar.progress(1.0)
                    status_text.empty()
                    diagram_placeholder.markdown(advanced_rag_diagram_html("indexing", 5), unsafe_allow_html=True)
                    time.sleep(0.4)
                    steps_html[-1] = pipeline_step(6, f"Stored in collection «{chosen_collection_name}»", True)
                    pipeline_log.markdown('<div class="pipeline-log">' + "".join(steps_html) + "</div>", unsafe_allow_html=True)
                    status.update(label="Pipeline complete", state="complete")
                    
                    with st.expander("📝 Transcription preview", expanded=True):
                        st.text_area("Transcribed text", full_transcription, height=200, disabled=True)
                    
                    st.session_state['chroma_collection'] = collection
                    st.session_state['audio_processed'] = True
                    st.success("✅ Done. Go to Chat to query this content.")
                    
                    # Preview chunks
                    with st.expander("Text chunks preview (first 5)", expanded=False):
                        for i, chunk in enumerate(text_chunks[:5]):
                            st.text(f"Chunk {i}:")
                            st.text(chunk[:1500] + "..." if len(chunk) > 1500 else chunk)
                    
                    # Navigation buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Go to Chat"):
                            try:
                                st.experimental_rerun()
                            except Exception:
                                try:
                                    st.rerun()
                                except Exception:
                                    raise RuntimeError('Streamlit rerun API not available. Please upgrade Streamlit.')
                    with col2:
                        if st.button("Clear & Upload Another"):
                            st.session_state['uploaded_audio'] = None
                            st.session_state['audio_processed'] = False
                            st.session_state.pop('last_audio_chunk_count', None)
                            try:
                                st.experimental_rerun()
                            except Exception:
                                try:
                                    st.rerun()
                                except Exception:
                                    raise RuntimeError('Streamlit rerun API not available. Please upgrade Streamlit.')
                
            except Exception as e:
                logger.error(f"Failed to process audio: {str(e)}", exc_info=True)
                st.error(f"❌ Failed to process audio: {e}")
                st.info("Check the log file for detailed error information.")
            
            finally:
                # Clean up temporary files
                for temp_file in temp_files:
                    try:
                        if os.path.exists(temp_file):
                            os.unlink(temp_file)
                            logger.debug(f"Cleaned up temp file: {temp_file}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up temp file {temp_file}: {e}")


if __name__ == "__main__":
    audio_processing_page()
