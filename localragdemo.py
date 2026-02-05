import os
from pathlib import Path

# Load .env from app directory (and optional GEMINI_ENV_PATH) so Gemini key is available
try:
    from dotenv import load_dotenv
    _env_dir = Path(__file__).resolve().parent
    load_dotenv(_env_dir / ".env", override=True)
    if os.environ.get("GEMINI_ENV_PATH"):
        load_dotenv(os.environ["GEMINI_ENV_PATH"], override=True)
except ImportError:
    pass

import streamlit as st
import PyPDF2
from Utilities import fixed_size_chunking, recursive_chunking, initialize_chroma_db, CHROMA_PATH, get_agentic_chunk_params
import logging
import time
from datetime import datetime
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # Also print to console
    ]
)

logger = logging.getLogger(__name__)
logger.info(f"Application started. Log file: {log_file}")

import ollama
try:
    import chromadb
except Exception:
    chromadb = None
from chat_page import chat_page  # import the chat interface
from audio_page import audio_processing_page  # import the audio processing page
from rag_diagram import advanced_rag_diagram_html, ui_node_header, ui_arrow

# #region agent log (optional debug; uses project logs dir so path is shareable)
def _dbg(payload):
    try:
        import json
        _log_dir = Path(__file__).resolve().parent / "logs"
        _log_dir.mkdir(parents=True, exist_ok=True)
        with open(_log_dir / "debug.log", "a") as f:
            f.write(json.dumps({**payload, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
    except Exception:
        pass
# #endregion

# Note: previously there was a compatibility wrapper for rerun here.
# To avoid an extra wrapper, call Streamlit's rerun APIs inline where needed
# and handle the AttributeError at the call site.

def show_debug_panel():
    """Shows a collapsible debug panel with system information and app state"""
    with st.sidebar.expander("🔧 Debug Panel", expanded=False):
        st.markdown("### System Info")
        if chromadb is not None:
            try:
                version = getattr(chromadb, '__version__', 'unknown')
            except Exception:
                version = 'unknown'
            st.text(f"ChromaDB Version: {version}")
        else:
            st.text("ChromaDB: not installed")
        st.text(f"Ollama Available: {hasattr(ollama, 'chat')}")
        
        st.markdown("### Logging")
        st.text(f"Log file: {log_file}")
        if st.button("View Recent Logs (last 50 lines)"):
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    recent_lines = lines[-50:] if len(lines) > 50 else lines
                    st.text_area("Recent Log Entries", ''.join(recent_lines), height=300)
            except Exception as e:
                st.error(f"Could not read log file: {e}")
        
        st.markdown("### Session State")
        st.json(dict(st.session_state))
        
        st.markdown("### Environment")
        st.text(f"ANONYMIZED_TELEMETRY: {os.environ.get('ANONYMIZED_TELEMETRY', 'Not Set')}")
        
        if st.button("Clear Session State"):
            logger.info("Session state cleared by user")
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            try:
                st.experimental_rerun()
            except Exception:
                try:
                    st.rerun()
                except Exception:
                    raise RuntimeError('Streamlit rerun API not available. Please upgrade Streamlit.')


# Mock document classifier (not currently used but preserved for future use)
def mock_document_classifier(text):
    classes = ['Category X', 'Category Y', 'Category Z']
    # Simple mock without numpy dependency
    from random import choice
    return choice(classes)

def _llm_for_agentic_chunking(prompt: str) -> str:
    """Call Gemini if key set, else Ollama. Used by get_agentic_chunk_params."""
    try:
        if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            import google.generativeai as genai
            genai.configure(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
            model = genai.GenerativeModel("gemini-1.5-flash")
            resp = model.generate_content(prompt)
            if resp and resp.text:
                return resp.text.strip()
    except Exception:
        pass
    try:
        resp = ollama.chat(model="phi", messages=[{"role": "user", "content": prompt}])
        return ((resp.get("message") or {}).get("content") or "").strip()
    except Exception:
        return ""

# Document processing page — layout mirrors RAG diagram: Documents → Chunking → Chunks → Embedding → Vector Store
def document_processing_page():
    st.title("📄 Document Processing")
    st.markdown(
        '<p class="rag-deps-intro">Pipeline below matches the RAG indexing flow. '
        'Fill <strong>Documents</strong> and <strong>Vector Store</strong>, then click Process.</p>',
        unsafe_allow_html=True
    )
    st.divider()

    # —— Node: Documents ——
    st.markdown(ui_node_header("Documents", "Required for RAG"), unsafe_allow_html=True)
    st.caption("Upload PDF and optional metadata")
    uploaded_doc = st.file_uploader("Upload Document File (PDF)", type=['pdf'], key='uploaded_doc')
    doc_source = st.text_input("Document Source", value="", key="doc_source", placeholder="Optional")
    doc_category = st.text_input("Document Category", value="", key="doc_category", placeholder="Optional")

    st.markdown(ui_arrow(), unsafe_allow_html=True)

    # —— Node: Chunking ——
    st.markdown(ui_node_header("Chunking", "Optional"), unsafe_allow_html=True)
    split_pdf_option = st.checkbox("Split PDF into smaller files", value=False, key="split_pdf_opt")
    if split_pdf_option:
        pages_per_file = st.number_input("Pages per file:", min_value=1, max_value=1000, value=10, step=1, key="pages_per_file")
    if split_pdf_option and uploaded_doc is not None:
        st.info(f"📄 Split mode: The PDF will be split into multiple files with {pages_per_file} pages each before processing.")
        if st.button("Split PDF and Save Files", key="split_pdf_btn"):
            with st.spinner('Splitting PDF...'):
                from PyPDF2 import PdfReader, PdfWriter
                import os
                
                try:
                    reader = PdfReader(uploaded_doc)
                    total_pages = len(reader.pages)
                    
                    # Calculate number of splits
                    num_splits = (total_pages + pages_per_file - 1) // pages_per_file
                    
                    # Get original filename without extension
                    original_filename = uploaded_doc.name
                    base_name = os.path.splitext(original_filename)[0]
                    
                    # Create output directory if it doesn't exist
                    output_dir = "split_pdfs"
                    os.makedirs(output_dir, exist_ok=True)
                    
                    split_files = []
                    
                    # Split the PDF
                    for split_num in range(1, num_splits + 1):
                        writer = PdfWriter()
                        
                        # Calculate page range for this split
                        start_page = (split_num - 1) * pages_per_file
                        end_page = min(split_num * pages_per_file, total_pages)
                        
                        # Add pages to this split
                        for page_num in range(start_page, end_page):
                            writer.add_page(reader.pages[page_num])
                        
                        # Generate output filename
                        output_filename = f"{base_name}-{split_num}.pdf"
                        output_path = os.path.join(output_dir, output_filename)
                        
                        # Write the split PDF
                        with open(output_path, 'wb') as output_file:
                            writer.write(output_file)
                        
                        split_files.append(output_filename)
                    
                    st.success(f"✅ Successfully split {total_pages} pages into {num_splits} PDF files!")
                    st.markdown(f"**Files saved in `{output_dir}/` folder:**")
                    for file in split_files:
                        st.text(f"  • {file}")
                    
                except Exception as e:
                    logger.error(f"Failed to split PDF: {str(e)}", exc_info=True)
                    st.error(f"Failed to split PDF: {e}")
        
    st.markdown(ui_arrow(), unsafe_allow_html=True)

    # —— Node: Chunks ——
    st.markdown(ui_node_header("Chunks", "Result after processing"), unsafe_allow_html=True)
    st.caption("Agentic chunking: the agent chooses chunk size and separator priority from your document. Chunk count appears after Process.")
    last_chunk_count = st.session_state.get("last_doc_chunk_count")
    if last_chunk_count is not None:
        st.success(f"✅ Last run: **{last_chunk_count}** chunks created.")
    else:
        st.info("👆 Upload a PDF and run Process below to see chunks.")

    st.markdown(ui_arrow(), unsafe_allow_html=True)

    # —— Node: Embedding ——
    st.markdown(ui_node_header("Embedding", "Ollama"), unsafe_allow_html=True)
    st.caption("Same embedding model as in Utilities (used automatically).")

    st.markdown(ui_arrow(), unsafe_allow_html=True)

    # —— Node: Vector Store ——
    st.markdown(ui_node_header("Vector Store", "Required for RAG"), unsafe_allow_html=True)
    st.caption("Choose existing collection or create new")
    collection_names = []
    try:
        import chromadb
        from chromadb.config import Settings
        client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False))
        cols = client.list_collections()
        # extract names safely
        for c in cols:
            try:
                name = c.name if hasattr(c, 'name') else str(c)
            except Exception:
                name = str(c)
            collection_names.append(name)
    except Exception:
        # chromadb not available or client couldn't connect
        collection_names = []

    new_collection_label = "-- Create new collection --"
    options = collection_names.copy()
    options.insert(0, new_collection_label)

    # show select + refresh button side-by-side
    # Use key so we can reset to "Create new" when user clears; default index=0 is "-- Create new collection --"
    sel_col, ref_col = st.columns([8, 1])
    with sel_col:
        _default_idx = 0
        selected_collection = st.selectbox(
            "Choose collection (existing or create new):",
            options,
            index=min(_default_idx, len(options) - 1) if options else 0,
            key="doc_page_collection_select",
        )
    with ref_col:
        if st.button("🔄", help="Refresh collections list", key="refresh_collections_doc_page"):
            # simple approach: rerun the page which will re-query collections
            try:
                st.rerun()
            except AttributeError:
                try:
                    st.experimental_rerun()
                except Exception:
                    raise RuntimeError('Streamlit rerun API not available. Please upgrade Streamlit.')

    new_collection_name = None
    chosen_collection_name = None
    is_new = selected_collection == new_collection_label
    if is_new:
        new_collection_name = st.text_input("New collection name:", value="")
        # validate new name immediately
        if new_collection_name:
            chosen_collection_name = new_collection_name.strip()
        else:
            chosen_collection_name = None
    else:
        chosen_collection_name = selected_collection

    # #region agent log
    _dbg({"sessionId": "debug-session", "runId": "post-fix", "hypothesisId": "H2", "location": "document_processing_page:before_validate", "message": "form state", "data": {"is_new": is_new, "selected_collection": selected_collection, "chosen_collection_name": chosen_collection_name, "collection_names": collection_names}})
    # #endregion

    # Validate the new collection name if provided (existing name = add to that collection)
    def _validate_collection_name(name: str, existing: list) -> tuple[bool, str]:
        if not name:
            return False, "Collection name cannot be empty"
        name = name.strip()
        if len(name) < 1 or len(name) > 64:
            return False, "Collection name must be between 1 and 64 characters"
        import re as _re
        if not _re.match(r"^[a-zA-Z0-9_-]+$", name):
            return False, "Use only letters, numbers, underscores (_), and hyphens (-). No spaces."
        if name in existing:
            return True, ""  # Allow: will add to existing collection
        return True, ""

    collection_name_valid = True
    collection_name_msg = ""
    if is_new:
        ok, msg = _validate_collection_name(chosen_collection_name or "", collection_names)
        collection_name_valid = ok
        collection_name_msg = msg
        # #region agent log
        _dbg({"sessionId": "debug-session", "runId": "post-fix", "hypothesisId": "H3", "location": "document_processing_page:after_validate", "message": "validation result", "data": {"ok": ok, "msg": msg, "show_error": bool(not ok and chosen_collection_name)}})
        # #endregion
        if not ok and chosen_collection_name:
            st.error(f"Invalid collection name: {msg}")
        elif chosen_collection_name and chosen_collection_name.strip() in collection_names:
            st.info(f"Adding to existing collection **{chosen_collection_name}**.")
        elif not chosen_collection_name:
            st.info("Enter a name for the new collection before processing")

    # Show success + actions when we have a recent processing result (e.g. after rerun when uploader may be empty)
    if not (uploaded_doc is not None) and st.session_state.get("doc_processed") and st.session_state.get("last_doc_chunk_count") is not None:
        n = st.session_state["last_doc_chunk_count"]
        coll_name = st.session_state.get("last_processed_collection_name") or "collection"
        st.success(f"✅ Document processed: **{n}** chunks added to **{coll_name}**. Go to Chat or clear and upload another.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Go to Chat", key="doc_go_chat_no_upload"):
                st.session_state["navigate_to_chat"] = True
                try:
                    st.rerun()
                except Exception:
                    try:
                        st.experimental_rerun()
                    except Exception:
                        pass
        with col2:
            if st.button("Clear & Upload Another", key="doc_clear_another_no_upload"):
                st.session_state['uploaded_doc'] = None
                st.session_state['doc_processed'] = False
                st.session_state.pop('last_doc_chunk_count', None)
                st.session_state.pop('last_processed_collection_name', None)
                st.session_state.pop('doc_page_collection_select', None)
                try:
                    st.rerun()
                except Exception:
                    try:
                        st.experimental_rerun()
                    except Exception:
                        pass
        st.divider()

    if uploaded_doc is not None:
        collection_ok = (chosen_collection_name and collection_name_valid) if is_new else bool(chosen_collection_name)
        st.markdown(
            f'<div class="rag-readiness">'
            f'<span class="rag-check">✓</span> PDF uploaded &nbsp;|&nbsp; '
            f'<span class="rag-check">{"✓" if collection_ok else "○"}</span> Collection chosen'
            f'</div>',
            unsafe_allow_html=True
        )
        disable_process = False
        if is_new and (not chosen_collection_name or not collection_name_valid):
            disable_process = True

        # After a successful run we rerun; show actions when we have a recent result (file still in uploader)
        if st.session_state.get("doc_processed") and st.session_state.get("last_doc_chunk_count") is not None:
            n = st.session_state["last_doc_chunk_count"]
            coll_name = st.session_state.get("last_processed_collection_name") or chosen_collection_name or "collection"
            st.success(f"✅ Document processed: **{n}** chunks added to **{coll_name}**. You can go to Chat or clear and upload another.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Go to Chat", key="doc_go_chat"):
                    st.session_state["navigate_to_chat"] = True
                    try:
                        st.rerun()
                    except Exception:
                        try:
                            st.experimental_rerun()
                        except Exception:
                            pass
            with col2:
                if st.button("Clear & Upload Another", key="doc_clear_another"):
                    st.session_state['uploaded_doc'] = None
                    st.session_state['doc_processed'] = False
                    st.session_state.pop('last_doc_chunk_count', None)
                    st.session_state.pop('last_processed_collection_name', None)
                    st.session_state.pop('doc_page_collection_select', None)
                    try:
                        st.rerun()
                    except Exception:
                        try:
                            st.experimental_rerun()
                        except Exception:
                            pass
            st.divider()

        if st.button("Extract text and process document", disabled=disable_process, type="primary"):
            diagram_ph = st.empty()
            with st.status("Document pipeline", state="running", expanded=True) as doc_status:
                diagram_ph.markdown(advanced_rag_diagram_html("indexing", 0), unsafe_allow_html=True)
                time.sleep(0.4)
                from PyPDF2 import PdfReader
                
                logger.info(f"Starting text extraction for file: {uploaded_doc.name}")
                logger.debug(f"File size: {uploaded_doc.size} bytes")
                logger.debug(f"Collection name: {chosen_collection_name}")
                logger.debug(f"Metadata - Source: {doc_source}, Category: {doc_category}")
                
                try:
                    # Reset file pointer to beginning
                    uploaded_doc.seek(0)
                    logger.debug("File pointer reset to beginning before reading")
                    
                    reader = PdfReader(uploaded_doc)
                    total_pages = len(reader.pages)
                    logger.info(f"PDF loaded. Total pages: {total_pages}")

                    # Real-time: live log and chunk list placeholders
                    live_log = st.empty()
                    live_chunks_container = st.empty()

                    # Agentic chunking: get sample from first page(s) and ask LLM for params
                    sample_text = ""
                    for p in reader.pages[:2]:
                        try:
                            sample_text += (p.extract_text() or "") + "\n"
                        except Exception:
                            pass
                    max_chunk_size, chunk_separators, priority_label = get_agentic_chunk_params(sample_text, llm_callback=_llm_for_agentic_chunking)
                    live_log.markdown(
                        f'<div class="pipeline-log">'
                        f'<span class="pipeline-line pipeline-done">🤖 Agentic chunking: max_size={max_chunk_size}, priority={priority_label}</span>'
                        f'</div>', unsafe_allow_html=True
                    )
                    time.sleep(0.3)

                    # Collect chunks from all pages using agent-chosen params
                    documents = []
                    for page_idx, page in enumerate(reader.pages):
                        diagram_ph.markdown(advanced_rag_diagram_html("indexing", 1), unsafe_allow_html=True)
                        live_log.markdown(
                            f'<div class="pipeline-log">'
                            f'<span class="pipeline-line pipeline-running">► Page {page_idx + 1}/{total_pages} → {len(documents)} chunks so far</span>'
                            f'</div>', unsafe_allow_html=True
                        )
                        time.sleep(0.15)
                        try:
                            page_text = page.extract_text()
                            if page_text:
                                if chunk_separators is not None:
                                    chunks = recursive_chunking(page_text, max_chunk_size, chunk_separators)
                                else:
                                    chunks = fixed_size_chunking(page_text, max_chunk_size)
                                documents.extend(chunks)
                                logger.debug(f"Page {page_idx + 1}: created {len(chunks)} chunks")
                                # Live chunk preview: show last few chunk previews
                                preview_lines = []
                                for i, c in enumerate(documents[-5:]):
                                    idx = len(documents) - 5 + i
                                    preview_lines.append(f"Chunk {idx + 1}: {c[:80].replace(chr(10), ' ')}...")
                                live_chunks_container.markdown(
                                    '<div class="pipeline-log" style="margin-top:0.5rem;">'
                                    + "".join(f'<span class="pipeline-line pipeline-done">{s}</span>' for s in preview_lines[-5:])
                                    + '</div>', unsafe_allow_html=True
                                )
                            else:
                                logger.warning(f"Page {page_idx + 1}: No text extracted")
                        except Exception as page_error:
                            logger.error(f"Error processing page {page_idx + 1}: {str(page_error)}", exc_info=True)

                    logger.info(f"Text extraction complete. Total chunks created: {len(documents)}")
                    st.session_state["last_doc_chunk_count"] = len(documents)
                    diagram_ph.markdown(advanced_rag_diagram_html("indexing", 2), unsafe_allow_html=True)
                    live_log.markdown(
                        f'<div class="pipeline-log">'
                        f'<span class="pipeline-line pipeline-done">✓ Pages 1–{total_pages} → {len(documents)} chunks (content-aware)</span>'
                        f'</div>', unsafe_allow_html=True
                    )
                    time.sleep(0.4)
                    if not documents:
                        logger.warning("No text could be extracted from any page of the PDF")
                        st.warning("No text could be extracted from the uploaded PDF.")
                        st.info("💡 This may happen if the PDF contains only images or scanned content. Try a text-based PDF.")
                    else:
                        st.info(f"Extracted {len(documents)} text chunks. Processing with ChromaDB...")
                        diagram_ph.markdown(advanced_rag_diagram_html("indexing", 3), unsafe_allow_html=True)
                        time.sleep(0.4)
                        logger.info(f"Starting ChromaDB processing for {len(documents)} chunks")
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        def update_progress(current: int, total: int):
                            progress = float(current) / float(total) if total > 0 else 0
                            progress_bar.progress(progress)
                            status_text.text(f"Processing chunks: {current}/{total}")
                            logger.debug(f"Progress: {current}/{total} chunks processed")

                        try:
                            # Initialize or get ChromaDB collection with progress tracking
                            # Create a metadata dict to attach to each chunk
                            doc_meta = {}
                            if doc_source:
                                doc_meta['source'] = doc_source
                            if doc_category:
                                doc_meta['category'] = doc_category
                            
                            logger.debug(f"Document metadata: {doc_meta}")

                            collection = initialize_chroma_db(
                                documents=documents,
                                batch_size=10,  # process 10 docs at a time
                                progress_callback=update_progress,
                                document_metadata=doc_meta if doc_meta else None,
                                collection_name=chosen_collection_name
                            )
                            logger.info(f"ChromaDB processing complete. Collection: {chosen_collection_name}")
                            diagram_ph.markdown(advanced_rag_diagram_html("indexing", 5), unsafe_allow_html=True)
                            time.sleep(0.4)
                            progress_bar.progress(1.0)
                            status_text.text(f"Successfully processed all {len(documents)} chunks!")
                            doc_status.update(label="Pipeline complete", state="complete")
                            
                            # Store collection in session state for chat page
                            st.session_state['chroma_collection'] = collection
                            st.session_state['doc_processed'] = True
                            st.session_state['last_processed_collection_name'] = chosen_collection_name
                            logger.debug("Session state updated with collection and doc_processed flag")

                            # Preview the first few chunks
                            with st.expander("Document chunks preview (first 5)", expanded=False):
                                for i, chunk in enumerate(documents[:5]):
                                    st.text(f"Chunk {i}:")
                                    st.text(chunk[:2500] + "..." if len(chunk) > 2500 else chunk)

                            # Rerun so the Chunks section at top of page shows the updated count
                            try:
                                st.rerun()
                            except Exception:
                                try:
                                    st.experimental_rerun()
                                except Exception:
                                    pass

                        except Exception as e:
                            logger.error(f"Failed to process documents: {str(e)}", exc_info=True)
                            st.error(f"Failed to process documents: {e}")
                            if st.button("Clear & Try Again"):
                                logger.info("User clicked 'Clear & Try Again'")
                                st.session_state['uploaded_doc'] = None
                                st.session_state['doc_processed'] = False
                                try:
                                    st.experimental_rerun()
                                except Exception:
                                    try:
                                        st.rerun()
                                    except Exception:
                                        raise RuntimeError('Streamlit rerun API not available. Please upgrade Streamlit.')
                
                except Exception as outer_e:
                    logger.error(f"Unexpected error during PDF processing: {str(outer_e)}", exc_info=True)
                    st.error(f"Unexpected error: {outer_e}")
                    st.info("Check the log file for detailed error information.")

def _do_clear_chromadb():
    """Delete all ChromaDB collections and clear related session state."""
    try:
        import chromadb
        from chromadb.config import Settings
        client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False))
        cols = client.list_collections()
        names = []
        for c in cols:
            try:
                name = c.name if hasattr(c, "name") else str(c)
                names.append(name)
            except Exception:
                pass
        for name in names:
            try:
                client.delete_collection(name)
                logger.info(f"Deleted ChromaDB collection: {name}")
            except Exception as e:
                logger.warning(f"Could not delete collection {name}: {e}")
        # Clear session state so app no longer references removed collection
        st.session_state["chroma_collection"] = None
        st.session_state.pop("doc_processed", None)
        st.session_state.pop("audio_processed", None)
        st.session_state.pop("last_doc_chunk_count", None)
        st.session_state.pop("last_processed_collection_name", None)
        st.session_state.pop("doc_page_collection_select", None)
        st.session_state["chromadb_cleared_message"] = True
        st.session_state["chromadb_cleared_count"] = len(names)
    except Exception as e:
        logger.error(f"Failed to clear ChromaDB: {e}", exc_info=True)
        st.session_state["chromadb_cleared_error"] = str(e)


# Main screen navigation
def main():
    st.set_page_config(
        page_title="Agentic RAG Workshop",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    # #region agent log
    _nav_choice = st.session_state.get("navigate_to_chat")
    _dbg({"sessionId": "debug-session", "runId": "post-fix", "hypothesisId": "H1", "location": "main:before_radio", "message": "navigate_to_chat in session", "data": {"navigate_to_chat": _nav_choice}})
    # #endregion
    st.sidebar.title("📁 Navigation")
    _nav_options = [
        "🏠 Home",
        "📄 Document Processing",
        "🎤 Audio Processing",
        "🎬 Video Processing",
        "💬 Chat"
    ]
    _default_index = 4 if st.session_state.pop("navigate_to_chat", False) else 0
    choice = st.sidebar.radio("Select a step:", _nav_options, index=min(_default_index, len(_nav_options) - 1), label_visibility="collapsed")
    # #region agent log
    _dbg({"sessionId": "debug-session", "runId": "post-fix", "hypothesisId": "H1", "location": "main:after_radio", "message": "current choice", "data": {"choice": choice}})
    # #endregion

    st.sidebar.divider()
    if st.sidebar.button("🗑️ Clear app history", help="Clear chat history and reset UI state (keeps your indexed collections)."):
        _keys_to_clear = [
            "messages", "last_doc_chunk_count", "last_audio_chunk_count",
            "doc_processed", "audio_processed", "navigate_to_chat",
            "confirm_delete", "show_details", "uploaded_doc", "uploaded_audio",
            "last_processed_collection_name", "doc_page_collection_select",
        ]
        for key in _keys_to_clear:
            st.session_state.pop(key, None)
        logger.info("App history cleared by user")
        try:
            st.rerun()
        except Exception:
            try:
                st.experimental_rerun()
            except Exception:
                pass

    with st.sidebar.expander("🗄️ Clear ChromaDB", expanded=False):
        if st.session_state.pop("chromadb_cleared_message", None):
            n = st.session_state.pop("chromadb_cleared_count", 0)
            st.sidebar.success(f"Deleted {n} collection(s).")
        err = st.session_state.pop("chromadb_cleared_error", None)
        if err:
            st.sidebar.error(f"Failed to clear ChromaDB: {err}")
        st.caption("Permanently delete all collections and their data. Cannot be undone.")
        if st.button("Delete all collections", key="clear_chromadb_btn", type="secondary"):
            st.session_state["confirm_clear_chromadb"] = True
            try:
                st.rerun()
            except Exception:
                try:
                    st.experimental_rerun()
                except Exception:
                    pass
        if st.session_state.get("confirm_clear_chromadb"):
            st.warning("Confirm: this will delete every collection.")
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("Yes, delete all", key="confirm_clear_chromadb_yes"):
                    _do_clear_chromadb()
                    st.session_state.pop("confirm_clear_chromadb", None)
                    try:
                        st.rerun()
                    except Exception:
                        try:
                            st.experimental_rerun()
                        except Exception:
                            pass
            with col_no:
                if st.button("Cancel", key="confirm_clear_chromadb_no"):
                    st.session_state.pop("confirm_clear_chromadb", None)
                    try:
                        st.rerun()
                    except Exception:
                        try:
                            st.experimental_rerun()
                        except Exception:
                            pass

    st.sidebar.divider()

    show_debug_panel()

    if choice == "🏠 Home":
        st.markdown(
            '<div class="home-hero">'
            '<h1>Agentic RAG Workshop</h1>'
            '<p class="home-hero-sub">Ingest documents, audio, or video — then chat with your content. The assistant decides when and how to search (Agentic RAG).</p>'
            '</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="home-cards">'
            '<div class="home-card"><div class="home-card-icon">📄</div><div class="home-card-title">Document Processing</div><div class="home-card-desc">Upload PDFs and add them to the knowledge base. Chunk, embed, and store in the vector DB.</div></div>'
            '<div class="home-card"><div class="home-card-icon">🎤</div><div class="home-card-title">Audio Processing</div><div class="home-card-desc">Upload audio (or video); we transcribe and index the content for retrieval.</div></div>'
            '<div class="home-card"><div class="home-card-icon">🎬</div><div class="home-card-title">Video Processing</div><div class="home-card-desc">Upload video; we extract audio, transcribe, and index for RAG.</div></div>'
            '<div class="home-card"><div class="home-card-icon">💬</div><div class="home-card-title">Chat</div><div class="home-card-desc">Select a collection and ask questions; the agent can search your docs multiple times (Agentic RAG).</div></div>'
            '</div>'
            '<div class="home-cta">Start by adding content in Document, Audio, or Video Processing, then open Chat to query.</div>',
            unsafe_allow_html=True
        )
    elif choice == "📄 Document Processing" or choice == "Document Processing":
        document_processing_page()
    elif choice == "🎤 Audio Processing" or choice == "Audio Processing":
        audio_processing_page(video_mode=False)
    elif choice == "🎬 Video Processing" or choice == "Video Processing":
        audio_processing_page(video_mode=True)
    elif choice == "💬 Chat" or choice == "Chat":
        # Pass the ChromaDB collection if documents have been processed
        collection = st.session_state.get('chroma_collection')
        chat_page(collection=collection)

# Main app entry point
if __name__ == "__main__":
    # Initialize session state for collection if needed
    if 'chroma_collection' not in st.session_state:
        st.session_state['chroma_collection'] = None

    # Call main() which handles navigation
    main()