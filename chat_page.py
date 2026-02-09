import streamlit as st
from typing import List, Dict, Any, Tuple, Generator
import ollama
from Utilities import CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL, LLM_MODEL, RAG_SYSTEM_PROMPT, run_evaluator_grounding_check
import logging
import os
import time
from rag_diagram import advanced_rag_diagram_html

# Set up logging for chat page
logger = logging.getLogger(__name__)

# Agentic RAG: run a single search on the collection and return context + chunk list for UI
def _run_search(collection, query: str, n_results: int = 4) -> Tuple[str, List[str]]:
    """Query the vector store and return (context_string, list_of_chunk_strings)."""
    results = collection.query(query_texts=[query], n_results=n_results)
    docs = results.get("documents") or results.get("results") or []
    if isinstance(docs, dict):
        docs = []
    context_items = []
    if docs:
        first = docs[0]
        context_items = [str(d) for d in (first if isinstance(first, list) else [first]) if d]
    context = "\n---\n".join(context_items) if context_items else ""
    return context, context_items


# Removed compatibility wrapper to avoid an extra function; calls to Streamlit rerun
# are handled inline at the call sites with try/except to support older versions.

def get_collection_metadata(collection) -> Dict:
    """Get metadata about a collection including size, creation time, etc."""
    try:
        if collection is None:
            return {"error": "No collection provided"}

        # count may raise if collection invalid
        count = collection.count()

        # Get a sample document to check ID format
        sample = collection.get(limit=1)
        has_content_ids = False
        try:
            ids = sample.get('ids', [])
            has_content_ids = any(isinstance(i, str) and i.startswith('d_') for i in ids)
        except Exception:
            has_content_ids = False

        # Safely get embedding function name
        emb_name = "unknown"
        try:
            emb_fn = getattr(collection, '_embedding_function', None)
            if emb_fn is not None:
                # Ollama-style embedding wrapper may have a .model attribute
                if hasattr(emb_fn, 'model'):
                    emb_name = f"ollama:{getattr(emb_fn, 'model') }"
                elif hasattr(emb_fn, 'name'):
                    name_attr = getattr(emb_fn, 'name')
                    emb_name = name_attr if isinstance(name_attr, str) else 'default'
                else:
                    emb_name = str(type(emb_fn))
        except Exception:
            emb_name = 'unknown'

        metadata = {
            "name": getattr(collection, 'name', 'unknown'),
            "count": count,
            "uses_content_hash": has_content_ids,
            "embedding_function": emb_name,
        }
        return metadata
    except Exception as e:
        # If collection is bad, try to return partial info
        return {
            "name": getattr(collection, 'name', 'unknown') if collection is not None else 'unknown',
            "error": str(e)
        }

def get_available_collections():
    """Get list of available ChromaDB collections with metadata."""
    try:
        import chromadb
        from chromadb.config import Settings
        from Utilities import CHROMA_PATH
        
        client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        collections = client.list_collections()
        
        # Enhance with metadata
        collection_info = []
        for col in collections:
            metadata = get_collection_metadata(col)
            collection_info.append((col, metadata))
            
        return client, collection_info
    except Exception as e:
        st.error(f"Failed to connect to ChromaDB: {e}")
        return None, []

def delete_collection(client, name: str) -> bool:
    """Safely delete a collection. Returns True if successful."""
    try:
        client.delete_collection(name)
        return True
    except Exception as e:
        st.error(f"Failed to delete collection {name}: {e}")
        return False


def chat_page(collection=None):
    """
    Displays a chat interface that uses RAG with the provided ChromaDB collection.
    If no collection is provided, tries to list available collections for selection.
    """
    # Set page config to reduce memory usage
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
    st.title("💬 Chat with Documents")
    
    # Get available collections
    client, collections = get_available_collections()
    
    if not collections:
        st.warning("No document collections found. Please upload and process documents in the Document Processing page first.")
        if st.button("Go to Document Processing"):
            # streamlit's switch_page only works for files in the pages/ directory.
            # Instead, just rerun the app which will return the user to the main flow.
            try:
                st.experimental_rerun()
            except Exception:
                try:
                    st.rerun()
                except Exception:
                    raise RuntimeError('Streamlit rerun API not available. Please upgrade Streamlit.')
        return
    
    # Collection management sidebar
    with st.sidebar:
        st.title("📚 Collections")
        
        # Refresh button
        col1, col2 = st.columns([4, 1])
        with col1:
            st.subheader("Available Collections")
        with col2:
            if st.button("🔄", help="Refresh collections list", key="refresh_collections_chat_page"):
                try:
                    st.rerun()
                except AttributeError:
                    try:
                        st.experimental_rerun()
                    except Exception:
                        raise RuntimeError('Streamlit rerun API not available. Please upgrade Streamlit.')
        
        # Model Settings Section (Ollama only)
        st.divider()
        st.subheader("⚙️ Model Settings (Ollama)")
        
        # Ollama: include Gemma 3 and other common models (must be pulled in Ollama)
        cpu_friendly_llm_models = [
            "gemma3:12b", "gemma3:4b", "gemma3:1b",
            "phi", "tinyllama", "gemma:2b", "qwen2:1.5b",
            "llama3.2:1b", "llama3.2:3b", "gemma2:2b", "mistral", "llama3.1:8b",
            "deepseek-r1:8b", "qwen3-coder:30b",
        ]
        _default_ollama = os.environ.get("LLM_MODEL", "gemma3:4b")
        if "selected_llm_model" not in st.session_state:
            st.session_state.selected_llm_model = _default_ollama
        if st.session_state.selected_llm_model not in cpu_friendly_llm_models:
            st.session_state.selected_llm_model = _default_ollama if _default_ollama in cpu_friendly_llm_models else cpu_friendly_llm_models[1]
        st.session_state.selected_llm_model = st.selectbox(
            "💬 Chat Model (Ollama)",
            options=cpu_friendly_llm_models,
            index=cpu_friendly_llm_models.index(st.session_state.selected_llm_model),
            help="Pick a model you have pulled in Ollama (e.g. gemma3:4b)",
            key="llm_model_selector"
        )
        model_sizes = {
            "gemma3:12b": "~7GB", "gemma3:4b": "~2.5GB", "gemma3:1b": "~1GB",
            "tinyllama": "~600MB", "phi": "~1.6GB", "gemma:2b": "~1.6GB",
            "qwen2:1.5b": "~900MB", "llama3.2:1b": "~1.3GB", "llama3.2:3b": "~2GB",
            "gemma2:2b": "~1.6GB", "mistral": "~4GB", "llama3.1:8b": "~4.7GB",
            "deepseek-r1:8b": "~5GB", "qwen3-coder:30b": "~18GB",
        }
        st.caption(f"📊 {model_sizes.get(st.session_state.selected_llm_model, '')}")
        st.caption("Models that don't support tools (e.g. gemma3) use single-shot RAG (one search).")
        
        with st.expander("ℹ️ Model Tips", expanded=False):
            st.markdown("**Ollama:** local, no API key required.")
            st.markdown("**Tool support:** For multi-search agentic RAG, use a model that supports tools (e.g. qwen3:4b, llama3.1:8b). Others use one search + answer.")
        
        # Collection selector if we don't have a specific collection
        active_collection = collection
        if active_collection is None:
            # Create selection options with metadata
            options = []
            for col, meta in collections:
                doc_count = meta.get('count', '?')
                label = f"{col.name} ({doc_count} docs)"
                options.append((col, label))
            
            if options:
                selected_idx = st.selectbox(
                    "Select a collection:",
                    range(len(options)),
                    format_func=lambda i: options[i][1]
                )
                active_collection = options[selected_idx][0]
                
                # Collection actions
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🗑️ Delete", help="Delete this collection"):
                        if st.session_state.get('confirm_delete') == active_collection.name:
                            # Second click - do the deletion
                            if delete_collection(client, active_collection.name):
                                st.success(f"Deleted collection {active_collection.name}")
                                st.session_state.pop('confirm_delete', None)
                                active_collection = None
                                try:
                                    st.experimental_rerun()
                                except Exception:
                                    try:
                                        st.rerun()
                                    except Exception:
                                        raise RuntimeError('Streamlit rerun API not available. Please upgrade Streamlit.')
                        else:
                            # First click - ask for confirmation
                            st.session_state['confirm_delete'] = active_collection.name
                            st.warning("Click delete again to confirm")
                
                with col2:
                    if st.button("📊 Details", help="Show collection details"):
                        st.session_state['show_details'] = not st.session_state.get('show_details', False)
                
                # Show collection details if requested
                if st.session_state.get('show_details'):
                    meta = next(m for _, m in collections if m['name'] == active_collection.name)
                    st.write("Collection Details:")
                    st.json(meta)
            
    if active_collection is None:
        st.error("Please select a collection to start chatting.")
        return

    # Initialize chat history in session state if not present
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Optional: show context when arriving from Home global search
    initial_query = st.session_state.pop("chat_initial_query", None)
    if initial_query:
        st.info(f"🔍 **You searched for:** “{initial_query[:80]}{'…' if len(initial_query) > 80 else ''}” — ask a follow-up below.")

    # RAG dependency: only requirement here is having a collection selected
    st.caption("Required for RAG: select a collection in the sidebar, then ask a question below.")
    meta = get_collection_metadata(active_collection)
    current_llm = st.session_state.get('selected_llm_model', LLM_MODEL)
    st.info(
        f"💬 **{meta['name']}** &nbsp;|&nbsp; "
        f"📑 {meta['count']} docs &nbsp;|&nbsp; "
        f"🤖 Ollama: {current_llm}"
    )
    
    # Debug: Show collection summary
    with st.expander("📊 Collection Debug Info", expanded=False):
        st.write(f"Collection name: {meta['name']}")
        st.write(f"Total documents: {meta['count']}")
        st.write(f"Embedding function: {meta['embedding_function']}")
        if meta['count'] > 0:
            st.write("Sample documents (first 3):")
            try:
                samples = active_collection.get(limit=3)
                if samples and 'documents' in samples:
                    for idx, doc in enumerate(samples['documents'][:3]):
                        st.text(f"Doc {idx + 1} preview: {doc[:200]}...")
            except Exception as e:
                st.error(f"Could not fetch sample documents: {e}")

    # Pipeline layout: Agentic RAG — agent can call search multiple times before answering
    st.markdown("**Agentic RAG architecture**")
    st.markdown(advanced_rag_diagram_html("both", -1), unsafe_allow_html=True)
    st.caption("The agent decides when and how to search (0 to many times); then answers. Your question → **Query**; **Response** in chat below.")
    st.divider()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        logger.info(f"User query: {prompt}")
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            if not hasattr(active_collection, "query") and client is not None:
                try:
                    active_collection = client.get_collection(active_collection.name)
                except Exception as e:
                    logger.error(f"Failed to fetch collection: {e}")
        except Exception as e:
            logger.error(f"Failed to query documents: {str(e)}", exc_info=True)
            st.error(f"Failed to query documents: {e}")
            return

        agent_steps: List[Dict[str, Any]] = []
        full_response: List[str] = []
        max_rounds = 5

        def search_fn_for_agent(query: str) -> dict:
            """Search the document knowledge base. Returns context and chunk count."""
            context, chunks = _run_search(active_collection, query, n_results=4)
            agent_steps.append({"type": "search", "query": query, "chunks": chunks})
            return {"context": context, "chunk_count": len(chunks)}

        model_name = st.session_state.get("selected_llm_model", LLM_MODEL)
        # Context isolation: system prompt (guardrails) separate from user content
        messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        final_text = ""
        try:
            for _ in range(max_rounds):
                response = ollama.chat(model=model_name, messages=messages, tools=[search_fn_for_agent])
                msg = getattr(response, "message", response) if not isinstance(response, dict) else response.get("message") or {}
                tool_calls = getattr(msg, "tool_calls", None) or (msg.get("tool_calls") if isinstance(msg, dict) else []) or []
                content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else "") or ""
                if not tool_calls:
                    final_text = (content or "").strip()
                    break
                tc_list = list(tool_calls) if hasattr(tool_calls, "__iter__") and not isinstance(tool_calls, dict) else [tool_calls]
                assistant_msg = {"role": "assistant", "content": content}
                if tc_list:
                    # #region agent log
                    import json as _json
                    _raw = getattr(getattr(tc_list[0], "function", None) or {}, "arguments", tc_list[0].get("function", {}).get("arguments") if isinstance(tc_list[0], dict) else {})
                    try:
                        _d = {"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H1", "location": "chat_page.py:tool_calls_args", "message": "arguments type before build", "data": {"type": type(_raw).__name__, "repr": str(_raw)[:80]}, "timestamp": int(__import__("time").time() * 1000)}
                        open("/Users/manasverma/Desktop/RAG Workshop/.cursor/debug.log", "a").write(_json.dumps(_d) + "\n")
                    except Exception:
                        pass
                    # #endregion
                    def _tc_args(t, i):
                        fn = getattr(t, "function", None) or (t.get("function") if isinstance(t, dict) else {})
                        name = (fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)) or "search_documents"
                        args_val = getattr(fn, "arguments", None) if hasattr(fn, "arguments") else (fn.get("arguments") if isinstance(fn, dict) else {})
                        if isinstance(args_val, str):
                            try:
                                args_val = _json.loads(args_val)
                            except Exception:
                                args_val = {}
                        if not isinstance(args_val, dict):
                            args_val = {}
                        return {"id": str(i), "function": {"name": name, "arguments": args_val}}
                    assistant_msg["tool_calls"] = [_tc_args(t, i) for i, t in enumerate(tc_list)]
                messages.append(assistant_msg)
                for tc in tc_list:
                    fn = getattr(tc, "function", None) or (tc.get("function") if isinstance(tc, dict) else {})
                    name = getattr(fn, "name", None) or (fn.get("name") if isinstance(fn, dict) else "search_documents")
                    args = getattr(fn, "arguments", None) or (fn.get("arguments") if isinstance(fn, dict) else {})
                    if isinstance(args, str):
                        import json
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}
                    q = (args or {}).get("query", "")
                    result = search_fn_for_agent(q)
                    result_str = (result.get("context") or "")[:8000] or "(no results)"
                    messages.append({"role": "tool", "content": result_str, "name": name})
            full_response = [final_text] if final_text else []
        except Exception as ollama_err:
            err_str = str(ollama_err).lower()
            if "does not support tools" in err_str or "status code: 400" in err_str:
                # Model doesn't support tool calling: fall back to single-shot RAG (one search + one answer)
                context, chunks = _run_search(active_collection, prompt, n_results=4)
                agent_steps.append({"type": "search", "query": prompt, "chunks": chunks})
                # Context isolation: system prompt + user question; context provided as the only augmentation
                augmented = f"Use ONLY the following context to answer. Do not make up information.\n\nContext:\n{context[:6000]}\n\nQuestion: {prompt}\n\nAnswer:"
                resp = ollama.chat(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": RAG_SYSTEM_PROMPT},
                        {"role": "user", "content": augmented},
                    ],
                )
                msg = getattr(resp, "message", resp) if not isinstance(resp, dict) else resp.get("message") or {}
                final_text = (getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else "") or "").strip()
                full_response = [final_text] if final_text else []
            else:
                raise

        # Build full retrieved context for Evaluator (all chunks from all searches)
        all_chunks = []
        for step in agent_steps:
            if step.get("type") == "search":
                all_chunks.extend(step.get("chunks", []))
        retrieved_context_for_eval = "\n---\n".join(all_chunks) if all_chunks else ""

        # Evaluator Agent: grounding check before showing answer
        is_grounded = True
        grounding_issues = ""
        if final_text and retrieved_context_for_eval:
            is_grounded, grounding_issues = run_evaluator_grounding_check(
                prompt, retrieved_context_for_eval, final_text, model_name
            )

        with st.chat_message("assistant"):
            with st.status("Agentic RAG pipeline", state="running", expanded=True) as status:
                diagram_ph = st.empty()
                diagram_ph.markdown(advanced_rag_diagram_html("query", 0), unsafe_allow_html=True)
                time.sleep(0.25)
                for i, step in enumerate(agent_steps):
                    if step.get("type") == "search":
                        st.write(f"🔍 Agent searched: **{step.get('query', '')[:60]}...** → **{len(step.get('chunks', []))}** chunks")
                        diagram_ph.markdown(advanced_rag_diagram_html("query", 3), unsafe_allow_html=True)
                        time.sleep(0.2)
                if agent_steps:
                    with st.expander("View retrieved context (all searches)", expanded=False):
                        for i, step in enumerate(agent_steps):
                            if step.get("type") == "search":
                                for j, ch in enumerate(step.get("chunks", [])[:3]):
                                    st.text_area(f"Search {i+1} Chunk {j+1}", (ch[:1500] + "..." if len(ch) > 1500 else ch), height=60, disabled=True, key=f"agent_ctx_{i}_{j}")
                diagram_ph.markdown(advanced_rag_diagram_html("query", 5), unsafe_allow_html=True)
                time.sleep(0.25)
                st.write("4️⃣ Agent answer")
                diagram_ph.markdown(advanced_rag_diagram_html("query", 6), unsafe_allow_html=True)
                answer = "".join(full_response)
                if answer:
                    st.markdown(answer)
                time.sleep(0.2)
                diagram_ph.markdown(advanced_rag_diagram_html("query", 7), unsafe_allow_html=True)
                st.write("5️⃣ Validation (Evaluator)")
                if is_grounded:
                    st.success("✅ Grounded: answer is supported by retrieved context.")
                else:
                    st.warning(f"⚠️ Unsupported claims or possible hallucination: {grounding_issues or 'Evaluator flagged ungrounded content.'}")
                diagram_ph.markdown(advanced_rag_diagram_html("query", 8), unsafe_allow_html=True)
                status.update(label="Done", state="complete")

        if full_response:
            st.session_state.messages.append({"role": "assistant", "content": "".join(full_response)})

    # Add a clear button
    if st.session_state.messages and st.button("Clear Chat History"):
        st.session_state.messages = []
        try:
            st.experimental_rerun()
        except Exception:
            try:
                st.rerun()
            except Exception:
                raise RuntimeError('Streamlit rerun API not available. Please upgrade Streamlit.')