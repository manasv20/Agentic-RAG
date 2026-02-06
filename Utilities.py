# Utils for handling operations

import os
import hashlib
import re
import json
import requests
import ollama
import chromadb
from datetime import datetime as pydatetime, timezone, timedelta
from typing import List, Optional, Dict

# Defaults for Chroma / Ollama; can be overridden by environment variables
CHROMA_PATH = os.environ.get("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "localrag")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "phi")  # Use phi for embeddings as it's smaller
# Keep a conservative default LLM model name to avoid implicitly forcing a large model
# that may require excessive memory. Prefer setting LLM_MODEL via environment variable
# when you need a specific model. If you want to provide a list of fallback models
# (tried in order when memory errors occur), set the `LLM_FALLBACKS` env var to a
# comma-separated list.
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma3:4b")

###
## Document Splitting Functions
###

def split_text_on_newlines(text: str) -> List[str]:
    """Split text into paragraphs based on newlines."""
    return [p.strip() for p in text.split('\n\n') if p.strip()]

def split_text_on_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Basic sentence splitting - could be improved with nltk or spacy
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def fixed_size_chunking(document: str, chunk_size: int) -> List[str]:
    """Split document into fixed-size chunks.
    
    Args:
        document: Text to split
        chunk_size: Maximum number of characters per chunk
    """
    chunks = []
    for i in range(0, len(document), chunk_size):
        chunks.append(document[i:i + chunk_size])
    return chunks

def sliding_chunking(document: str, chunk_size: int, overlap: int) -> List[str]:
    """Split document into overlapping chunks.
    
    Args:
        document: Text to split
        chunk_size: Maximum number of characters per chunk
        overlap: Number of characters to overlap between chunks
    """
    chunks = []
    start = 0
    while start < len(document):
        end = min(start + chunk_size, len(document))
        chunks.append(document[start:end])
        start += chunk_size - overlap
    return chunks

def recursive_chunking(document: str, max_chunk_size: int, separators: Optional[List[str]] = None) -> List[str]:
    """Split document recursively using a hierarchy of separators.
    
    Args:
        document: Text to split
        max_chunk_size: Maximum number of characters per chunk
        separators: List of strings to use as separators, in priority order.
                   Defaults to ['\n\n', '.', ',']
    """
    if separators is None:
        separators = ['\n\n', '.', ',']
    
    def split_recursively(text: str, sep_index: int) -> List[str]:
        if sep_index >= len(separators) or len(text) <= max_chunk_size:
            return [text]
        
        sep = separators[sep_index]
        parts = text.split(sep)
        chunks = []
        current_chunk = ""
        
        for part in parts:
            if len(current_chunk) + len(part) + len(sep) <= max_chunk_size:
                if current_chunk:
                    current_chunk += sep + part
                else:
                    current_chunk = part
            else:
                if current_chunk:
                    chunks.extend(split_recursively(current_chunk, sep_index + 1))
                current_chunk = part
        
        if current_chunk:
            chunks.extend(split_recursively(current_chunk, sep_index + 1))
        
        return chunks
    
    return split_recursively(document, 0)


def get_agentic_sample_pages(total_pages: int, llm_callback=None) -> tuple:
    """Let the agent decide how many pages to sample from the start of the document.
    Returns (n_pages_to_sample, raw_llm_response). n_pages is clamped to [1, total_pages]."""
    if total_pages <= 0:
        return 1, ""
    prompt = """You are a RAG document advisor. A PDF has been uploaded with {} page(s).

Decide how many pages from the start you want to sample to later choose chunking (size, separators, style). You have full autonomy: use 1 page for a short doc, or more to capture variety (tables, sections, prose).

Reply with exactly one line: sample_pages=N
- N = integer number of pages to sample (from the start). Must be between 1 and {}.
Optional second line: reasoning=Your brief reason.

Example: sample_pages=3
Example: sample_pages=5
reasoning=Need enough to detect tables and paragraphs.
""".format(total_pages, total_pages)
    out_text = ""
    if callable(llm_callback):
        try:
            out_text = (llm_callback(prompt) or "").strip()
        except Exception:
            pass
    if not out_text:
        try:
            resp = ollama.chat(model=LLM_MODEL or "gemma3:4b", messages=[{"role": "user", "content": prompt}])
            out_text = (resp.get("message") or {}).get("content") or ""
        except Exception:
            pass
    raw_response = out_text or "(no response; using default)"
    n = max(1, min(total_pages, 10))  # default
    m = re.search(r"sample_pages\s*=\s*(\d+)", out_text, re.I)
    if m:
        try:
            n = max(1, min(total_pages, int(m.group(1))))
        except ValueError:
            pass
    return n, raw_response


def get_agentic_chunk_params(sample_text: str, llm_callback=None) -> tuple:
    """Agentic chunking: ask LLM for chunk size, separator priority, and optional chunking_style.
    Returns (max_chunk_size, separators or None, priority_label, sample_preview, raw_llm_response, agent_reasoning, chunking_style).
    chunking_style: paragraph|sentence|comma|fixed|table_row|table_section (table_* for menus/tables)."""
    sample = (sample_text or "")[:2800].strip()
    sample_preview = (sample[:500] + "..." if len(sample) > 500 else sample) if sample else ""
    # #region agent log
    try:
        import json
        _d = {"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H1", "location": "Utilities.py:get_agentic_chunk_params", "message": "entry", "data": {"len_sample": len(sample), "sample_preview_100": (sample[:100] if sample else "(empty)")}, "timestamp": int(__import__("time").time() * 1000)}
        open("/Users/manasverma/Desktop/RAG Workshop/.cursor/debug.log", "a").write(json.dumps(_d) + "\n")
    except Exception:
        pass
    # #endregion
    if not sample:
        # #region agent log
        try:
            _d = {"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H1", "location": "Utilities.py:early_return", "message": "early return empty sample", "data": {}, "timestamp": int(__import__("time").time() * 1000)}
            open("/Users/manasverma/Desktop/RAG Workshop/.cursor/debug.log", "a").write(json.dumps(_d) + "\n")
        except Exception:
            pass
        # #endregion
        return 2000, ["\n\n", ". ", "? ", "! ", ", "], "paragraph", "", "", "paragraph"
    prompt = """You are a RAG chunking advisor. Look at the document sample and decide how to chunk it for retrieval.

Think step by step: Is the text paragraph-heavy, sentence-heavy, a table/menu (rows with numbers like calories, protein), or very granular?

Output exactly two lines:
Line 1: max_chunk_size=NUMBER separator_priority=WORD chunking_style=STYLE
- NUMBER: integer between 500 and 3000 (character limit per chunk).
- WORD: one of paragraph, sentence, comma, fixed.
- STYLE: one of paragraph, sentence, comma, fixed, table_row, table_section. Use table_row when the sample looks like a table or menu with rows (e.g. dish name, Page No., Calories, Protein). Use table_section when there are clear section headers (e.g. Breakfast, Starters, Main Course). Otherwise use the same as WORD.
Line 2: reasoning=Your brief explanation (one short sentence).

Example for prose:
max_chunk_size=1500 separator_priority=paragraph chunking_style=paragraph
reasoning=Sample has clear paragraphs; 1500 chars keeps 1-2 paragraphs per chunk.

Example for a menu/table:
max_chunk_size=800 separator_priority=paragraph chunking_style=table_row
reasoning=Sample is a table with dish names and nutrition; row-based chunking preserves each entry.

Document sample:
"""
    prompt += sample[:2400] + "\n\nYour two-line output:"
    out_text = ""
    if callable(llm_callback):
        try:
            out_text = (llm_callback(prompt) or "").strip()
        except Exception:
            pass
    if not out_text:
        try:
            resp = ollama.chat(model=LLM_MODEL or "gemma3:4b", messages=[{"role": "user", "content": prompt}])
            out_text = (resp.get("message") or {}).get("content") or ""
        except Exception:
            pass
    # #region agent log
    try:
        import json
        _d = {"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H2", "location": "Utilities.py:after_llm", "message": "out_text from LLM", "data": {"len_out_text": len(out_text or ""), "out_preview": (out_text or "")[:80]}, "timestamp": int(__import__("time").time() * 1000)}
        open("/Users/manasverma/Desktop/RAG Workshop/.cursor/debug.log", "a").write(json.dumps(_d) + "\n")
    except Exception:
        pass
    # #endregion
    raw_llm_response = out_text or "Agent fallback: LLM did not return parseable output; applied sensible defaults."
    agent_reasoning = ""
    m_reason = re.search(r"reasoning\s*=\s*(.+)", out_text, re.I | re.DOTALL)
    if m_reason:
        agent_reasoning = m_reason.group(1).strip().split("\n")[0].strip()
    max_size = 2000
    priority = "paragraph"
    m = re.search(r"max_chunk_size\s*=\s*(\d+)", out_text, re.I)
    if m:
        max_size = max(500, min(3000, int(m.group(1))))
    m = re.search(r"separator_priority\s*=\s*(\w+)", out_text, re.I)
    if m:
        priority = m.group(1).lower()
    chunking_style = "paragraph"
    m_style = re.search(r"chunking_style\s*=\s*(\w+)", out_text, re.I)
    if m_style:
        cs = m_style.group(1).lower()
        if cs in ("paragraph", "sentence", "comma", "fixed", "table_row", "table_section"):
            chunking_style = cs
    sep_map = {
        "paragraph": ["\n\n", ". ", "? ", "! ", ", "],
        "sentence": [". ", "? ", "! ", "\n\n", ", "],
        "comma": [", ", "\n\n", ". ", "? ", "! "],
    }
    separators = sep_map.get(priority, sep_map["paragraph"])
    if priority == "fixed":
        return max_size, None, "fixed", sample_preview, raw_llm_response, agent_reasoning, chunking_style
    return max_size, separators, priority, sample_preview, raw_llm_response, agent_reasoning, chunking_style


def get_agentic_video_frame_params(duration_sec: float, llm_callback=None) -> tuple:
    """Agentic video framing: ask LLM for fps and max_frames given video duration.
    Returns (fps, max_frames, raw_llm_response, agent_reasoning)."""
    duration_sec = max(1.0, min(7200.0, float(duration_sec)))  # 1s to 2h
    prompt = f"""You are a video indexing advisor. A video is {duration_sec:.0f} seconds long. We will sample frames and embed them for visual search.

Think step by step: For a {duration_sec:.0f}s video, how many frames per second should we sample? How many total frames is enough for search without being redundant or too heavy?

Output exactly two lines:
Line 1: fps=NUMBER max_frames=NUMBER
- fps: between 0.25 and 2.0 (frames per second to sample).
- max_frames: integer between 10 and 200 (cap on total frames to embed).
Line 2: reasoning=Your brief explanation (one short sentence).

Example for a 120s video:
fps=1 max_frames=50
reasoning=1 fps gives one frame per second; 50 frames keeps indexing fast while covering key moments.

Your two-line output:"""
    out_text = ""
    if callable(llm_callback):
        try:
            out_text = (llm_callback(prompt) or "").strip()
        except Exception:
            pass
    if not out_text:
        try:
            resp = ollama.chat(model=LLM_MODEL or "gemma3:4b", messages=[{"role": "user", "content": prompt}])
            out_text = (resp.get("message") or {}).get("content") or ""
        except Exception:
            pass
    raw_llm_response = out_text or "Agent fallback: LLM did not return parseable output; applied sensible defaults."
    agent_reasoning = ""
    m_reason = re.search(r"reasoning\s*=\s*(.+)", out_text, re.I | re.DOTALL)
    if m_reason:
        agent_reasoning = m_reason.group(1).strip().split("\n")[0].strip()
    fps = 1.0
    max_frames = 50
    m = re.search(r"fps\s*=\s*([\d.]+)", out_text, re.I)
    if m:
        try:
            fps = max(0.25, min(2.0, float(m.group(1))))
        except ValueError:
            pass
    m = re.search(r"max_frames\s*=\s*(\d+)", out_text, re.I)
    if m:
        max_frames = max(10, min(200, int(m.group(1))))
    return fps, max_frames, raw_llm_response, agent_reasoning


def chunk_by_table_rows(text: str, max_chunk_size: int) -> List[str]:
    """Split table/menu-like text by rows (lines), merging into chunks up to max_chunk_size."""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return []
    chunks = []
    current = []
    current_len = 0
    for line in lines:
        line_len = len(line) + 1
        if current_len + line_len > max_chunk_size and current:
            chunks.append("\n".join(current))
            current = []
            current_len = 0
        current.append(line)
        current_len += line_len
    if current:
        chunks.append("\n".join(current))
    return chunks


def chunk_by_table_section(text: str, max_chunk_size: int) -> List[str]:
    """Split by sections (double newline or header-like lines), then merge up to max_chunk_size."""
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    if not blocks:
        return []
    chunks = []
    current = []
    current_len = 0
    for block in blocks:
        block_len = len(block) + 2
        if current_len + block_len > max_chunk_size and current:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(block)
        current_len += block_len
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def extract_chunk_metadata_llm(chunk_text: str, llm_callback) -> dict:
    """Use LLM to extract structured metadata from a menu/recipe chunk. Returns dict for Chroma (calories, protein, category, diet_type)."""
    if not callable(llm_callback):
        return {}
    prompt = """From this menu or recipe text, extract structured data. Return ONLY a valid JSON object, no other text.
Keys: "dish_name" (string or null), "calories" (integer or null), "protein" (integer or null), "category" (string e.g. Breakfast, Starters - Veg), "diet_type" (one of: "veg", "non-veg", "unknown").
If a value is not found use null. Example: {"dish_name": "Chicken Gravy", "calories": 320, "protein": 28, "category": "Main Course - Non-Veg", "diet_type": "non-veg"}

Text:
"""
    prompt += (chunk_text or "")[:1500] + "\n\nJSON:"
    try:
        out = (llm_callback(prompt) or "").strip()
        # Extract JSON (handle markdown code block)
        if "```" in out:
            m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", out, re.DOTALL)
            if m:
                out = m.group(1)
        m = re.search(r"\{[^{}]*\}", out)
        if m:
            data = json.loads(m.group(0))
            result = {}
            if "calories" in data and data["calories"] is not None:
                try:
                    result["calories"] = int(data["calories"])
                except (TypeError, ValueError):
                    pass
            if "protein" in data and data["protein"] is not None:
                try:
                    result["protein"] = int(data["protein"])
                except (TypeError, ValueError):
                    pass
            if data.get("category") and isinstance(data["category"], str):
                result["category"] = data["category"][:200]
            if data.get("diet_type") and isinstance(data["diet_type"], str):
                dt = data["diet_type"].lower().replace(" ", "")
                result["diet_type"] = "veg" if "veg" in dt and "non" not in dt else ("non-veg" if "non" in dt or "nonveg" in dt else "unknown")
            return result
    except Exception:
        pass
    return {}


def extract_metadata_batch(chunks: List[str], llm_callback, batch_size: int = 3) -> List[dict]:
    """Extract metadata for each chunk via LLM (in small batches for quality). Returns list of dicts, one per chunk."""
    results = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        for j, chunk in enumerate(batch):
            meta = extract_chunk_metadata_llm(chunk, llm_callback)
            results.append(meta)
    return results


def markdown_chunking(document: str) -> List[str]:
    """Split document at markdown headers.
    
    Args:
        document: Text in markdown format to split
    """
    chunks = []
    current_chunk = []
    
    for line in document.split('\n'):
        if line.startswith('#'):
            if current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
        current_chunk.append(line)
    
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return chunks

def get_ollama_embedding_function(model_name: str):
    """Create a ChromaDB-compatible embedding function using Ollama."""
    class OllamaEmbeddingFunction:
        def __init__(self, model: str):
            self.model = model
            self.name = f"ollama:{model}"

        def __call__(self, texts: List[str]):
            embeddings_list = []
            for text in texts:
                # Call embeddings without options to let Ollama manage resources
                resp = ollama.embeddings(model=self.model, prompt=text)
                if isinstance(resp, dict) and "embedding" in resp:
                    embeddings_list.append(resp["embedding"])
                else:
                    try:
                        embeddings_list.append(resp[0]["embedding"])
                    except Exception:
                        raise RuntimeError("Unexpected Ollama embeddings response format")
            return embeddings_list

    return OllamaEmbeddingFunction(model_name)

def initialize_chroma_db(documents: List[str], batch_size: int = 10, progress_callback=None, document_metadata: dict | None = None, metadatas_per_chunk: Optional[List[dict]] = None, collection_name: str | None = None):
    """
    Initialize a ChromaDB collection with the provided documents.
    Tries upsert first with different parameter orders for compatibility,
    falls back to add if upsert fails.

    Args:
        documents: list of strings to add (already chunked as desired)
        batch_size: how many documents to process in each batch (default 10)
        progress_callback: optional function called after each batch. Preferred signature:
            progress_callback(current:int, total:int)
        document_metadata: optional dict to attach as metadata to every document/chunk
        metadatas_per_chunk: optional list of dicts, one per chunk; merged with document_metadata per chunk
    Returns:
        chroma collection object
    """
    # Use provided collection_name or default
    target_collection = collection_name if collection_name else COLLECTION_NAME

    try:
        import chromadb
        from chromadb.config import Settings

        # Initialize the persistent client pointing to CHROMA_PATH with telemetry disabled
        client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False))

        # Prepare the ollama-backed embedding function
        ollama_embed_fn = get_ollama_embedding_function(EMBEDDING_MODEL)

        print(f"Checking for existing collection '{target_collection}'...")
        try:
            # Try to get or create collection
            collection = client.get_or_create_collection(
                name=target_collection,
                embedding_function=ollama_embed_fn
            )
        except Exception as e:
            try:
                # Fallback: try to get existing collection first
                collection = client.get_collection(name=target_collection)
            except Exception:
                # If that fails, create new collection
                collection = client.create_collection(
                    name=target_collection,
                    embedding_function=ollama_embed_fn
                )

        # Process documents in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # Generate unique IDs: content hash + global index to avoid duplicates when chunks are identical
            batch_ids = [
                f"d_{hashlib.sha256(doc.encode('utf-8')).hexdigest()}_{i + j}"
                for j, doc in enumerate(batch)
            ]
            
            # Prepare metadata for this batch: merge document_metadata with per-chunk metadatas if provided
            batch_metadatas = None
            if metadatas_per_chunk is not None and len(metadatas_per_chunk) == len(documents):
                batch_per = metadatas_per_chunk[i : i + len(batch)]
                base = (document_metadata or {}).copy()
                merged = []
                for j, meta in enumerate(batch_per):
                    m = base.copy()
                    if meta and isinstance(meta, dict):
                        for k, v in meta.items():
                            if v is not None and isinstance(v, (str, int, float, bool)):
                                m[k] = v
                    if m:
                        merged.append(m)
                batch_metadatas = merged if merged else None
            if batch_metadatas is None and document_metadata and len(document_metadata) > 0:
                batch_metadatas = [document_metadata.copy() for _ in batch]

            # Try to upsert the batch (omit metadatas when none to avoid "Expected metadata to be a non-empty dict, got {}")
            try:
                if batch_metadatas is not None:
                    collection.upsert(ids=batch_ids, documents=batch, metadatas=batch_metadatas)
                else:
                    collection.upsert(ids=batch_ids, documents=batch)
            except Exception:
                try:
                    if batch_metadatas is not None:
                        collection.add(documents=batch, ids=batch_ids, metadatas=batch_metadatas)
                    else:
                        collection.add(documents=batch, ids=batch_ids)
                except Exception as e:
                    raise RuntimeError(f"Failed to add batch to collection: {e}")

            # Update progress if callback provided
            if progress_callback:
                current_chunk = i + len(batch)
                total_chunks = len(documents)
                try:
                    progress_callback(current_chunk, total_chunks)
                except TypeError:
                    # If the callback doesn't accept two arguments, fall back to progress ratio
                    progress_callback(float(current_chunk) / total_chunks)

        return collection

    except ImportError:
        print("ChromaDB not available - falling back to in-memory collection")
        
        # Fallback in-memory implementation
        class InMemoryCollection:
            def __init__(self, documents: List[str]):
                self.docs = []
                self.ids = []
                self.embeddings = []
                self.metadatas = []
                
                # Process documents in batches
                for i in range(0, len(documents), batch_size):
                    batch = documents[i:i + batch_size]
                    
                    for j, doc in enumerate(batch):
                        # Add document
                        self.docs.append(doc)
                        
                        # Generate unique ID (content hash + index to avoid duplicates)
                        global_idx = i + j
                        doc_id = f"d_{hashlib.sha256(doc.encode('utf-8')).hexdigest()}_{global_idx}"
                        self.ids.append(doc_id)
                        
                        # Generate embedding
                        try:
                            # emb_options = {"no_gpu": True, "device": "cpu", "num_gpu": 0}
                            emb_options = {
                                "num_gpu": 0,
                                "num_thread": 8 # Example: use 8 CPU threads
                                }
                            resp = ollama.embeddings(model=EMBEDDING_MODEL, prompt=doc, options=emb_options)
                            if isinstance(resp, dict) and "embedding" in resp:
                                emb = resp["embedding"]
                            else:
                                emb = resp[0]["embedding"]
                            self.embeddings.append(emb)
                        except Exception as e:
                            raise RuntimeError(f"Failed to generate embedding: {e}")
                        
                        # Add metadata
                        self.metadatas.append(
                            document_metadata.copy() if document_metadata else {}
                        )
                    
                    # Update progress if callback provided
                    if progress_callback:
                        current_chunk = i + len(batch)
                        total_chunks = len(documents)
                        try:
                            progress_callback(current_chunk, total_chunks)
                        except TypeError:
                            # If the callback doesn't accept two arguments, fall back to progress ratio
                            progress_callback(float(current_chunk) / total_chunks)

            def query(self, query_texts: List[str], n_results: int = 3):
                # Only support single query for now
                query = query_texts[0]
                
                # Get query embedding
                try:
                    # emb_options = {"no_gpu": True, "device": "cpu", "num_gpu": 0}
                    emb_options = {
                            "num_gpu": 0,
                            "num_thread": 8 # Example: use 8 CPU threads
                        }
                    resp = ollama.embeddings(model=EMBEDDING_MODEL, prompt=query, options=emb_options)
                    if isinstance(resp, dict) and "embedding" in resp:
                        query_embedding = resp["embedding"]
                    else:
                        query_embedding = resp[0]["embedding"]
                except Exception as e:
                    raise RuntimeError(f"Failed to generate query embedding: {e}")

                # Cosine similarity function
                def cosine_similarity(a, b):
                    import math
                    dot = sum(x * y for x, y in zip(a, b))
                    norm_a = math.sqrt(sum(x * x for x in a))
                    norm_b = math.sqrt(sum(x * x for x in b))
                    if norm_a == 0 or norm_b == 0:
                        return 0.0
                    return dot / (norm_a * norm_b)

                # Calculate similarities and rank results
                similarities = [
                    cosine_similarity(query_embedding, doc_embedding) 
                    for doc_embedding in self.embeddings
                ]
                
                ranked_indices = sorted(
                    range(len(similarities)), 
                    key=lambda i: similarities[i],
                    reverse=True
                )[:n_results]
                
                # Gather results
                results = {
                    "documents": [[self.docs[i] for i in ranked_indices]],
                    "ids": [[self.ids[i] for i in ranked_indices]],
                    "metadatas": [[self.metadatas[i] for i in ranked_indices]],
                    "distances": [[1 - similarities[i] for i in ranked_indices]]
                }
                
                return results

        try:
            # Create and return in-memory collection
            collection = InMemoryCollection(documents)
            return collection
        except Exception as e:
            raise RuntimeError(f"Failed to create in-memory collection: {e}")