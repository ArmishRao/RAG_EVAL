import os
import json
import pickle
import re
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
import logging
import time
from pathlib import Path
from tqdm import tqdm

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DOCS_FOLDER = "data_docs"
# Moved off C: (which was critically low on space, causing repeated
# disk-full failures) onto D:, which has room to spare. Change this path
# if your D: drive layout is different.
VECTORSTORE_FOLDER = r"D:\legal-advisor-vectorstore"
EMBEDDING_MODEL = "mistral-embed"
# Reduced from 1500/200 -> smaller chunks carry less unrelated content each,
# which directly improves context_precision in RAGAS. Overlap kept at ~12-15%
# of chunk size to preserve continuity across boundaries.
CHUNK_SIZE = 600
CHUNK_OVERLAP = 75

EXCLUDED_FOLDERS = {
    'css', 'js', 'img', 'images', 'static', 'assets',
    'overrides', 'data', '_data', '__pycache__',
    'node_modules', 'venv', 'env', '.git',
    'build', 'dist', 'temp', 'tmp',
    'fonts',
    'code-samples',
    'snippets',
    'fleet',
    'managed-deep-agents-channels',
    'managed-deep-agents-connectors',
}

EXCLUDED_FILES = {
    'mkdocs.yml', 'mkdocs.yaml', 'docs.json',
    'style.css', '.css', '.js',
    'integration-downloads-table.js',
    'language-toggle.js',
    'comarketing.mdx',
    'publish-langchain.mdx',
    'OUTLINE', 'TIMELINE',
    '_llm-test.md',
    'newsletter.md', 'management.md',
    'project-generation.md', 'virtual-environments.md',
    'history-design-future.md',
    'translation-banner.md', 'translations.md',
}

KEEP_FILES = {
    'index.mdx',
    'use-these-docs.mdx',
    'agent-lifecycle.mdx',
    'build-overview.mdx',
    'playground.mdx',
    'environment-variables.md',
    'async.md',
    'alternatives.md',
    'benchmarks.md',
    'contributing.md',
    'editor-support.md',
    'agent-server-changelog.mdx',
    'agent-server-distributed-tracing.mdx',
    'agent-server-feedback.mdx',
    'abac.mdx',
    'admin.mdx',
    'administration-overview.mdx',
    'agent-auth.mdx',
    'add-auth-server.mdx',
    'add-human-in-the-loop.mdx',
    'add-metadata-tags.mdx',
    'access-current-span.mdx',
    'agent-server-changelog-link.mdx',
}


def load_markdown_documents(file_path: str) -> List[Document]:
    documents = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        sections = re.split(r'\n#{1,6}\s+', content)
        headings = re.findall(r'^#{1,6}\s+(.+)$', content, re.MULTILINE)
        if len(sections) > 1:
            for i, section in enumerate(sections[1:], 1):
                if len(section.strip()) > 50:
                    heading = headings[i-1] if i-1 < len(headings) else f"Section {i}"
                    metadata = {
                        'source_file': os.path.basename(file_path),
                        'heading': heading,
                        'source_type': 'markdown',
                        'file_path': file_path,
                        'doc_type': detect_doc_type(file_path)
                    }
                    documents.append(Document(
                        page_content=f"## {heading}\n\n{section}",
                        metadata=metadata
                    ))
        else:
            metadata = {
                'source_file': os.path.basename(file_path),
                'source_type': 'markdown',
                'file_path': file_path,
                'doc_type': detect_doc_type(file_path)
            }
            documents.append(Document(page_content=content, metadata=metadata))
    except Exception as e:
        logger.error(f"Error loading Markdown {file_path}: {e}")
    return documents


def load_mdx_documents(file_path: str) -> List[Document]:
    documents = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        content = re.sub(r'^import\s+.*?;?\s*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'^export\s+.*?;?\s*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'<[^>]+>', '', content)
        content = re.sub(r'```[\w]*\n', '```\n', content)
        content = re.sub(r'\n\s*\n', '\n\n', content)
        sections = re.split(r'\n#{1,6}\s+', content)
        headings = re.findall(r'^#{1,6}\s+(.+)$', content, re.MULTILINE)
        if len(sections) > 1:
            for i, section in enumerate(sections[1:], 1):
                if len(section.strip()) > 50:
                    heading = headings[i-1] if i-1 < len(headings) else f"Section {i}"
                    metadata = {
                        'source_file': os.path.basename(file_path),
                        'heading': heading,
                        'source_type': 'mdx',
                        'file_path': file_path,
                        'doc_type': detect_doc_type(file_path)
                    }
                    documents.append(Document(
                        page_content=f"## {heading}\n\n{section}",
                        metadata=metadata
                    ))
        else:
            metadata = {
                'source_file': os.path.basename(file_path),
                'source_type': 'mdx',
                'file_path': file_path,
                'doc_type': detect_doc_type(file_path)
            }
            documents.append(Document(page_content=content, metadata=metadata))
    except Exception as e:
        logger.error(f"Error loading MDX {file_path}: {e}")
    return documents


def load_rst_documents(file_path: str) -> List[Document]:
    documents = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        sections = re.split(r'\n={3,}\n', content)
        headings = re.findall(r'^(.+)\n={3,}$', content, re.MULTILINE)
        if len(sections) > 1:
            for i, section in enumerate(sections):
                if len(section.strip()) > 50:
                    heading = headings[i] if i < len(headings) else f"Section {i+1}"
                    metadata = {
                        'source_file': os.path.basename(file_path),
                        'heading': heading,
                        'source_type': 'rst',
                        'file_path': file_path,
                        'doc_type': detect_doc_type(file_path)
                    }
                    documents.append(Document(
                        page_content=f"# {heading}\n\n{section}",
                        metadata=metadata
                    ))
        else:
            metadata = {
                'source_file': os.path.basename(file_path),
                'source_type': 'rst',
                'file_path': file_path,
                'doc_type': detect_doc_type(file_path)
            }
            documents.append(Document(page_content=content, metadata=metadata))
    except Exception as e:
        logger.error(f"Error loading RST {file_path}: {e}")
    return documents


def load_text_documents(file_path: str) -> List[Document]:
    documents = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        if len(content.strip()) < 50:
            return documents
        sections = []
        if '\n\n' in content:
            parts = content.split('\n\n')
            if len(parts) > 3:
                for i, part in enumerate(parts):
                    if len(part.strip()) > 100:
                        sections.append({'content': part, 'section': i + 1})
        if not sections:
            section_pattern = r'(?i)(?:section|chapter|part)\s+(\d+[A-Za-z]?)\s*[:.]?\s*(.+?)(?=(?:(?:section|chapter|part)\s+\d+[A-Za-z]?|$))'
            matches = re.findall(section_pattern, content, re.DOTALL)
            for match in matches:
                section_content = match[1].strip()
                if section_content and len(section_content) > 50:
                    sections.append({'content': section_content, 'section': match[0]})
        if sections:
            for sec in sections:
                metadata = {
                    'source_file': os.path.basename(file_path),
                    'source_type': 'text',
                    'section': str(sec.get('section', 'Unknown')),
                    'file_path': file_path,
                    'doc_type': detect_doc_type(file_path)
                }
                documents.append(Document(page_content=sec['content'], metadata=metadata))
        else:
            metadata = {
                'source_file': os.path.basename(file_path),
                'source_type': 'text',
                'file_path': file_path,
                'doc_type': detect_doc_type(file_path)
            }
            documents.append(Document(page_content=content, metadata=metadata))
    except Exception as e:
        logger.error(f"Error loading text file {file_path}: {e}")
    return documents


def load_html_documents(file_path: str) -> List[Document]:
    documents = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
        for element in soup(["script", "style", "nav", "header", "footer"]):
            element.decompose()
        main_content = soup.find('main') or soup.find('article') or soup.find('div', {'class': 'content'})
        if main_content:
            content = main_content.get_text(separator='\n', strip=True)
        else:
            content = soup.get_text(separator='\n', strip=True)
        content = '\n'.join([line.strip() for line in content.split('\n') if line.strip()])
        title = soup.title.string if soup.title else os.path.basename(file_path)
        if len(content.strip()) > 100:
            sections = re.split(r'\n[A-Z][^\n]+\n[-=]+\n', content)
            if len(sections) > 1:
                for i, section in enumerate(sections):
                    if len(section.strip()) > 100:
                        metadata = {
                            'source_file': os.path.basename(file_path),
                            'title': str(title)[:200],
                            'section': i + 1,
                            'source_type': 'html',
                            'file_path': file_path,
                            'doc_type': detect_doc_type(file_path)
                        }
                        documents.append(Document(page_content=section.strip(), metadata=metadata))
            else:
                metadata = {
                    'source_file': os.path.basename(file_path),
                    'title': str(title)[:200],
                    'source_type': 'html',
                    'file_path': file_path,
                    'doc_type': detect_doc_type(file_path)
                }
                documents.append(Document(page_content=content, metadata=metadata))
    except Exception as e:
        logger.error(f"Error loading HTML {file_path}: {e}")
    return documents


def load_pdf_documents(file_path: str) -> List[Document]:
    documents = []
    try:
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                full_text = ""
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text and len(text.strip()) > 50:
                        full_text += text + "\n\n"
                        metadata = {
                            'source_file': os.path.basename(file_path),
                            'page_number': page_num + 1,
                            'source_type': 'pdf',
                            'file_path': file_path,
                            'doc_type': detect_doc_type(file_path)
                        }
                        documents.append(Document(page_content=text, metadata=metadata))
                if full_text.strip():
                    metadata = {
                        'source_file': os.path.basename(file_path),
                        'source_type': 'pdf',
                        'file_path': file_path,
                        'doc_type': detect_doc_type(file_path)
                    }
                    documents.append(Document(page_content=full_text, metadata=metadata))
        except ImportError:
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    full_text = ""
                    for page_num, page in enumerate(reader.pages):
                        text = page.extract_text()
                        if text and len(text.strip()) > 50:
                            full_text += text + "\n\n"
                            metadata = {
                                'source_file': os.path.basename(file_path),
                                'page_number': page_num + 1,
                                'source_type': 'pdf',
                                'file_path': file_path,
                                'doc_type': detect_doc_type(file_path)
                            }
                            documents.append(Document(page_content=text, metadata=metadata))
                    if full_text.strip():
                        metadata = {
                            'source_file': os.path.basename(file_path),
                            'source_type': 'pdf',
                            'file_path': file_path,
                            'doc_type': detect_doc_type(file_path)
                        }
                        documents.append(Document(page_content=full_text, metadata=metadata))
            except ImportError:
                logger.warning("PDF support requires: pip install pdfplumber or PyPDF2")
    except Exception as e:
        logger.error(f"Error loading PDF {file_path}: {e}")
    return documents


def load_json_documents(file_path: str) -> List[Document]:
    documents = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        filename = os.path.basename(file_path)
        items_to_process = []
        if isinstance(data, list):
            items_to_process = data
        elif isinstance(data, dict):
            nested_keys = ['data', 'documents', 'content', 'items', 'records', 'pages']
            found = False
            for key in nested_keys:
                if key in data and isinstance(data[key], list):
                    items_to_process = data[key]
                    found = True
                    break
            if not found:
                items_to_process = [data]
        for item in items_to_process:
            doc = process_json_item(item, filename)
            if doc:
                documents.append(doc)
    except Exception as e:
        logger.error(f"Error loading JSON {file_path}: {e}")
    return documents


def process_json_item(item: Dict, filename: str) -> Optional[Document]:
    content_keys = ['content', 'text', 'description', 'body', 'summary', 'docstring', 'markdown', 'page_content']
    content = None
    for key in content_keys:
        if key in item and item[key]:
            content = str(item[key])
            break
    if not content or len(content.strip()) < 50:
        return None
    metadata = {
        'source_file': filename,
        'source_type': 'json',
        'doc_type': detect_doc_type(filename)
    }
    for key, value in item.items():
        if key not in content_keys and value is not None:
            try:
                if isinstance(value, (str, int, float, bool)):
                    metadata[key] = str(value)[:500]
                elif isinstance(value, list):
                    metadata[key] = str(len(value)) + " items"
            except:
                pass
    return Document(page_content=content, metadata=metadata)


def detect_doc_type(file_path: str) -> str:
    path_lower = file_path.lower()
    if 'python' in path_lower or 'cpython' in path_lower:
        return 'python'
    elif 'langchain' in path_lower or 'langsmith' in path_lower:
        return 'langchain'
    elif 'fastapi' in path_lower:
        return 'fastapi'
    elif 'pytorch' in path_lower:
        return 'pytorch'
    elif 'tensorflow' in path_lower:
        return 'tensorflow'
    elif 'django' in path_lower:
        return 'django'
    elif 'flask' in path_lower:
        return 'flask'
    elif 'numpy' in path_lower:
        return 'numpy'
    elif 'pandas' in path_lower:
        return 'pandas'
    else:
        return 'other'


def should_include_file(file_path: str) -> bool:
    path_parts = file_path.split(os.sep)
    filename = os.path.basename(file_path)
    filename_lower = filename.lower()
    ext = os.path.splitext(filename)[1].lower()
    if filename in KEEP_FILES or filename_lower in KEEP_FILES:
        return True
    for part in path_parts:
        if part.lower() in EXCLUDED_FOLDERS:
            logger.debug(f"Skipping due to excluded folder '{part}': {file_path}")
            return False
    if filename in EXCLUDED_FILES or filename_lower in EXCLUDED_FILES:
        logger.debug(f"Skipping excluded file: {file_path}")
        return False
    allowed_extensions = {'.md', '.mdx', '.html', '.htm', '.rst', '.txt', '.text', '.pdf', '.json'}
    if ext not in allowed_extensions:
        logger.debug(f"Skipping unsupported extension '{ext}': {file_path}")
        return False
    try:
        if os.path.getsize(file_path) < 100:
            logger.debug(f"Skipping tiny file (<100 bytes): {file_path}")
            return False
    except:
        pass
    return True


def load_documents_from_folder(folder_path: str) -> List[Document]:
    documents = []
    if not os.path.exists(folder_path):
        logger.warning(f"Folder '{folder_path}' not found")
        return documents
    handlers = {
        '.md': load_markdown_documents,
        '.markdown': load_markdown_documents,
        '.mdx': load_mdx_documents,
        '.html': load_html_documents,
        '.htm': load_html_documents,
        '.rst': load_rst_documents,
        '.txt': load_text_documents,
        '.text': load_text_documents,
        '.pdf': load_pdf_documents,
        '.json': load_json_documents,
    }
    for root, dirs, files in os.walk(folder_path):
        dirs[:] = [d for d in dirs if d.lower() not in EXCLUDED_FOLDERS]
        for file in files:
            file_path = os.path.join(root, file)
            if not should_include_file(file_path):
                continue
            ext = os.path.splitext(file)[1].lower()
            if ext in handlers:
                logger.info(f"Loading: {file_path}")
                try:
                    docs = handlers[ext](file_path)
                    documents.extend(docs)
                    logger.info(f"   Added {len(docs)} documents from {file}")
                except Exception as e:
                    logger.error(f"   Error loading {file}: {e}")
    return documents


def load_all_documents() -> List[Document]:
    all_documents = []
    if os.path.exists(DOCS_FOLDER):
        logger.info(f"Loading from {DOCS_FOLDER}...")
        docs = load_documents_from_folder(DOCS_FOLDER)
        all_documents.extend(docs)
        logger.info(f"Loaded {len(docs)} documents from {DOCS_FOLDER}")
    else:
        logger.warning(f"Documentation folder '{DOCS_FOLDER}' not found!")
        logger.info("Please add documentation files to 'docs_data/' folder.")
        logger.info("   Supported formats: .md, .mdx, .html, .rst, .txt, .pdf, .json")
    return all_documents


def create_vectorstore(documents: List[Document]):
    """Create and save FAISS vector store, with rate-limit resilience and checkpointing."""

    if not documents:
        logger.error("No documents found to ingest.")
        return

    start_time = time.time()
    logger.info(f"Total Documents Loaded: {len(documents)}")

    logger.info("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Code-fence markers ("```") are prioritized ahead of generic
        # whitespace splits so code blocks are less likely to be cut
        # in half across chunk boundaries.
        separators=["\n\n", "\n", "```", ". ", " ", ""],
        keep_separator=True,
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks")

    # Assign a stable, unique chunk_id to every chunk. This is required so that
    # retrieval results (FAISS output) can later be matched against ground-truth
    # "expected_chunks" in the eval dataset to compute Recall@K, Precision@K,
    # Hit Rate, MRR, and NDCG. Format: <source_file>::<heading_or_section>::<index>
    # The trailing index disambiguates cases where the same file/heading was
    # split into multiple chunks.
    seen_counts = {}
    for chunk in chunks:
        source_file = chunk.metadata.get("source_file", "unknown")
        heading = chunk.metadata.get("heading") or chunk.metadata.get("section") or "full"
        base_key = f"{source_file}::{heading}"
        seen_counts[base_key] = seen_counts.get(base_key, 0) + 1
        chunk.metadata["chunk_id"] = f"{base_key}::{seen_counts[base_key]}"

    logger.info(f"Loading embedding model: {EMBEDDING_MODEL} (Mistral API)")
    embeddings = MistralAIEmbeddings(model=EMBEDDING_MODEL)

    logger.info("Building FAISS index...")

    # --- Rate-limit-resilient batching with checkpointing ---
    BATCH_SIZE = 25          # smaller batches = fewer tokens/request = fewer 429s
    BATCH_DELAY = 2.0        # seconds to sleep between successful batches
    MAX_RETRIES = 6          # per-batch retries on 429 before giving up
    MIN_FREE_BYTES = 2 * 1024 * 1024 * 1024  # stop before saving if <2GB free

    Path(VECTORSTORE_FOLDER).mkdir(exist_ok=True)
    checkpoint_path = Path(VECTORSTORE_FOLDER) / "_checkpoint.pkl"

    def has_enough_disk_space(path: str, min_bytes: int) -> bool:
        """Check free disk space on the drive containing `path`."""
        try:
            import shutil as _shutil
            usage = _shutil.disk_usage(Path(path).resolve().anchor)
            return usage.free >= min_bytes
        except Exception as e:
            logger.warning(f"Could not check disk space ({e}); proceeding anyway.")
            return True

    def atomic_save_local(vs, folder: str):
        """
        Save FAISS index to a temp folder first, then atomically replace the
        real folder's index files. This prevents a crash (e.g. disk full,
        power loss) from leaving index.faiss/index.pkl half-written and
        corrupted -- the previous good version is only overwritten AFTER
        the new save fully succeeds.
        """
        import shutil as _shutil

        folder_path = Path(folder)
        tmp_folder = folder_path.parent / f"{folder_path.name}__tmp_save"

        if tmp_folder.exists():
            _shutil.rmtree(tmp_folder)
        tmp_folder.mkdir(parents=True)

        # This is the risky write -- but it only touches the tmp folder,
        # so if it fails/crashes, the real vectorstore/ files are untouched.
        vs.save_local(str(tmp_folder))

        # Only reached if save_local fully succeeded above.
        for fname in ("index.faiss", "index.pkl"):
            src = tmp_folder / fname
            dst = folder_path / fname
            if src.exists():
                # os.replace is atomic on both Windows and POSIX
                os.replace(str(src), str(dst))

        _shutil.rmtree(tmp_folder, ignore_errors=True)

    vectorstore = None
    start_index = 0

    # Resume from a previous run if a checkpoint exists
    if checkpoint_path.exists():
        with open(checkpoint_path, "rb") as f:
            ckpt = pickle.load(f)
        start_index = ckpt["next_index"]
        vectorstore = FAISS.load_local(
            VECTORSTORE_FOLDER, embeddings, allow_dangerous_deserialization=True
        )
        logger.info(f"Resuming from checkpoint at chunk {start_index}/{len(chunks)}")

    batch_starts = list(range(start_index, len(chunks), BATCH_SIZE))

    for i in tqdm(batch_starts, desc="Embedding chunks", unit="batch"):
        batch = chunks[i:i + BATCH_SIZE]

        # Refuse to proceed if disk space is critically low -- better to pause
        # here (nothing lost, checkpoint intact) than to crash mid-save and
        # corrupt index.pkl like before.
        if not has_enough_disk_space(VECTORSTORE_FOLDER, MIN_FREE_BYTES):
            logger.error(
                "Less than 2GB free disk space. Stopping BEFORE saving to avoid "
                "corrupting the vectorstore. Free up space, then re-run this "
                "script to resume from the last checkpoint."
            )
            raise RuntimeError("Insufficient disk space -- stopped safely before save.")

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if vectorstore is None:
                    vectorstore = FAISS.from_documents(documents=batch, embedding=embeddings)
                else:
                    vectorstore.add_documents(batch)
                break
            except Exception as e:
                is_rate_limit = "429" in str(e) or "Too Many Requests" in str(e)
                if attempt == MAX_RETRIES:
                    logger.error(f"Batch at index {i} failed after {MAX_RETRIES} attempts: {e}")
                    if vectorstore is not None:
                        atomic_save_local(vectorstore, VECTORSTORE_FOLDER)
                    with open(checkpoint_path, "wb") as f:
                        pickle.dump({"next_index": i}, f)
                    logger.info(f"Progress saved. Re-run the script to resume from chunk {i}.")
                    raise
                wait = (10 * attempt) if is_rate_limit else (2 ** attempt)
                logger.warning(
                    f"Batch at index {i} failed (attempt {attempt}/{MAX_RETRIES}): {e}. "
                    f"Waiting {wait}s before retry..."
                )
                time.sleep(wait)

        time.sleep(BATCH_DELAY)

        # Checkpoint every batch so a crash never loses more than one batch.
        # Uses atomic_save_local so a crash/disk-full DURING this save cannot
        # corrupt the previously-good index.faiss/index.pkl.
        atomic_save_local(vectorstore, VECTORSTORE_FOLDER)
        with open(checkpoint_path, "wb") as f:
            pickle.dump({"next_index": i + BATCH_SIZE}, f)

    logger.info("Saving final vectorstore...")
    atomic_save_local(vectorstore, VECTORSTORE_FOLDER)

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    elapsed = time.time() - start_time
    logger.info(f"Vectorstore created in {elapsed:.1f} seconds")

    metadata = {
        "total_documents": len(documents),
        "total_chunks": len(chunks),
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "source_files": list(set(doc.metadata.get("source_file", "unknown") for doc in documents)),
        "doc_types": list(set(doc.metadata.get("doc_type", "unknown") for doc in documents)),
        "timestamp": time.time(),
    }
    with open("docs_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved.")


def main():
    print("=" * 70)
    print("  Programming Documentation Assistant - Data Ingestion")
    print("=" * 70)
    print("\nSupported Formats:")
    print("   Markdown (.md), MDX (.mdx), RST (.rst)")
    print("   Text (.txt), HTML (.html), PDF (.pdf), JSON (.json)")
    print("\nDocumentation Types Detected:")
    print("   Python, LangChain, FastAPI, PyTorch, and more!")
    print("=" * 70)

    print("\nLoading documentation...")
    documents = load_all_documents()

    if not documents:
        print("\nNo documents found to ingest.")
        print("\nPlease add documentation files to:")
        print("   'docs_data/' folder")
        print("\nSupported formats:")
        print("   - Python docs: .html or .pdf from docs.python.org")
        print("   - LangChain: .md, .mdx from langchain docs")
        print("   - FastAPI: .md from fastapi docs")
        print("   - PyTorch: .html or .rst from pytorch docs")
        print("\nExample directory structure:")
        print("   docs_data/")
        print("   |-- python/")
        print("   |   +-- *.html")
        print("   |-- langchain/")
        print("   |   |-- index.mdx")
        print("   |   +-- *.mdx")
        print("   |-- fastapi/")
        print("   |   +-- *.md")
        print("   +-- pytorch/")
        print("       +-- *.rst")
        return

    print(f"\nLoaded {len(documents)} documents")

    doc_types = {}
    for doc in documents:
        doc_type = doc.metadata.get('doc_type', 'unknown')
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

    print("\nDocument Types Summary:")
    for doc_type, count in doc_types.items():
        print(f"   {doc_type}: {count} documents")

    create_vectorstore(documents)

    print("\n" + "=" * 70)
    print("Done! Vector store created successfully.")
    print(f"\nFiles Created:")
    print(f"   {VECTORSTORE_FOLDER}/ - FAISS index")
    print(f"   docs_metadata.json - Document metadata")
    print("=" * 70)
    print("\nYou can now run the documentation assistant:")
    print("   python app.py")
    print("   or")
    print("   streamlit run app_ui.py")


if __name__ == "__main__":
    main()