import os
import glob
import pandas as pd

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATASET_FOLDER = "datasets"
VECTORSTORE_FOLDER = "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def read_csv_file(file_path):
    """Read CSV with better error handling."""
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    
    for encoding in encodings:
        try:
            # Try reading with quotechar and escapechar
            df = pd.read_csv(
                file_path, 
                encoding=encoding,
                quotechar='"',
                escapechar='\\',
                on_bad_lines='skip'
            )
            print(f" Loaded with {encoding}")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f" Error with {encoding}: {e}")
            continue
    
    raise Exception(f"Cannot read {file_path}")

def load_documents():
    documents = []
    csv_files = glob.glob(os.path.join(DATASET_FOLDER, "*.csv"))
    
    print(f"\nFound {len(csv_files)} datasets\n")
    
    for csv_file in csv_files:
        print("=" * 60)
        print("Reading:", os.path.basename(csv_file))
        
        df = read_csv_file(csv_file)
        
        print("Columns:", df.columns.tolist())
        
        filename = os.path.basename(csv_file)
        
        # Check required columns
        required_columns = [
            "Book",
            "Chapter Number",
            "Chapter Title",
            "Section",
            "Heading",
            "Defination"
        ]
        
        missing = [c for c in required_columns if c not in df.columns]
        
        if missing:
            print(f"Skipping {filename}")
            print("Missing Columns:", missing)
            continue
        
        for _, row in df.iterrows():
            if pd.isna(row["Defination"]):
                continue
            
            text = f"""
Book: {row['Book']}

Chapter Number: {row['Chapter Number']}

Chapter Title: {row['Chapter Title']}

Section: {row['Section']}

Heading: {row['Heading']}

Definition:
{row['Defination']}
"""
            
            metadata = {
                "book": row["Book"],
                "chapter_number": row["Chapter Number"],
                "chapter_title": row["Chapter Title"],
                "section": row["Section"],
                "heading": row["Heading"],
                "source_file": filename
            }
            
            documents.append(
                Document(
                    page_content=text,
                    metadata=metadata
                )
            )
    
    return documents

def create_vectorstore(documents):
    print("\nLoading embedding model...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )
    
    print("Creating FAISS index...")
    
    vectorstore = FAISS.from_documents(
        documents,
        embeddings
    )
    
    os.makedirs(VECTORSTORE_FOLDER, exist_ok=True)
    
    vectorstore.save_local(VECTORSTORE_FOLDER)
    
    print("\n Vector Store Saved")
    print(f" Total Documents: {len(documents)}")

def main():
    print("=" * 60)
    print("Pakistan Legal Advisor - Data Ingestion")
    print("=" * 60)
    
    documents = load_documents()
    
    if not documents:
        print("\n No documents found to ingest!")
        print("Please check your CSV files in the 'datasets' folder.")
        return
    
    print(f"\n Total Documents: {len(documents)}")
    
    create_vectorstore(documents)
    
    print("\n Done! Vector store created successfully.")

if __name__ == "__main__":
    main()