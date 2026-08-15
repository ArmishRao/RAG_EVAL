"""
Complete documentation downloader - gets ALL pages at once
"""

import os
import zipfile
import subprocess
import shutil
from pathlib import Path
import requests

def download_and_extract_zip(url, dest_dir):
    """Download and extract a zip file"""
    os.makedirs(dest_dir, exist_ok=True)
    
    zip_path = dest_dir / "docs.zip"
    
    print(f"📥 Downloading: {url}")
    response = requests.get(url, stream=True)
    
    with open(zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"📂 Extracting to: {dest_dir}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)
    
    os.remove(zip_path)
    print(f"✅ Complete!")

def clone_repo(repo_url, dest_dir):
    """Clone a git repository"""
    print(f"📥 Cloning: {repo_url}")
    subprocess.run([
        "git", "clone", "--depth", "1", repo_url, str(dest_dir)
    ])
    print(f"✅ Cloned to: {dest_dir}")

def main():
    print("=" * 60)
    print("📚 Download Complete Documentation (ALL pages)")
    print("=" * 60)
    
    docs_dir = Path("docs_data")
    docs_dir.mkdir(exist_ok=True)
    
    # 1. Python Documentation - Download HTML archive
    print("\n🐍 Python Documentation (all pages)")
    download_and_extract_zip(
        "https://docs.python.org/3/archives/python-3.11-docs-html.zip",
        docs_dir / "python"
    )
    
    # 2. LangChain - Clone repository
    print("\n🦜 LangChain Documentation")
    clone_repo(
        "https://github.com/langchain-ai/langchain.git",
        docs_dir / "langchain"
    )
    
    # 3. FastAPI - Clone repository
    print("\n⚡ FastAPI Documentation")
    clone_repo(
        "https://github.com/tiangolo/fastapi.git",
        docs_dir / "fastapi"
    )
    
    # 4. PyTorch - Clone tutorials
    print("\n🔥 PyTorch Tutorials")
    clone_repo(
        "https://github.com/pytorch/tutorials.git",
        docs_dir / "pytorch"
    )
    
    print("\n" + "=" * 60)
    print("✅ All documentation downloaded!")
    print("📁 Location: docs_data/")
    print("=" * 60)

if __name__ == "__main__":
    main()