import os
import shutil
from pathlib import Path

def cleanup():
    """Remove legal-specific files and folders"""
    
    print("🧹 Cleaning up legal-specific files...")
    
    # Directories to remove
    dirs_to_remove = ['datasets', 'data', 'vectorstore']
    
    # Files to remove
    files_to_remove = [
        'custom_ragas_evaluator.py',
        'ragas_evaluate.py',
        'ragas_evaluator.py',
        'custom_ragas_evaluate.py',
        'evaluate.py',
        'evaluators.py',
        'judge_prompts.py',
        'legal_fallback.py',
        'test_questions.csv',
        'test_ragas.py',
        'test_duckduckgo.py',
        'test_hybrid_retrival.py',
        'test_search_method.py',
        'test_web_search.py',
        'Progress.json',
        'documents_metadata.json',
    ]
    
    # Remove directories
    for dir_name in dirs_to_remove:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"🗑️ Removed: {dir_name}/")
    
    # Remove files
    for file_name in files_to_remove:
        if os.path.exists(file_name):
            os.remove(file_name)
            print(f"🗑️ Removed: {file_name}")
    
    # Remove CSV files (except requirements.txt if it's CSV)
    for file in Path('.').glob('*.csv'):
        # Skip if it's a requirements file (just in case)
        if file.name != 'requirements.txt':
            try:
                file.unlink()
                print(f"🗑️ Removed: {file.name}")
            except:
                pass
    
    print("✅ Cleanup complete!")
    print()
    print("📝 Next steps:")
    print("1. Create docs_data directory: mkdir docs_data")
    print("2. Download documentation")
    print("3. Run: python ingest.py")

if __name__ == "__main__":
    confirm = input("⚠️ This will delete legal-specific files. Continue? (y/n): ")
    if confirm.lower() == 'y':
        cleanup()
    else:
        print("❌ Cleanup cancelled.")