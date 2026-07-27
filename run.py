#!/usr/bin/env python
"""
Launcher script for Pakistan Legal Advisor
"""

import sys
import subprocess
import os

def main():
    print("=" * 70)
    print("⚖️  Pakistan Legal Advisor Launcher")
    print("=" * 70)
    print("\nChoose interface:")
    print("1.  Web Interface (Streamlit)")
    print("2.  Command Line Interface")
    print("3.  Run Tests")
    print("4.  Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        print("\n🚀 Starting Streamlit web interface...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app_ui.py"])
    
    elif choice == "2":
        print("\n🚀 Starting command line interface...")
        subprocess.run([sys.executable, "app.py"])
    
    elif choice == "3":
        print("\n🧪 Running tests...")
        subprocess.run([sys.executable, "test_web_search.py"])
    
    elif choice == "4":
        print("👋 Goodbye!")
        sys.exit(0)
    
    else:
        print("❌ Invalid choice!")

if __name__ == "__main__":
    main()