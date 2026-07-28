#!/usr/bin/env python
"""
Flask API backend for Pakistan Legal Advisor
Serves the frontend with CORS support
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from app import ask_rag
import logging
import os

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for all routes

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('static', 'index.html')

@app.route('/api/ask', methods=['POST'])
def ask_question():
    """
    API endpoint for asking legal questions
    Expects JSON: {"question": "Your question here"}
    Returns JSON: {"Response": "Answer", "source_type": "local|web|fallback"}
    """
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({'error': 'Missing question field'}), 400
        
        question = data['question'].strip()
        if not question:
            return jsonify({'error': 'Question cannot be empty'}), 400
        
        logger.info(f"Received question: {question[:50]}...")
        
        # Call your existing RAG function
        result = ask_rag(question)
        
        response_data = {
            'Response': result.get('Response', 'No response generated.'),
            'source_type': result.get('source_type', 'local'),
            'sources': len(result.get('sources', [])),
        }
        
        logger.info(f"Response source: {response_data['source_type']}")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        return jsonify({
            'Response': f"I encountered an error: {str(e)}\n\nPlease try rephrasing your question.",
            'source_type': 'error'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Pakistan Legal Advisor API is running'})

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    return jsonify({
        'status': 'online',
        'version': '2.0',
        'features': ['local_db', 'web_search', 'fallback_kb']
    })

if __name__ == '__main__':
    # Run the Flask server
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print("=" * 60)
    print("  Pakistan Legal Advisor - API Server")
    print("=" * 60)
    print(f" Server running at: http://localhost:{port}")
    print(f"API endpoint: http://localhost:{port}/api/ask")
    print(f"Debug mode: {debug}")
    print("=" * 60)
    print("\nPress Ctrl+C to stop the server")
    
    app.run(host='0.0.0.0', port=port, debug=debug)