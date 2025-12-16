#!/usr/bin/env python3
"""
Simple Flask server that provides the basic API endpoints for the GUI demo
"""
from flask import Flask, jsonify, request
from flask_cors import CORS
import argparse
import json

app = Flask(__name__)
CORS(app)

# Mock data for demo
MOCK_MODELS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
MOCK_RESULTS = []
MOCK_PROGRESS = []
IS_PROCESSING = False

@app.route('/graphql/', methods=['GET', 'POST'])
def graphql():
    """Handle GraphQL requests"""
    global IS_PROCESSING, MOCK_RESULTS, MOCK_PROGRESS
    
    query = request.args.get('query', '')
    
    if 'awake' in query:
        return jsonify({"data": {"awake": "Awake"}})
    
    if 'hello' in query:
        return jsonify({"data": {"hello": "World"}})
    
    if 'getModels' in query:
        return jsonify({"data": {"getModels": json.dumps(MOCK_MODELS)}})
    
    if 'loadModel' in query:
        return jsonify({"data": {"loadModel": "success"}})
    
    if 'startProcessing' in query:
        IS_PROCESSING = True
        MOCK_PROGRESS = ["Processing started", "Found 5 image files", "Processing image 1/5"]
        return jsonify({"data": {"startProcessing": "processing started"}})
    
    if 'stopProcessing' in query:
        IS_PROCESSING = False
        MOCK_PROGRESS.append("Processing stopped by user")
        return jsonify({"data": {"stopProcessing": "processing stopped"}})
    
    if 'getStatus' in query:
        status = {
            "is_processing": IS_PROCESSING,
            "results_count": len(MOCK_RESULTS)
        }
        return jsonify({"data": {"getStatus": json.dumps(status)}})
    
    if 'getProgress' in query:
        return jsonify({"data": {"getProgress": json.dumps(MOCK_PROGRESS)}})
    
    if 'getResults' in query:
        return jsonify({"data": {"getResults": json.dumps(MOCK_RESULTS)}})
    
    if 'calc' in query:
        return jsonify({"data": {"calc": "42"}})
    
    if 'echo' in query:
        return jsonify({"data": {"echo": "Hello from Python!"}})
    
    return jsonify({"data": {"error": "Unknown query"}})

@app.route('/graphiql/')
def graphiql():
    """GraphiQL interface for testing"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>GraphiQL</title>
        <style>body { font-family: Arial; padding: 20px; }</style>
    </head>
    <body>
        <h1>Python Backend is Running!</h1>
        <p>This is a simple GraphQL endpoint for the Face Recognition app demo.</p>
        <p>Available endpoints:</p>
        <ul>
            <li>awake - Check if server is awake</li>
            <li>hello - Say hello</li>
            <li>getModels - Get available models</li>
            <li>loadModel - Load a model</li>
            <li>startProcessing - Start processing</li>
            <li>stopProcessing - Stop processing</li>
            <li>getStatus - Get processing status</li>
            <li>getProgress - Get progress messages</li>
            <li>getResults - Get results</li>
        </ul>
    </body>
    </html>
    """

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apiport", type=int, default=5000)
    parser.add_argument("--signingkey", type=str, default="devkey")
    args = parser.parse_args()
    
    print(f"Starting simple Python backend on port {args.apiport}")
    print(f"GraphiQL available at: http://127.0.0.1:{args.apiport}/graphiql/")
    print(f"Signing key: {args.signingkey}")
    
    app.run(host='127.0.0.1', port=args.apiport, debug=True)