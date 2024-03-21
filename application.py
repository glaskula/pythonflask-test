from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

import os

from Chat import redis_url, index_name, schema_name, inference_server_url, embeddings, rds, llm, PROMPT_EN, QA_CHAIN_PROMPT_EN, PROMPT_SV, QA_CHAIN_PROMPT_SV, memory_Rephrase, memory, askQuestion
app = Flask(__name__)

CORS(app)
#CORS(app, resources={r"/ask": {"origins": "https://react-chat-bot-002ep-ai-xjob.apps.ocpext.gbgpaas.se"}})


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
async def ask():
    data = request.get_json()  # Parse JSON data from the request body
    question = data.get('question')  # Access the question field
    if question:
        # Process the question. Replace the next line with your processing logic.
        # For demonstration, I'm returning the question as received.
        chatask = await askQuestion(question,llm, rds, PROMPT_SV, PROMPT_EN, memory_Rephrase, memory,QA_CHAIN_PROMPT_SV, QA_CHAIN_PROMPT_EN)    
        # If you have an asynchronous function for processing, ensure it's awaited properly
        # e.g., chatask = await askQuestion(...)
        # Just for example, replace with your actual processing function.
        # Ensure your Flask app is set up to handle async routes if using `await`.
        
        return jsonify(chatask['result'])

    else:
        return jsonify({"error": "No question provided"}), 400

