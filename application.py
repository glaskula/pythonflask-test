from flask import Flask, jsonify, make_response, render_template, request
from flask_cors import CORS

import os

from ChatOld import redis_url, index_name, schema_name, inference_server_url, embeddings, rds, llm, PROMPT_EN, QA_CHAIN_PROMPT_EN, PROMPT_SV, QA_CHAIN_PROMPT_SV, memory_Rephrase, memory, askQuestion
app = Flask(__name__)

CORS(app)
#CORS(app, resources={r"/ask": {"origins": "https://react-chat-bot-002ep-ai-xjob.apps.ocpext.gbgpaas.se"}})


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
async def ask():
    data = request.get_json()
    question = data.get('question')
    history = data.get('history', [])  # Assuming history is sent as a list of messages
    language = data.get('language')
    if question:
        chatask = await askQuestion(question, history, language) 
        # Assuming chatask['result'] can be directly serialized with jsonify
        response = make_response(jsonify({"response": chatask['result']}))
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
    else:
        return jsonify({"error": "No question provided"}), 400


