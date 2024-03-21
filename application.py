from flask import Flask, render_template, request
 
import os

from Chat import redis_url, index_name, schema_name, inference_server_url, embeddings, rds, llm, PROMPT_EN, QA_CHAIN_PROMPT_EN, PROMPT_SV, QA_CHAIN_PROMPT_SV, memory_Rephrase, memory, askQuestion

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
async def ask():
    q = request.form.get('question')
    print(q)
    chatask = await askQuestion(q,llm, rds, PROMPT_SV, PROMPT_EN, memory_Rephrase, memory,QA_CHAIN_PROMPT_SV, QA_CHAIN_PROMPT_EN)
    return chatask['result']

