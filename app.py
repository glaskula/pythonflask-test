from flask import Flask
import os
from Chat import redis_url, index_name, schema_name, inference_server_url, embeddings, rds, llm, PROMPT_EN, QA_CHAIN_PROMPT_EN, PROMPT_SV, QA_CHAIN_PROMPT_SV, memory_Rephrase, memory, askQuestion 


app = Flask(__name__)

@app.route('/')
def hello():
    return redis_url

if __name__ == '__main__':
    port = os.environ.get('FLASK_PORT') or 8080
    port = int(port)

    app.run(port=port,host='0.0.0.0')