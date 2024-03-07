from flask import Flask
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.redis import Redis
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceTextGenInference
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langdetect import detect
import os

redis_url = "redis://redisearch:6379"
index_name = "docs"
schema_name = "redis_schema.yaml"
inference_server_url = "http://hf-text-generation-inference-server.002ep-ai-xjob.svc.cluster.local:3000"

# Initialize embeddings and Redis
embeddings = HuggingFaceEmbeddings()

rds = Redis.from_existing_index(
    embeddings,
    redis_url=redis_url,
    index_name=index_name,
    schema=schema_name
)

llm = HuggingFaceTextGenInference(
    inference_server_url=inference_server_url,
    max_new_tokens=254,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.2,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)
app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello Worlds!"

if __name__ == '__main__':
    port = os.environ.get('FLASK_PORT') or 8080
    port = int(port)

    app.run(port=port,host='0.0.0.0')