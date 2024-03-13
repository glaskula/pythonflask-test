from flask import Flask, request, jsonify
from Chat import llm, rds, PROMPT_EN, PROMPT_SV, memory_Rephrase, memory, QA_CHAIN_PROMPT_SV, QA_CHAIN_PROMPT_EN, askQuestion

app = Flask(__name__)

# Route for handling chat requests
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('question')
    if not question:
        return jsonify({'error': 'Missing question parameter'}), 400

    chat_response = askQuestion(question, llm, rds, PROMPT_SV, PROMPT_EN, memory_Rephrase, memory, QA_CHAIN_PROMPT_SV, QA_CHAIN_PROMPT_EN)
    return jsonify(chat_response)

if __name__ == '__main__':
    app.run(debug=True)
