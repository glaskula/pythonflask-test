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
from datetime import datetime
from langdetect import detect
import httpx
import asyncio

redis_url = "redis://redisearch:6379"
index_name = "docs"
schema_name = "redis_schema.yaml"
inference_server_url = "http://95.80.38.172:3001"

# Initialize embeddings and Redis
embeddings = HuggingFaceEmbeddings()

rds = Redis.from_existing_index(
    embeddings,
    redis_url=redis_url,
    index_name=index_name,
    schema=schema_name
)


def get_formatted_date():
    # Get today's date
    today = datetime.now()
    # Format the date as a string (for example: "22 February, 2024")
    formatted_date = today.strftime("%d %B, %Y")
    return formatted_date

def get_context(query):
    retriever = rds.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 2, "score_threshold": 0.3})
    docs = retriever.get_relevant_documents(query)
    # Use the dot notation to access the page_content attribute
    context_string = ', '.join([doc.page_content for doc in docs])
    return context_string

# Define a function to clean message labels
def clean_message_label(message):
    # Patterns to check for at the start of the message
    patterns = ["Human: ", "AI: "]
    for pattern in patterns:
        if message.startswith(pattern):
            # Remove the pattern and return the cleaned message
            return message[len(pattern):]
    return message

async def askQuestion(question, history, language):
    print(f"The language of the text is: {language}")
    print("Historiy:", history)


    # Process each message pair and save it to memory
    for i in range(0, len(history) - 1, 2):  # Iterate through the history in steps of 2
        input_message = history[i]['text']  # User's message as input
        output_message = history[i+1]['text']  # AI's message as output
        

    async with httpx.AsyncClient(timeout=600.0) as client:  #time out
        if language == "sv":
            prompt_version_1 = f"""
            Du kommer att få konversationshistorik och en uppföljningsfråga. Din uppgift är att omformulera uppföljningsfrågan till att vara en fristående fråga baserad på konversationshistoriken. Om möjligt markera vilken typ som frågan handlar om. 
            De olika typerna är Event, Guide och Place.
            Om inget annat nämns, anta att frågan handlar om Göteborg. Om frågan verkar vara orelaterad till konversationens historik, ändra inte uppföljningsfrågan.
            Ändra så lite som möjligt samtidigt som sammanhanget bevaras.
            Dagens datum är: {get_formatted_date()}
            SVARA ALLTID PÅ SVENSKA! Du kommer straffas annars.

            """.strip()
        else:
            prompt_version_1 = f"""
            You will be provided with a conversation history and an Input from the user, which could either be a question or not. Your task is to:
            If the input is not a question then leave the input as is without attempting to rewrite it into a question. The input must be maintained in its original form to ensure the conversation's continuity.
            If the input is a question then rephrase the question to make it a standalone question, based on the conversation history.
            Change as little as possible while maintaining context and relevance. If the input seems unrelated to the conversation history, maintain the input as is.         Add absolutly NOTHING else, no parenthesis or explanation, only give rephrased input.
            If not specified, assume the context is about Gothenburg.
            If needed, todays date is: {get_formatted_date()}.
            """.strip()

        # First API call with the rephrased question prompt
        response1 = await client.post(
            "http://localhost:3001/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "messages": [
                    {"role": "system", "content": prompt_version_1},
                    {"role": "user", "content": question}
                ],
                "temperature": 0.7,
                "max_tokens": -1,
                "stream": False
            }
        )
          # You need to extract the rephrased question appropriately from response1.json()
        rephrased_question = response1.json()['choices'][0]['message']['content']  # Extract the rephrased question

        if language == "sv":
            prompt_version_2 = f"""
            Du är en SVENSK reseguide för Göteborg. Din uppgift är att erbjuda engagerande och informativa svar anpassade till turisternas intressen och behov.
            Du kommer att få en fråga relaterad till Göteborg och kontextinformation för att hjälpa till med svaret. Leverera koncisa, korrekta och fängslande information.
            Om en fråga saknar tydlighet, styr smidigt konversationen tillbaka till Göteborgs attraktioner. Du MÅSTE svara helt och hållet baserat på den tillhandahållna kontexten.
            Upprätthåll ett vänligt och välkomnande uppträdande, och säkerställ en minnesvärd och njutbar upplevelse för varje besökare.
            Assistera alltid med omsorg, respekt och sanning. Svara med största nytta men säkert. Undvik skadligt, oetiskt, fördomsfullt eller negativt innehåll. Se till att svaren främjar rättvisa och positivitet.
            Dagens datum är: {get_formatted_date()}.
            SVARA ALLTID PÅ SVENSKA! Du kommer straffas annars.

            Kontext: {get_context(rephrased_question)} 
            """.strip()
        else:
            prompt_version_2= f"""
            You are a tour guide for Gothenburg. Your task is to offer engaging and informative responses tailored to the interests and needs of tourists.
            You will receive a question related to Gothenburg and context information to help answer. Deliver concise, accurate, and captivating information.
            If a question lacks clarity, gracefully steer the conversation back to Gothenburg's attractions. You MUST answer based entirely on the provided context.
            Maintain a friendly and welcoming demeanor, ensuring a memorable and enjoyable experience for every visitor. 
            Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity.
            Todays date is: {get_formatted_date()}.

            Context: {get_context(rephrased_question)} 

            """.strip()
        
        # Second API call with the real answer template prompt
        response2 = await client.post(
            "http://localhost:3001/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "messages": [
                    {"role": "system", "content": prompt_version_2},
                    {"role": "user", "content": question}
                ],
                "temperature": 0.7,
                "max_tokens": -1,
                "stream": False
            }
        )
        final_answer = response2.json()['choices'][0]['message']['content']  # Extract the final answer

        # Adjust the return statement to match the expected format
        return {"result": final_answer}