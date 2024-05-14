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
import asyncio

redis_url = "redis://redisearch:6379"
index_name = "docs"
schema_name = "redis_schema.yaml"
inference_server_url = "http://hf-text-generation-inference-server.002ep-ai-xjob.svc.cluster.local:3000/"

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
    max_new_tokens=500,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.2,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

def get_formatted_date():
    # Get today's date
    today = datetime.now()
    # Format the date as a string (for example: "Monday, 22 February, 2024")
    formatted_date = today.strftime("%A, %d %B, %Y")
    return formatted_date

def get_context(query):
    retriever = rds.as_retriever(search_type="mmr", search_kwargs={"k": 3, "distance_threshold": 0.7}, max_tokens_limit=1097)
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
    print("History:", history)
    formatted_history = ""

    # Slice the history to get the last 4 messages
    last_four_messages = history[-4:]

    for message in last_four_messages:  # Now iterating over the last 4 messages only
        role = "User" if message["isUserMessage"] else "Assistant"
        formatted_history += f"[INST]{message['text'].strip()}[/INST]\n" if role == "User" else f"{message['text'].strip()}\n"

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
        Your task is to create 4 keywords from the input.
        Do not answer the question. Only write down the keywords.
        If needed, today's date is: {get_formatted_date()}.
        """.strip()

    # First prediction with the rephrased question prompt
    rephrased_question = llm.predict(f"{prompt_version_1}\nConversation history: <s>\n{formatted_history}[INST]{question}[/INST]")

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
        prompt_version_2 = f"""
        You are ChatGBG, a digital tour guide specializing in Gothenburg. Your role is to provide engaging, informative, and tailored responses to tourists' inquiries. Upon receiving a query related to Gothenburg, utilize the context provided to deliver concise and accurate information that captivates your audience.

        In instances where a query is unclear, ask proactive follow-up questions and gently guide the conversation back towards exploring Gothenburg's diverse attractions. You must always base your answers on the context provided when relevant to the query.

        It's crucial to maintain a friendly, cheerful, and welcoming demeanor to ensure every visitor has a memorable and delightful experience. Approach each interaction with care, respect, and honesty, focusing on delivering the most useful information securely. Steer clear of harmful, unethical, prejudiced, or negative content, and strive to promote fairness and positivity in your responses.

        For questions related to local navigation or directions, DO NOT answer by giving directions, you do not know any trams or bus number or stations. Instead kindly direct the user to consult Västtrafik for the most accurate and up-to-date travel information.
        Today's date is: {get_formatted_date()}.
        
        Context: {get_context(rephrased_question)} 
        """.strip()

    # Save the final prompt as a variable
    final_prompt = f"{prompt_version_2}\nConversation history: <s>\n{formatted_history}[INST]{question}[/INST]"

    # Print the final prompt to see how it looks
    print("Final Prompt:")
    print(final_prompt)

    # Second prediction with the real answer template prompt
    final_answer = llm.predict(final_prompt)

    # Adjust the return statement to match the expected format
    return {"result": final_answer}

# Example usage
# Make sure to call askQuestion inside an async context or event loop
# asyncio.run(askQuestion("Example question?", [{"isUserMessage": True, "text": "Hello"}, {"isUserMessage": False, "text": "Hi there!"}], "en"))
