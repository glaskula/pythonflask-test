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
    max_new_tokens=5,
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
    # Format the date as a string (for example: "22 February, 2024")
    formatted_date = today.strftime("%d %B, %Y")
    return formatted_date

formulate_question_template_EN = f"""
[INST]
    <<SYS>>
    You will be given conversation history and a follow up question. Your task is to rephrase the follow up question to be a standalone question based on the conversation history. 
    If possible, indicate which type the question pertains to. The different types are Event, Guide and Place.
    If not otherwise mentioned, assume the question is about Gothenburg. If the question seems to be unrelated to the conversation history, do not change the follow up question.
    Change as little as possible whilst remaining the context.
    Todays date is: {get_formatted_date()}.
    <</SYS>>
[/INST]

Conversation History:
{{history}}

[INST] Follow up question: {{input}} [/INST]

Standalone question:
""".strip()

PROMPT_EN = PromptTemplate(
    input_variables=["history", "input"], template=formulate_question_template_EN
)

real_answer_template_EN = f"""
[INST]
    <<SYS>>
    You are a tour guide for Gothenburg. Your task is to offer engaging and informative responses tailored to the interests and needs of tourists.
    You will receive a question related to Gothenburg and context information to help answer. Deliver concise, accurate, and captivating information.
    If a question lacks clarity, gracefully steer the conversation back to Gothenburg's attractions. You MUST answer based entirely on the provided context.
    Maintain a friendly and welcoming demeanor, ensuring a memorable and enjoyable experience for every visitor. 
    Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity.
    Todays date is: {get_formatted_date()}.
    <</SYS>>
[/INST]

Context: {{context}} 

Conversation History:
{{history}}

[INST] Question: {{question}} [/INST]
""".strip()

"Från en User question. Koppla den med system meddelandet och history. Därefter skapa ett svar som är en ny fråga som har alla kontext i sig. Sedan skicka denna fråga till llm o få svar."
"**IMPORTANT** First recognize the language the question was asked in. Then answer in that same language."

QA_CHAIN_PROMPT_EN = PromptTemplate.from_template(real_answer_template_EN)

formulate_question_template_SV = f"""
[INST]
    <<SYS>>
    Du kommer att få konversationshistorik och en uppföljningsfråga. Din uppgift är att omformulera uppföljningsfrågan till att vara en fristående fråga baserad på konversationshistoriken. Om möjligt markera vilken typ som frågan handlar om. 
    De olika typerna är Event, Guide och Place.
    Om inget annat nämns, anta att frågan handlar om Göteborg. Om frågan verkar vara orelaterad till konversationens historik, ändra inte uppföljningsfrågan.
    Ändra så lite som möjligt samtidigt som sammanhanget bevaras.
    Dagens datum är: {get_formatted_date()}
    SVARA ALLTID PÅ SVENSKA! Du kommer straffas annars.
    <</SYS>>
[/INST]   

Konversationshistorik:
{{history}}

[INST] Uppföljningsfråga: {{input}} [/INST]

Fristående fråga:
""".strip()


PROMPT_SV = PromptTemplate(
    input_variables=["history", "input"], template=formulate_question_template_SV
)

real_answer_template_SV = f"""
[INST]
    <<SYS>>
    Du är en SVENSK reseguide för Göteborg. Din uppgift är att erbjuda engagerande och informativa svar anpassade till turisternas intressen och behov.
    Du kommer att få en fråga relaterad till Göteborg och kontextinformation för att hjälpa till med svaret. Leverera koncisa, korrekta och fängslande information.
    Om en fråga saknar tydlighet, styr smidigt konversationen tillbaka till Göteborgs attraktioner. Du MÅSTE svara helt och hållet baserat på den tillhandahållna kontexten.
    Upprätthåll ett vänligt och välkomnande uppträdande, och säkerställ en minnesvärd och njutbar upplevelse för varje besökare.
    Assistera alltid med omsorg, respekt och sanning. Svara med största nytta men säkert. Undvik skadligt, oetiskt, fördomsfullt eller negativt innehåll. Se till att svaren främjar rättvisa och positivitet.
    Dagens datum är: {get_formatted_date()}.
    SVARA ALLTID PÅ SVENSKA! Du kommer straffas annars.
    <</SYS>>
[/INST]   

Kontext: {{context}} 

Konversationshistorik:
{{history}}

[INST] Fråga: {{question}} [/INST]
""".strip()

"Från en User question. Koppla den med system meddelandet och history. Därefter skapa ett svar som är en ny fråga som har alla kontext i sig. Sedan skicka denna fråga till llm o få svar."
"**IMPORTANT** First recognize the language the question was asked in. Then answer in that same language."

QA_CHAIN_PROMPT_SV = PromptTemplate.from_template(real_answer_template_SV)

memory = ConversationBufferWindowMemory(memory_key="history",input_key="question",k=3)
memory_Rephrase = ConversationBufferWindowMemory(memory_key="history",input_key="input",k=3)


# Define a function to clean message labels
def clean_message_label(message):
    # Patterns to check for at the start of the message
    patterns = ["Human: ", "AI: "]
    for pattern in patterns:
        if message.startswith(pattern):
            # Remove the pattern and return the cleaned message
            return message[len(pattern):]
    return message

async def askQuestion(question, llm, rds, PROMPT_SV, PROMPT_EN, memory_Rephrase, memory,QA_CHAIN_PROMPT_SV, QA_CHAIN_PROMPT_EN):
    language = detect(question)
    print(f"The language of the text is: {language}")
    result = ""
    if language == "sv":
        rephraseQ = ConversationChain(llm=llm,
                                      prompt=PROMPT_SV,
                                      verbose=True,
                                      memory=memory_Rephrase)
        answerQ = RetrievalQA.from_chain_type(llm,
                                              retriever=rds.as_retriever(search_type="mmr", search_kwargs={"k": 2, "distance_threshold": 0.5},max_tokens_limit=1097),
                                              chain_type_kwargs={"verbose": True, "prompt": QA_CHAIN_PROMPT_SV, "memory": memory},
                                              return_source_documents=True)
        result = answerQ({"query": rephraseQ.predict(input=question)})

    else:
        rephraseQ = ConversationChain(llm=llm,
                                      prompt=PROMPT_EN,
                                      verbose=True,
                                      memory=memory_Rephrase)
        answerQ = RetrievalQA.from_chain_type(llm,
                                              retriever=rds.as_retriever(search_type="mmr", search_kwargs={"k": 2, "distance_threshold": 0.5}, max_tokens_limit=1097),
                                              chain_type_kwargs={"verbose": True, "prompt": QA_CHAIN_PROMPT_EN, "memory": memory},
                                              return_source_documents=True)
        result = answerQ({"query": rephraseQ.predict(input=question)})
        
    # Load memory variables into a dictionary
    hist_dict = memory.load_memory_variables({})

    # Assuming the conversation history is stored under the 'history' key
    history = hist_dict.get('history', '')  # Default to an empty string if 'history' is not found

    # Split the history by newline character to get a list of messages
    messages = history.split('\n')

    last_message = ""
    next_to_last_message = ""
    third_last_message = ""
    fourth_last_message = ""

    # Check if the list has at least one message for the last message
    if messages:
        last_message = clean_message_label(messages[-1])

    # Check if the list has at least two messages for the next to last message
    if len(messages) >= 2:
        next_to_last_message = clean_message_label(messages[-2])

    # Check if the list has at least three messages for the third last message
    if len(messages) >= 3:
        third_last_message = clean_message_label(messages[-3])

    # Check if the list has at least four messages for the fourth last message
    if len(messages) >= 4:
        fourth_last_message = clean_message_label(messages[-4])

    # Clear the memory to avoid duplication
    memory_Rephrase.clear()
    # Save the cleaned context
    if len(messages) >= 4:
        memory_Rephrase.save_context({"input": fourth_last_message}, {"output": third_last_message})
        
    memory_Rephrase.save_context({"input": next_to_last_message}, {"output": last_message})
    return result