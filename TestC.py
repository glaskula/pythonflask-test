import httpx
import asyncio
from datetime import datetime

# Function to simulate getting today's date
def get_formatted_date():
    return datetime.now().strftime("%d %B, %Y")

# Mock function to simulate context retrieval
def get_mock_context():
    # Return a simulated context string
    return "This is a simulated context based on the query."

async def askQuestion(question, language):
    # Use a mock context for simplification
    context_string = get_mock_context()
    print(f"The language of the text is: {language}")

    async with httpx.AsyncClient(timeout=60.0) as client:  # 30 seconds timeout 
        # Prepare the first prompt for rephrasing the question
        prompt_version_1 = "Your rephrase prompt here, adjusted for simplicity."

        # Example API call to rephrase the question (Simulated here)
        # Simulate fetching the rephrased question from the API
        rephrased_question = "This is a simulated rephrased question."

        # Prepare the second prompt with the context
        prompt_version_2 = f"""
        Your final answer template here. Include placeholders for dynamic content like date and context.
        Todays date is: {get_formatted_date()}.
        Context: {context_string}
        """.strip()

        # Actual API call to get the final answer based on the rephrased question and context
        response = await client.post(
            "http://95.80.38.172:3001/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "messages": [
                    {"role": "system", "content": prompt_version_2},
                    {"role": "user", "content": rephrased_question}
                ],
                "temperature": 0.7,
                "max_tokens": -1,
                "stream": False
            }
        )
        
        # Extract and return the final answer from the response
        final_answer = response.json()['choices'][0]['message']['content']
        
        return final_answer

# Example of how to call the function
async def main():
    question = "What is the history of Gothenburg?"
    language = "en"
    answer = await askQuestion(question, language)
    print(f"Final Answer: {answer}")

# Run the main function
asyncio.run(main())
