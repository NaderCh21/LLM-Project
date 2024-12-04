# Import required modules from langchain
from langchain_google_genai import ChatGoogleGenerativeAI  # For interacting with Google's Gemini model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage  # Message types for chat
from google.oauth2.service_account import Credentials  # For authentication using the JSON key

def create_gemini_llm(model_name: str = "gemini-pro", temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    """
    Creates a Gemini LLM using a service account for authentication.

    Args:
        model_name (str): The model to use. Default is "gemini-pro".
        temperature (float): The temperature to use.

    Returns:
        ChatGoogleGenerativeAI: The Gemini LLM.
    """
    # Path to your service account JSON key file
    credentials = Credentials.from_service_account_file(
        "./service_account_key.json"  # Replace with the correct path to your JSON file
    )
    
    # Initialize the Gemini language model with specified parameters
    llm = ChatGoogleGenerativeAI(
        model=model_name,  # Specify which Gemini model to use
        credentials=credentials,  # Use the service account credentials
        temperature=temperature  # Controls randomness in responses (0.0 = deterministic)
    )
    return llm

# Create an instance of the Gemini LLM using default parameters
llm = create_gemini_llm()

# Send a test message to the LLM and print its response
# invoke() sends the message and returns an AIMessage object
# .content gets the actual response text
response = llm.invoke("How much is the temprature in canada")
print(response.content)

# Initialize chat history with the first human message
messages = [HumanMessage(content="How much is the temprature in canada?")]

# Send messages to LLM and get response
response = llm.invoke(messages)

# Add AI's response to chat history
messages.append(response)

# Add follow-up question to chat history
# This tests if the LLM can remember previous messages
messages.append(HumanMessage(content="What is the first thing I asked?"))

# Get response from LLM with full chat history context
response = llm.invoke(messages)

# Print just the content of the AI's response
print(response.content)
