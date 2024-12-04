# Import required modules from langchain
from langchain_google_genai import ChatGoogleGenerativeAI  # For interacting with Google's Gemini model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage  # Message types for chat
from langgraph.prebuilt import create_react_agent # create_react_agent is a function that takes a language model and tools and returns an agent
from google.oauth2.service_account import Credentials

credentials = Credentials.from_service_account_file("./service_account_key.json")
class GeminiChat:
    def __init__(self, model_name: str = "gemini-pro", temperature: float = 0.0):
        """
        Initialize GeminiChat with a language model.

        Args:
            model_name (str): The model to use. Default is "gemini-pro".
            temperature (float): The temperature to use. Default is 0.0.
        """
        # Initialize the Gemini language model with specified parameters
        self.llm = ChatGoogleGenerativeAI(
                model=model_name,
                credentials=credentials,
                temperature=temperature
            )

        # Define tools for the agent (empty list for now)
        tools = []  # You can later add tools like APIs, calculators, etc.

        # Create a react agent that can use the LLM and tools
        self.agent = create_react_agent(self.llm, tools)

        # Initialize empty list to store conversation history
        self.messages = []
        
    def send_message(self, message: str) -> str:
        """
        Send a message and get response from the model.
        
        Args:
            message (str): The message to send
            
        Returns:
            str: The model's response content
        """
        # Add user's message to conversation history
        self.messages.append(HumanMessage(content=message))
        # Get response from LLM using full conversation history
        response = self.llm.invoke(self.messages)
        # Add AI's response to conversation history
        self.messages.append(response)
        # Return just the content of the response
        return response.content

# Example usage
chat = GeminiChat()  # Create new chat instance with default parameters
print(chat.send_message("Hello, how are you?"))  # Send initial greeting
print(chat.send_message("What is the first thing I asked?"))  # Test conversation memory
