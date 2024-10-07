import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')

from langchain_groq import ChatGroq

model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

print(model)