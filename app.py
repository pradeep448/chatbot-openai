from fastapi import FastAPI
# from dotenv import load_dotenv
import os
import requests
import json

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferWindowMemory


app = FastAPI()
# load_dotenv()

# #test our api key
if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") != "":
    loader = UnstructuredFileLoader(r"Safari-BSVI-Owners-Manual.pdf")
    # loader = UnstructuredFileLoader(r"C:\Users\34491\Downloads\nexon-owner-manual-2022.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)

    underlying_embeddings = OpenAIEmbeddings()

    store = LocalFileStore("./cache/")
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings, store, namespace=underlying_embeddings.model)
  
    db = Chroma.from_documents(texts,cached_embedder)
    # db = Chroma(persist_directory='cache', embedding_function=underlying_embeddings)

    template_pdf = """You are a chatbot having a conversation with a human. You give answers very precisely to the point in less than 50 words where it is necessary, if the answer is no, reply with, I apologize, but it seems I specific information is not available in the Safari 2023 manual. Is there anything else I can assist you with? I'm here to help in any way I can

    Given the following extracted parts of a long document and a question, create a final answer.

    {context}

    {chat_history}
    Human: {human_input}
    Chatbot:"""

    prompt_pdf = PromptTemplate(
        input_variables=["chat_history", "human_input", "context"], template=template_pdf
    )
    memory_pdf = ConversationBufferWindowMemory(memory_key="chat_history", input_key="human_input", k=1)
    chain_pdf = load_qa_chain(
        OpenAI(temperature=0), chain_type="stuff", memory=memory_pdf, prompt=prompt_pdf
    )

############################ api calling realtime##########################
    url = "https://demo.nio.deepyan.people.aws.dev/data/cvp/v1/vehicles/data/realTimeData"


    # Set up the headers
    headers = {
        'Authorization': 'allow',
        'x-api-key': 'gR0vowWiYo2YR5hHzyOCd6pvEwYwUIko9foRQhu2',
        'Content-Type': 'application/json',
        
    }

    # Set up the request body
    payload = {
    "vin": "MAT022024TEST0002",
    "interval": {
            "second": 10
        },
    "limit": 20
    }
    json_payload = json.dumps(payload)

    # Make the POST request

    
    @app.get("/")
    async def root():
        return{"message": "Chat bot and pdf query"}

        

    @app.post("/ask_query") #ask_query
    async def hello_end(pdf_input: str = ''):
        
        if pdf_input.lower() == "exit":
            return{"message":"Thanks for using our manual"}
        else:
            docs = db.similarity_search(pdf_input)
            pdf_response = chain_pdf({"input_documents": docs, "human_input": pdf_input}, return_only_outputs=True)['output_text']
            return{"Pdf":f"{pdf_response}"}

    @app.post("/ask_car")
    async def realtime_endpoint(query: str = ''):

        response = requests.post(url, headers=headers, data=json_payload)

        # Check the response
        if response.status_code == 200:
            json_file_re = response.json()
            template = """
                you give the answers to the questions from {json_file} very strictly, This json file contains the telemetry data coming from car. if the "data =[]" is empty or nothing is present in the json_file, is empty you return "car is not moving" follow this strictly.
                Current conversation: The question to this is {input}
                """
            template = template.format(json_file = json_file_re, input =query)
            llm =OpenAI()
            car_response= llm.predict(template)
            return{"message":car_response}
        else:
            return{"Request Failed": response.text}
