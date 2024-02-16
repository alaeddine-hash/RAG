import time
import dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Weaviate
import weaviate
from weaviate.embedded import EmbeddedOptions
from langchain.prompts import ChatPromptTemplate
# Update import based on the actual package structure and availability
from langchain_openai.chat_models import ChatOpenAI 
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Load environment variables
dotenv.load_dotenv()

# Assuming your local document is named "long_document.txt"
local_document_path = './long_document.txt'

# Load and split the document
loader = TextLoader(local_document_path)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# Initialize Weaviate client for embeddings
client = weaviate.Client(embedded_options=EmbeddedOptions())
vectorstore = Weaviate.from_documents(client=client, documents=chunks, embedding=OpenAIEmbeddings(), by_text=False)
retriever = vectorstore.as_retriever()

# Define the tailored prompt for documentation tasks
template = """
You are an AI assistant.
Generate concise YAML documentation for the given context, adhering to Swagger specification standards.
Question: {question}
Context: {context}
Answer:
"""
prompt_template = ChatPromptTemplate.from_template(template)

# Setup the ChatOpenAI model
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Define the rag_chain using the components
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)

# Function to process chunks with controlled request size and rate limit management
def process_chunks(chunks, query, batch_size=5, sleep_time=60):
    responses = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        for chunk in batch:
            try:
                # Assuming chunk is a dictionary and actual text is under a key named 'text'
                chunk_text = chunk['text'] if 'text' in chunk else str(chunk)  # Fallback to converting chunk to string if no 'text' key
                structured_input = {"question": query, "context": chunk_text}
                answer = rag_chain.invoke(structured_input)
                responses.append(answer)
            except Exception as e:
                print(f"Error processing chunk: {e}")
        print(f"Processed batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
        time.sleep(sleep_time)
    return responses



# Invoke processing with a specific query
query = "Generate a Swagger API description for user management endpoints including registration, login, and profile update."
responses = process_chunks(chunks, query)
for response in responses[:5]:  # Printing a subset of responses for demonstration
    print(response)

