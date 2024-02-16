import time
import dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Weaviate
import weaviate
from weaviate.embedded import EmbeddedOptions
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
# Ensure to update the import statement according to the actual package and class names
from langchain_openai.chat_models import ChatOpenAI 

# Load environment variables
dotenv.load_dotenv()

# Assuming your local document is named "long_document.txt" and is located in the current directory
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

# Define the tailored prompt for the task
template = """
You are an AI assistant.
Generate concise YAML documentation for the given context, adhering to Swagger specification standards.
Question: {question}
Context: {context}
Answer:
"""
prompt_template = ChatPromptTemplate.from_template(template)

# Setup the ChatOpenAI model (assuming langchain-openai package is correctly used)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)  # Adapt based on actual parameter support

# Define the rag_chain using the components
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)

# Function to process chunks with controlled request size and rate limit management
def process_chunks(chunks, batch_size=5, sleep_time=60):
    responses = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        for chunk in batch:
            try:
                # Here, ensure the chunk or question is formatted as needed for your query
                answer = rag_chain.invoke({"question": "Generate a Swagger API description", "context": chunk})
                responses.append(answer)
            except Exception as e:
                print(f"Error processing chunk: {e}")
        print(f"Processed batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
        time.sleep(sleep_time)  # Wait to help manage API usage
    return responses

# Example usage of the process_chunks function
responses = process_chunks(chunks)
for response in responses[:5]:  # Just printing the first few for demonstration
    print(response)

