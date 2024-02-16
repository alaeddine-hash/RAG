import dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Weaviate
import weaviate
from weaviate.embedded import EmbeddedOptions
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Load environment variables
dotenv.load_dotenv()

# Assuming your local document is named "long_document.txt" and is located in the current directory
local_document_path = './long_document.txt'

# Loading the document from the file
loader = TextLoader(local_document_path)
documents = loader.load()

# Splitting the document into chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# Embedding the chunks
client = weaviate.Client(embedded_options=EmbeddedOptions())
vectorstore = Weaviate.from_documents(client=client, documents=chunks, embedding=OpenAIEmbeddings(), by_text=False)

# Retrieving the chunks
retriever = vectorstore.as_retriever()

# Tailored Prompt for YAML Swagger API Documentation
template = """You are an AI specialized in generating Swagger API documentation in YAML format. Given the context, produce a concise YAML document describing the API endpoints, parameters, and responses. Be precise and adhere to the Swagger specification standards.
Question: {question}
Context: {context}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

# Theoretical setup for ChatOpenAI to include a max_tokens parameter
# Note: This assumes the ability to limit output length, which may require adjusting the underlying library or API call setup.
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)  # Assuming max_tokens can be set within this or related call

# Output parser
rag_chain = (
    {"context": retriever,  "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Invoke the chain with a specific query
query = "Generate a Swagger API description for user management endpoints including registration, login, and profile update."
answer = rag_chain.invoke(query)
print(answer)  # This will print the Swagger YAML documentation

