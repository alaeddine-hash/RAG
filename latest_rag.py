import dotenv
import requests
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

# Downloading the document from the internet
url = "https://example.com/renewable_energy_advancements.txt"  # Replace with actual URL
res = requests.get(url)
with open("renewable_energy_advancements.txt", "w") as f:
    f.write(res.text)

# Loading the document from the file
loader = TextLoader('./renewable_energy_advancements.txt')
documents = loader.load()

# Splitting the document into chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# Embedding the chunks
client = weaviate.Client(embedded_options=EmbeddedOptions())
vectorstore = Weaviate.from_documents(client=client, documents=chunks, embedding=OpenAIEmbeddings(), by_text=False)

# Retrieving the chunks
retriever = vectorstore.as_retriever()

# Prompt
template = """
You are an assistant knowledgeable about recent advancements in renewable energy. 
Use the following pieces of retrieved context to inform about the latest technologies, breakthroughs, or policies. 
If you don't have enough information, suggest areas for further research.
Question: {question} 
Context: {context} 
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

# Chat model
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Output parser
rag_chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())

# Invoke the chain
query = "What are the latest breakthroughs in solar energy?"
answer = rag_chain.invoke(query)
print(answer)  # This will print the answer

