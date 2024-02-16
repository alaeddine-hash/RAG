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

# Assuming your local document is named "document.txt" and is located in the current directory
local_document_path = './contract.txt'

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

# Prompt
template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

# Chat model
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Output parser
rag_chain = (
    {"context": retriever,  "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Invoke the chain
query = "What is the total loan amount that Jane Doe has agreed to lend to John Smith?"
answer = rag_chain.invoke(query)
print(answer)  # This will print the answer


