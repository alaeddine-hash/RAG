import dotenv
import requests
import faiss
import numpy as np
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

dotenv.load_dotenv()

# Downloading the document from the internet
url = "https://raw.githubusercontent.com/langchain-ai/langchain/master/docs/docs/modules/state_of_the_union.txt"
res = requests.get(url)
with open("state_of_the_union.txt", "w") as f:
    f.write(res.text)

# Loading the document from the file
loader = TextLoader('./state_of_the_union.txt')
documents = loader.load()

# Splitting the document into chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# Embedding the chunks
embedding_model = OpenAIEmbeddings()
# Assuming the embed function returns a list of embeddings
embeddings = [embedding_model.embed(chunk) for chunk in chunks]
# Convert embeddings list to a numpy array
embeddings_matrix = np.array(embeddings)

# Creating the FAISS index
dimension = embeddings_matrix.shape[1]  # Dimension of the embeddings
index = faiss.IndexFlatL2(dimension)  # Using L2 distance for similarity search
index.add(embeddings_matrix)  # Adding the embeddings to the index

# Define a function to retrieve the chunks using FAISS
def retrieve_chunks(query_embedding, k=5):
    distances, indices = index.search(np.array([query_embedding]), k)
    retrieved_chunks = [chunks[idx] for idx in indices[0]]
    return " ".join(retrieved_chunks)

# Modify the retriever function to work with the new FAISS-based retrieval
def faiss_retriever(query):
    query_embedding = embedding_model.embed(query)
    return retrieve_chunks(query_embedding, k=5)

# Update the prompt to include the modified retriever function
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
    {"context": RunnablePassthrough(lambda query: faiss_retriever(query)), "question": RunnablePassthrough()} 
    | prompt 
    | llm
    | StrOutputParser() 
)

# Invoke the chain
query = "What did the president say about Justice Breyer"
response = rag_chain.invoke(query)
print(response)
