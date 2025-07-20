import os
import openai
import pinecone
from langchain.embeddings import OpenAIEmbeddings
import streamlit as st
import arize.pandas.logger as arize
import pandas as pd
import uuid
from dotenv import load_dotenv

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
ARIZE_SPACE_KEY = os.getenv('ARIZE_SPACE_KEY')
ARIZE_API_KEY = os.getenv('ARIZE_API_KEY')

# -------------------------------
# Initialize Pinecone
# -------------------------------
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
INDEX_NAME = "genai-chatbot"

if INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(INDEX_NAME, dimension=1536)

index = pinecone.Index(INDEX_NAME)
embeddings = OpenAIEmbeddings()

# Sample documents for embedding & retrieval
documents = [
    "Artificial Intelligence is transforming industries across the globe.",
    "Climate change is a pressing issue affecting all nations.",
    "Quantum computing could revolutionize problem-solving in science."
]

# Index the sample documents
for i, doc in enumerate(documents):
    vector = embeddings.embed_query(doc)
    index.upsert([(str(i), vector)])

# -------------------------------
# Initialize Arize AI
# -------------------------------
arize_client = arize.Client(space_key=ARIZE_SPACE_KEY, api_key=ARIZE_API_KEY)

# -------------------------------
# Helper Functions
# -------------------------------

def retrieve_documents(query, top_k=2):
    """Retrieve top_k relevant documents from Pinecone."""
    query_vector = embeddings.embed_query(query)
    results = index.query(query_vector, top_k=top_k, include_metadata=True)
    return results['matches']

def build_prompt(query, retrieved_docs):
    """Construct a prompt for the LLM based on retrieved documents."""
    context = "\n".join([documents[int(doc['id'])] for doc in retrieved_docs])
    prompt = (
        "You are a knowledgeable assistant. Use the following context to answer the question accurately.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{query}\n\n"
        "Answer:"
    )
    return prompt, context

def generate_answer(prompt):
    """Generate a response from OpenAI GPT-4 based on the constructed prompt."""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )
    return response['choices'][0]['message']['content'].strip()

def log_interaction_to_arize(query, response, context):
    """Log the interaction details to Arize AI for evaluation."""
    prediction_id = str(uuid.uuid4())
    data = pd.DataFrame([{
        "query": query,
        "context": context,
        "response": response
    }])
    arize_client.log(
        model_id="genai-chatbot",
        model_version="v1",
        prediction_id=prediction_id,
        features={"context": context},
        prediction=response
    )

# -------------------------------
# Streamlit Frontend
# -------------------------------
st.title("🤖 GenAI Chatbot with RAG, Prompt Engineering & Arize AI")

user_query = st.text_input("Ask a question:")

if user_query:
    # Retrieval
    retrieved_docs = retrieve_documents(user_query)
    
    # Prompt construction
    prompt, context = build_prompt(user_query, retrieved_docs)
    
    # Response Generation
    answer = generate_answer(prompt)

    # Display Results
    st.subheader("Answer")
    st.write(answer)

    st.subheader("Context Used")
    st.write(context)

    # Log to Arize AI
    log_interaction_to_arize(user_query, answer, context)
