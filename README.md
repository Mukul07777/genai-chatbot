GenAI Chatbot with RAG, Pinecone, Arize AI & Streamlit
A Generative AI chatbot leveraging Retrieval-Augmented Generation (RAG), Pinecone vector database, and Arize AI for evaluation, deployed using Streamlit.

🚀 Features
RAG pipeline for knowledge retrieval
OpenAI embeddings for semantic search
Pinecone as the vector database
Prompt engineering for accurate responses
Evaluation with Arize AI
Interactive Streamlit interface

# Setup Instructions

1. Clone the repository:
git clone https://github.com/yourusername/genai-chatbot.git
cd genai-chatbot

2. Create a `.env` file based on `.env.example`:
cp .env.example .env

3. Add your API keys to the `.env` file:
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENV=your_pinecone_environment
ARIZE_SPACE_KEY=your_arize_space_key
ARIZE_API_KEY=your_arize_api_key

4. Install the required dependencies:
pip install -r requirements.txt

5. Run the Streamlit application:
streamlit run main.py
