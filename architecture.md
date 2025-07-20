# Architecture Overview

1. **User Query**: Provided via Streamlit frontend.
2. **Embedding**: Query is converted into a vector using OpenAI embeddings.
3. **Retrieval**: Pinecone searches for the most relevant documents.
4. **Prompt Construction**: Retrieved data + user query forms the prompt.
5. **Generation**: OpenAI or Groq API generates the final response.
6. **Evaluation**: All interactions are logged to Arize AI for monitoring.
7. **Display**: Final answer is shown on Streamlit with context.
