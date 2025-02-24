# Book Recommendation System

This project is a Book Recommendation System leveraging **Retrieval-Augmented Generation (RAG)**, built using **LangChain**, **DeepSeek Embedding** through **Ollama**, and **ChromaDB** as the vector database. The system is designed to provide highly accurate book suggestions by combining traditional information retrieval techniques with large language model (LLM) capabilities.

## Features

- **Advanced RAG Pipeline**: Utilizes a hybrid RAG model for more accurate and context-aware book recommendations.
- **DeepSeek Embedding**: Employs deep embedding techniques through Ollama for enhanced semantic search.
- **Vector Storage**: Uses **ChromaDB** to store and retrieve book-related embeddings efficiently.
- **Self-Querying Retrieval**: Improves search efficiency by allowing the system to dynamically generate and refine queries.
- **Interactive Interface**: Built with **Gradio** to provide a user-friendly interface for book recommendations.

## Technologies Used

- **Python**: Core programming language
- **LangChain**: Framework for LLM and RAG system development
- **DeepSeek Embedding**: For generating high-quality text embeddings
- **ChromaDB**: Vector database for fast and efficient similarity search
- **Gradio**: User interface for interacting with the recommendation system
- **Ollama**: Integration for using advanced LLMs

## System Architecture

1. ***Input Handling***: User inputs book preferences or related queries via the Gradio interface.
2. **Embedding Generation**: Converts input into dense vectors using DeepSeek Embedding.
3. **Vector Search**: Searches the ChromaDB vector store for the most relevant book data.
4. **Self-Querying Retrieval**: Enhances query results by dynamically adjusting and optimizing search parameters.
5. **Response Generation**: Provides book recommendations based on retrieved data and LLM responses.

## Usage

1. Open your browser and navigate to the provided Gradio link.
2. Enter book preferences or keywords.
3. Receive personalized book recommendations.


