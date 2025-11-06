# Final Product

## Overview

This directory contains the near "production-ready" implementation of the Canvas Assistant chatbot, leveraging LangGraph for the backend and Streamlit for the frontend interface. The system implements an advanced, agentic RAG-based chatbot with sophisticated conversation handling and efficient information retrieval.

## What has been built

### Agentic RAG Chatbot with LangGraph and Streamlit

#### Tools
- `rewrite_query`: Optimizes queries for better retrieval performance by breaking down and refining user questions
- `fetch_canvas_guides`: Retrieves relevant information from vector store based on optimized queries
- `compress_information`: Implements LLM-based contextual compression to extract the most relevant information for answering queries

#### Features
- Multi-turn conversation support
- Multiple individual chat sessions
- Last message editing capability
- Long-term memory retention
- Dynamic and user-friendly interface
- Real-time response streaming
- Indexed knowledge base with custom pre-processing
- Tool-call trace for enhanced user experience during response generation

## System Performance and Optimization

The RAG system involves numerous hyperparameters that significantly impact its performance:
- Embedding model configuration
- Retrieval parameters (k-nearest neighbors, similarity thresholds)
- Context window sizes for document chunking
- LLM temperature and sampling settings
- Query rewriting and compression parameters

While initial evaluations show promising results, more comprehensive fine-tuning and rigorous testing would be beneficial before production deployment. This includes:
- Extended parameter optimization across different scenarios
- Larger-scale relevancy testing
- Performance benchmarking under various load conditions
- Systematic evaluation of tool-calling strategies
- More extensive use of LangSmith for debugging and performance analysis (currently only used for basic debugging)

## Future Development Opportunities

### Potential Features
1. Text Input with Screenshot Support
   - Current limitation: AI models lack reliable capability to understand screenshots and UI components

2. User Feedback Collection
   - Not implemented due to technical challenges
   - Would enable response quality tracking and system improvement
   - Could inform hyperparameter optimization

3. Model Context Protocol Integration
   - Future capability to access student's Canvas through API keys
   - Would enable direct interaction with Canvas system


### Running the Application
- Create a new virtual environment and run `installing_requirements.ipynb` file to install all the necessary dependencies
```bash
   streamlit run streamlit_frontend.py
```