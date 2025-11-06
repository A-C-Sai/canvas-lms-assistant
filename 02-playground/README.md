# Playground Section

This section documents a systematic exploration of Anthropic's Claude API through AWS Bedrock, progressing from low-level boto3 implementations to higher-level abstractions with LangChain. The notebooks follow a learning path from basic API understanding to advanced features and integrations.

## Purpose

This playground section focuses on two main learning objectives:
- Gaining deep understanding of Anthropic's Claude API through hands-on experimentation with AWS Bedrock, progressing from low-level boto3 implementations to higher-level abstractions
- Exploring LangGraph's capabilities for building stateful chatbots

## Notebook Details

### 1. Claude on Amazon Bedrock (`00-claude-on-amazon-bedrock.ipynb`)

This notebook provides a comprehensive exploration of Claude's capabilities through AWS Bedrock, starting with foundational boto3 implementations and progressing to more advanced features. The exploration follows this sequence:

1. **Fundamental API Interactions**
   - Simple API invocation using boto3 and response handling
   - Multi-turn conversation
   - Citation capabilities
   - Stop sequence configuration
   
2. **Parameter Tuning**
   - Stop sequence configuration
   - Temperature adjustment and few-shot prompting
   - System prompt optimization
   
3. **Prototyping**
   - RAG (Retrieval-Augmented Generation) prototypes with ipywidgets
   
4. **Tool Integration**
   - Basic tool usage implementation
   - Streaming responses with and without tools and handling
   - RAG application prototyping with tools
   
5. **High-Level Abstractions**
   - Integration with ChatBedrockConverse from LangChain
   - Comparison of low-level vs. high-level implementations


### 2. LangGraph Implementation (`01-langgraph.ipynb`)

This notebook demonstrates a progressive exploration of LangGraph capabilities, focusing on different aspects of message management and state persistence. The exploration follows this sequence:

1. **Basic Memory-less Chatbot**
   - Initial implementation with StateGraph
   - Integration with AWS Bedrock

2. **State Persistence Evolution**
   - Implementation of in-memory persistence
   - Exploration of the `add_messages` reducer

3. **Custom Reducer Implementation**
   - Development of custom reducer logic
   - Implementation of last user message editing capability

4. **Message Manipulation**
   - Exploration of LangChain message types (HumanMessage, AIMessage)
   - Message editing capabilities
