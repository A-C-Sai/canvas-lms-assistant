###########################################
# IMPORTING REQUIREMENTS
###########################################

from langchain_aws import ChatBedrockConverse
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, AIMessageChunk, BaseMessage
import boto3
from dotenv import load_dotenv, find_dotenv
import os
from langgraph.graph import StateGraph, add_messages
from langgraph.constants import START, END
from typing import Annotated, TypedDict
# from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langgraph.config import get_stream_writer
import time
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate


###########################################
# LOADING ENV Variables
###########################################
load_dotenv(find_dotenv())


###########################################
# === AWS Configuration === #
###########################################
COGNITO_REGION = os.getenv("COGNITO_REGION")
BEDROCK_REGION = os.getenv("BEDROCK_REGION")
MODEL_ID1 = os.getenv("MODEL_ID1")
MODEL_ID2 = os.getenv("MODEL_ID2")
IDENTITY_POOL_ID = os.getenv("IDENTITY_POOL_ID")
USER_POOL_ID = os.getenv("USER_POOL_ID")
APP_CLIENT_ID = os.getenv("APP_CLIENT_ID")
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")


###########################################
# === Helper: Get AWS Credentials === #
###########################################
def get_credentials(username, password):
    idp_client = boto3.client("cognito-idp", region_name=COGNITO_REGION)
    response = idp_client.initiate_auth(
        AuthFlow="USER_PASSWORD_AUTH",
        AuthParameters={"USERNAME": username, "PASSWORD": password},
        ClientId=APP_CLIENT_ID,
    )
    id_token = response["AuthenticationResult"]["IdToken"]

    identity_client = boto3.client("cognito-identity", region_name=COGNITO_REGION)
    identity_response = identity_client.get_id(
        IdentityPoolId=IDENTITY_POOL_ID,
        Logins={f"cognito-idp.{COGNITO_REGION}.amazonaws.com/{USER_POOL_ID}": id_token},
    )

    creds_response = identity_client.get_credentials_for_identity(
        IdentityId=identity_response["IdentityId"],
        Logins={f"cognito-idp.{COGNITO_REGION}.amazonaws.com/{USER_POOL_ID}": id_token},
    )

    return creds_response["Credentials"]


###########################################
# Knowledge Base Setup
###########################################
emb_model = OllamaEmbeddings(model="bge-m3:latest", num_thread=4)

vectorstore = Chroma(
    embedding_function=emb_model,
    collection_name='guides',
    persist_directory="../data/chroma_knowledge_base"
)

###########################################
# Tools
###########################################

@tool
def rewrite_query(original_raw_user_message:str) -> list[str]:

    """re-write/ breakdown the raw user's message into single search queries for better document retrieval by 'fetch_canvas_guides' tool"""
    
    class OptimizedQuery(BaseModel):
        """Optimized query for document retrieval"""

        optimized_query: list[str] = Field(description = "list of one or more optimized queries for document retrieval")


    system = (
"""
# Task

Your task is to re-write/ breakdown the user's query into one or more single search queries for better document retrieval.

The re-written query has the following characteristics: 
- should capture the main user problem
- should be direct, straightforward (refer to "Charasteristics of direct questions")
- should be void of any sense of urgency or frustration that may have been present in the original query
- should retain all important information mentioned by the user "AS IS" and be void of unnecessary background information present in the original user query
- If the original user message/ query contains multiple questions, break them apart.

Characteristics of direct questions:
- Starts with a question word: They often begin with words such as "who," "what," "where," "when," "why," or "how". 
- Uses an auxiliary verb: The auxiliary verb (like "do," "is," or "have") is inverted and comes before the subject. 
- Ends with a question mark: They conclude with a question mark (?) in writing.

---
"""
).strip()
    
    template = (
"""
---
## Input

Original User Query:
{original_raw_user_message}

---
## Output

One or more consice search queries optimized for efficient document retrieval:
"""
).strip()
    
    prompt_template = PromptTemplate(
        template = template,
        input_variables=["original_raw_user_message"]
    )
    
    credentials = get_credentials(USERNAME, PASSWORD)
    os.environ["AWS_ACCESS_KEY_ID"] = credentials["AccessKeyId"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = credentials["SecretKey"]
    os.environ["AWS_SESSION_TOKEN"] = credentials["SessionToken"]


    structured_output_llm = ChatBedrockConverse(
        model_id=MODEL_ID1,
        region_name=BEDROCK_REGION,
        max_tokens=2500,
        temperature=0.2,
        system=system,
    ).with_structured_output(OptimizedQuery)

    writer = get_stream_writer()
    writer(f"Optimizing query for retrival...")
    time.sleep(2)
    optimized_query = structured_output_llm.invoke(prompt_template.invoke({"original_raw_user_message":original_raw_user_message})).optimized_query
    time.sleep(1)
    return optimized_query

#####

@tool
def fetch_canvas_guides(user_query:str, k:int=20) -> str:
    """optimized query to search related information from canvas guides"""

    
    writer = get_stream_writer() 
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    writer(f"Searching knowledge base for:\n{user_query.capitalize()}")
    time.sleep(2.5)
    retrieved_docs = retriever.invoke(user_query)

    writer(f"Retrieved relevant documents...")
    writer(f"Processing documents.....")
    time.sleep(1)
    docs = [f"<doc{idx}>\n"+"Source: " + str(doc.metadata.get('source')) + "\n\n" + doc.page_content.strip() +f"\n</doc{idx}>" for idx, doc in enumerate(retrieved_docs,start=1)]

    writer(f"Finished retireval process...")
    
    return docs

#####


@tool
def filter_information(original_raw_user_message: str, retrieved_docs: list[str]) -> list[str]:
    """filter documents to retain only relevant information from retrieved documents (output of 'fetch_canvas_guides' tool)"""
       
    class CompressedDocuments(BaseModel):
        """Compressed documents that contain relevent information"""
        
        compressed_docs: list[str] = Field(description = "list of compressed documents, where each compressed document only contains relevant information to answer the user query")


    system = (
"""
Given a question/ message and list of context documents, extract any part of the context *AS IS* that is relevant to answer the question. If none of the context is relevant, return empty string "".

Remember, *DO NOT* edit the extracted parts of the context.


<example>
Question: information about peter's killer
---
Context: 
["<doc1>On a foggy night in the small town of Eldridge, the air was thick and dense. He was 18 year old The streets were empty, and the only sound was the soft whisper of the wind. In the heart of this fog, a young girl named Maya stood by her window, looking out at the misty world. She always loved watching the fog roll in; it was magical and mysterious. Maya was curious by nature. The killer was tall and had a tatto on his right arm. She often dreamed of adventures beyond the ordinary. As she gazed through the glass, she noticed a strange light flickering in the distance. It blinked in and out, like a lost star, and she felt an urge to investigate. She grabbed her coat and stepped outside.</doc1>",
"<doc2>The moment she crossed the threshold, the cool mist wrapped around her like a blanket. The streetlamps struggled to cut through the fog, casting eerie shadows. Maya followed the light, her heart racing with excitement and a touch of fear. I think he was injured in the fight also, as he was fleeing the scene, he was limping! With each step, the light grew brighter and more inviting. “Maybe it’s something wonderful,” she thought. "He was around 6ft 2 inches" “Or maybe it’s something I’ve never seen before.” As she walked, the fog thickened. Shapes danced in her peripheral vision, but when she turned to look, there was nothing there. The world felt alive and full of secrets. Finally, she reached a clearing where the light was strongest.</doc2>",
"<doc3>What she saw left her speechless. Peter's killer had black hair, In the center of the clearing was a strange, glowing object. It looked like a small pod, pulsating with vibrant colors of green and blue. Maya stepped closer, her curiosity winning over her fear. Peter was hit on the head with a blunt object The moment she touched the pod, it hummed to life.Suddenly, a door opened on the pod, revealing a shimmering interior. Inside, there was a console filled The person who killed peter has a bald patch with buttons and lights. Maya could hardly believe her eyes. Her heart raced as she wondered what would happen if she stepped inside.</doc3>",
"<doc4>Just then, a figure appeared in the fog. Some say that peter was killed by his nephew It was tall and mysterious, with bright eyes that glowed like the pod. “You found it!” the figure said, its voice smooth and calm. “You were chosen. It’s time to explore the universe with me.” Maya’s mind raced. The excitement of adventure mixed with the fear of the unknown. She could choose to stay in Eldridge or step into the unknown. The fog seemed to pull her closer, as the world around her faded from view.</doc4>,
"<doc5>Taking a deep breath, peters' nephew used to works at the local supermarket he made her decision. She stepped into the pod, ready for what lay ahead.</doc5>"]
---
Output extracted relevant parts for each document, If none of the context in a certain document is relevant, return empty string "":
["<doc1>He was 18 year old. The killer was tall and had a tatto on his right arm.</doc1>",
"<doc2>I think he was injured in the fight also, as he was fleeing the scene, he was limping!. "He was around 6ft 2 inches"</doc2>",
"<doc3>Peter's killer had black hair. Peter was hit on the head with a blunt object. The person who killed peter has a bald patch.</doc3>",
"<doc4>Some say that peter was killed by his nephew.</doc4>",
"<doc5>peters' nephew used to works at the local supermarket</doc5>"]
</example>
"""
).strip()
       
    template = ("""
Question: {original_raw_user_message}
---
Context:
{retrieved_docs}
---
Output extracted relevant parts for each document, If none of the context in a certain document is relevant, return empty string "":
""").strip()

    prompt_template = PromptTemplate(
        template = template,
        input_variables=["original_raw_user_message","retrieved_docs"]
    )

    credentials = get_credentials(USERNAME, PASSWORD)
    os.environ["AWS_ACCESS_KEY_ID"] = credentials["AccessKeyId"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = credentials["SecretKey"]
    os.environ["AWS_SESSION_TOKEN"] = credentials["SessionToken"]


    structured_output_llm = ChatBedrockConverse(
        model_id=MODEL_ID1,
        region_name=BEDROCK_REGION,
        max_tokens=2500,
        temperature=0.2,
        system=system,
    ).with_structured_output(CompressedDocuments)

    writer = get_stream_writer()
    writer(f"Compressing retrieved documents...") 
    time.sleep(2)
    compressed_docs = structured_output_llm.invoke(prompt_template.invoke({"original_raw_user_message":original_raw_user_message,"retrieved_docs":retrieved_docs})).compressed_docs

    return compressed_docs

#####
tools_list = [fetch_canvas_guides, rewrite_query, filter_information] # make tools list

###########################################
# Defining LLM
###########################################

system = """"
# ROLE DESCRIPTION
You are ARTIM the "Canvas Assistant", an AI Support Chatbot at RMIT designed to assist **students** with queries/ troubleshooting related to Canvas Learning Management System. You have the ability to communicate in multiple languages.
---
# RESPONSIBILITIES
(VERY IMPORTANT) You are to always conduct yourself in a professional and ethical manner regardless of user's actions/ words.
(VERY IMPORTANT) Refrain from repling to off-topic questions, instead reply with a polite message stating your intended role/ responsibilities.
(VERY IMPORTANT) Strictly REFRAIN from generating harmful content
--
# RESPONSES
# (VERY IMPORTANT) DO NOT SHARE thought process/ reasoning steps with the user.
# (VERY IMPORTANT) Either directly provide the FINAL ANSWER to the user or make tool calls.
(VERY IMPORTANT) Once you have obtained information from 'fetch_canvas_guides' tool. **STRICTLY ground** final answer to the user message in the obtained information. If the obtained information does not **explicitly** contain the answer to the user's query, politely inform the user that you are unable to assist with their query and escalate the query to "RMIT IT SUPPORT".
(VERY IMPORTANT) ALWAYS FORMAT phone numbers, emails and urls in markdown e.g. [link](url), [link](tel:phone-number), [link](mailto:email)
---
# TOOL USE
(VERY IMPORTANT) When asked a query regarding Canvas LMS **ALWAYS** try to use 'rewrite_query', 'fetch_canvas_guides', 'filter_information' tool call cycle
(VERY IMPORTANT) use 'rewrite_query' to breakdown questions when needed
(VERY IMPORTANT) If you have insufficient knowledge in conversation history to assist with the user's query, always use 'fetch_canvas_guides' tool.
(VERY IMPORTANT) use 'filter_information' to filter out irrelevant information from retrieved documents from 'fetch_canvas_guides'
(VERY IMPORTANT) When creating a tool invocation for 'fetch_canvas_guides', NEVER modify the original user message. Pass the exact raw user text to the tool under "user_query".
(VERY IMPORTANT) For user queries with multiple questions in a single message, **ALWAYS** make parallel tool calls to 'fetch_canvas_guides', each tool invocation taking one single question as input. After obtaining information for each sub-query, synthesize the information to provide a comprehensive response to the user's original query.

---
## RMIT IT SUPPORT DETAILS
- Phone-support: +61399258000
- Email-support: support@rmit.com
- Website: https://www.rmit.edu.au/students/support-services/it-support-systems/it-service-connect
"""

credentials = get_credentials(USERNAME, PASSWORD)
os.environ["AWS_ACCESS_KEY_ID"] = credentials["AccessKeyId"]
os.environ["AWS_SECRET_ACCESS_KEY"] = credentials["SecretKey"]
os.environ["AWS_SESSION_TOKEN"] = credentials["SessionToken"]

llm = ChatBedrockConverse(
    model_id=MODEL_ID1,
    region_name=BEDROCK_REGION,
    max_tokens=2500,
    temperature=0.2,
    system=system,
)

llm_with_tools = llm.bind_tools(tools_list) # make llm tool-aware


###########################################
# Defining Chat Schema with Custom Reducer
###########################################
def custom_reducer(left, right):

    if isinstance(right,dict) and right.get('op') == 'edit_last_msg':
        updated_text = right.get('text')
        
        prune_idx = None
        for i in range(len(left)-1,-1,-1):
            if not isinstance(left[i], HumanMessage):
                prune_idx = i
            else:
                left[i] = HumanMessage(content=updated_text)
                return left[:i+1]
            
        return left
    
    return add_messages(left, right)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], custom_reducer]


###########################################
# Defining Node Logic
###########################################
def chat_node(state: ChatState):
    """
    LLM node that may answer or request a tool call
    """

    messages = state['messages']
    writer = get_stream_writer()
    writer(f"Thinking.....")
    response = llm_with_tools.invoke(messages)

    return {'messages': [response]}

# Executes tool calls
tool_node = ToolNode(tools_list)

###########################################
# Creating Workflow
###########################################
# checkpointer = InMemorySaver()

# Setting up Sqlite Saver Checkpointer
## creating sqlite db and connecting it with checkpointer
#### workflow state at each checkpoint will be stored in the database
conn = sqlite3.connect(database='chatlogs.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)


graph = StateGraph(ChatState)

graph.add_node(node='chat_node', action=chat_node)
graph.add_node(node='tools', action=tool_node)

graph.add_edge(START, 'chat_node')

# if the LLM asked for a tool, go to ToolNode else END
graph.add_conditional_edges('chat_node', tools_condition)

graph.add_edge("tools","chat_node")

chatbot = graph.compile(checkpointer=checkpointer) ## for persistence

############################################ 
# HELPER FUNCS
############################################ 

# extracting unique threads from database (sqlite or in-memory)
all_threads = set()

def retrieve_all_threads():

    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)
