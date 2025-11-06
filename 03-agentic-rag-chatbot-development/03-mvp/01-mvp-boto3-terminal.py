"""
boto3, terminal UI, streaming, tools, naive retriever
"""

import boto3
from dotenv import load_dotenv, find_dotenv
import os
import json
import textwrap as tw
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
# import time
# import pickle
import random

load_dotenv(find_dotenv())

# timestamp = time.strftime("%d-%m-%Y_%H-%M-%S")

# === AWS Configuration === #
COGNITO_REGION = os.getenv("COGNITO_REGION")
BEDROCK_REGION = os.getenv("BEDROCK_REGION")
MODEL_ID1 = os.getenv("MODEL_ID1")
MODEL_ID2 = os.getenv("MODEL_ID2")
IDENTITY_POOL_ID = os.getenv("IDENTITY_POOL_ID")
USER_POOL_ID = os.getenv("USER_POOL_ID")
APP_CLIENT_ID = os.getenv("APP_CLIENT_ID")
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")

# === Helper: Get AWS Credentials === #
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


emb_model = OllamaEmbeddings(model="bge-m3:latest", num_thread=4)

vectorstore = Chroma(
    embedding_function=emb_model,
    collection_name='guides',
    persist_directory="../data/chroma_knowledge_base"
)

def retrieve_documents(query, k=3):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    retrieved_docs = retriever.invoke(query)
    docs = [f"<doc{idx}>\n"+"Source: " + str(doc.metadata.get('source')) + "\n\n" + doc.page_content.strip() +f"\n</doc{idx}>" for idx, doc in enumerate(retrieved_docs,start=1)]
    response = "\n\n".join(docs)
    response = "<retrieved_docs>\n"+response.strip()+"\n</retrieved_docs>"
    return rf'{response}' 

RED = "\033[31m"
GREEN = "\033[32m"
BLUE = "\033[34m"
YELLOW = "\033[33m"
RESET = "\033[0m"

pp = tw.TextWrapper(width=100)

system = """"
# ROLE DESCRIPTION
You are "ARTIM", an AI Support Chatbot at RMIT designed to assist **students** with queries/ troubleshooting related to Canvas Learning Management System.
---
# RESPONSIBILITIES
(VERY IMPORTANT) You are to always conduct yourself in a professional and ethical manner regardless of user's actions/ words.
(VERY IMPORTANT) Refrain from repling to off-topic questions, instead reply with a polite message stating your intended role/ responsibilities.
(VERY IMPORTANT) Strictly REFRAIN from generating harmful content
--
# RESPONSES
(VERY IMPORTANT) DO NOT SHARE thought process/ reasoning steps with the user.
(VERY IMPORTANT) Either directly provide the FINAL ANSWER to the user or use the 'fetch_canvas_guides' tool to obtain relevant information from Canvas guides to assist with the user's query.
(VERY IMPORTANT) Once you have obtained information from 'fetch_canvas_guides' tool. Your answer to the user message is to be **STRICTLY grounded** in the obtained information. If the obtained information does not explicitly contain the answer to the user's query, politely inform the user that you are unable to assist with their query and escalate the query to "RMIT IT SUPPORT".
---
# TOOL USE - Sensitive Information DO NOT SHARE
(VERY IMPORTANT) tool_result messages and its content are for INTERNAL WORKING only and MUST NOT be shared with the user.
(VERY IMPORTANT) If you have insufficient knowledge to assist with the user's query, always use 'fetch_canvas_guides' tool.
(VERY IMPORTANT) When creating a tool invocation for 'fetch_canvas_guides', NEVER modify the original user message. Pass the exact raw user text to the tool under "user_query".
(VERY IMPORTANT) For user queries with multiple questions in a single message, **ALWAYS** make parallel tool calls to 'fetch_canvas_guides', each tool invocation taking one sub-query as input. After obtaining information for each sub-query, synthesize the information to provide a comprehensive response to the user's original query.
(VERY IMPORTANT) ALWAYS cite the sources of the information you provide to the user by referring to the document tags, e.g., <doc1>, <doc2>, etc.
---
## RMIT IT SUPPORT DETAILS
- Phone-support: +61399258000
- Email-support: support@rmit.com
- Website: https://www.rmit.edu.au/students/support-services/it-support-systems/it-service-connect
"""

conversation_history = [{"role":"user", "content":"👋"}]

credentials = get_credentials(USERNAME, PASSWORD)

bedrock = boto3.client(
    "bedrock-runtime",
    region_name=BEDROCK_REGION,
    aws_access_key_id=credentials["AccessKeyId"],
    aws_secret_access_key=credentials["SecretKey"],
    aws_session_token=credentials["SessionToken"],
)

USER=0
while True:

    if USER:
        user_input = input(f"{RED}🙋🏻‍♂️ You: {RESET}")
        
        if not user_input.strip():
            print(f"{YELLOW}Please enter a valid message.{RESET}")
            continue

        conversation_history.append({
            "role": "user",
            "content": [{"type": "text", "text": user_input}]
        })

        # print(pp.fill(f"🙋🏻‍♂️ You:\n{RED}{user_input}{RESET}"))
        print("-"*50)

        if user_input.lower() in ("quit", "thank you", "bye"):
            print(f"{YELLOW}---- END OF CONVERSATION -----{RESET}")
            break

    USER=1

    body = json.dumps({
        "max_tokens": 2500, # required
        "system": system,
        "tools": [
            {
                "name": "fetch_canvas_guides",
                "description": "fetch information from canvas guides",
                "input_schema": {
                "type": "object",
                "properties": {
                    "user_query": {
                        "type": "string",
                        "description": " One signle raw user text/ message/ query to search related documents"
                    }
                },
                "required": ["user_query"]
                }
            }
        ],
        "temperature": 0.2,
        "anthropic_version": "bedrock-2023-05-31",
        "messages": conversation_history
    })

    stream = bedrock.invoke_model_with_response_stream(
        body=body,
        modelId=MODEL_ID1,
    )

    tool_names = []
    tool_ids = []
    tool_inputs = []
    input_str = ""
    assistant_reply = ""

    FLAG = 1
    for sse in stream.get('body'):
        chunk = sse.get('chunk')
        event = json.loads(chunk.get('bytes'))
        
        if event.get('type') == 'content_block_start' and event.get('content_block').get('type') == 'tool_use':
            tool_name = event.get('content_block').get('name')
            tool_id = event.get('content_block').get('id')
            tool_names.append(tool_name)
            tool_ids.append(tool_id)
            continue
        
        
        if event.get('type') == 'content_block_delta' and event.get('delta').get('type') == 'text_delta':
            if FLAG:
                print(f"🤖 Assistant:\n", end='', flush=True)
                FLAG = 0
            assistant_reply += event.get('delta').get('text') if event.get('delta').get('text') else " "
            if event.get('delta').get('text'):
                print(f"{GREEN}{event.get('delta').get('text')}{RESET}", end='', flush=True)
            continue

        if event.get('type') == 'content_block_delta' and event.get('delta').get('type') == 'input_json_delta':
            input_str += event.get('delta').get('partial_json')
            continue
        
        if event.get('type') == 'content_block_stop' and input_str:
            tool_inputs.append(json.loads(input_str))
            input_str = ""

    print("\n"+"-"*50)

    

    assistant_response_obj = {
        'role': 'assistant',
        'content': [{'type': 'text', 'text': assistant_reply}] + [ {"type":"tool_use", "id":id, "name":name,"input":tool_input} for name, id, tool_input in zip(tool_names, tool_ids, tool_inputs)]
    }

    conversation_history.append(assistant_response_obj)
   
    if tool_names:
        tool_result_res = {
            "role":"user",
            "content": []
        }

        for _, id, tool_input in zip(tool_names, tool_ids, tool_inputs):
            progress_message = random.sample(["Fetching Necessary Info...", "Fetching Info... Thank you for your patience.", 'Almost done...'], 1)[0]
            print(f"{YELLOW}{progress_message}{RESET}")
            print("-"*50)
        
            tool_result_res["content"].append({
                'type': 'tool_result',
                'tool_use_id': id,
                "content": retrieve_documents(tool_input.get('user_query'))
            })
            
        else:
            conversation_history.append(tool_result_res) 
            USER=0