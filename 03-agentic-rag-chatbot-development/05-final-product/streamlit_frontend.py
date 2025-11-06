import streamlit as st
from langchain_core.messages import  HumanMessage, AIMessage, AIMessageChunk
from langgraph_backend import chatbot, retrieve_all_threads
import uuid

############################################ 
# Utilities
############################################ 
def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['current_session'] = thread_id
    st.session_state['conversation_history'] = []
    return thread_id

def add_chat_to_session(chat_number, thread_id):
    
    if thread_id not in st.session_state['chat_threads'].values():
        st.session_state['chat_threads'][f'chat-{chat_number}'] = thread_id

def load_chat(thread_id):
    
    state = chatbot.get_state(config={'configurable':{'thread_id':thread_id}}).values
    messages = state.get('messages',[])
    filtered_messages = list(filter(lambda msg: isinstance(msg,(HumanMessage,AIMessage)) and msg.text.strip(), messages))

    return filtered_messages

def message_converter(message):
    if isinstance(message, HumanMessage):
        return {
            'role': 'user',
            'content': message.text
        }
    
    if isinstance(message, AIMessage):
        return {
            'role': 'assistant',
            'content': message.text,
        }


def edit_last_message():
    st.session_state["last_message"] = st.session_state['conversation_history'][-2].get('content')
    st.session_state['edit_mode'] = True
    st.session_state["conversation_history"] = st.session_state["conversation_history"][:-2]
    

############################################ 
# Session Setup
############################################ 

# initilize conversation_history if not present
if 'conversation_history' not in st.session_state:
    st.session_state['conversation_history'] = []

# initilize chat_threads if not present
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = {f'chat-{chat_number}':thread_id for chat_number, thread_id in enumerate(retrieve_all_threads(), start=1)}
    st.session_state['total_chats'] = len(st.session_state['chat_threads']) + 1


# initilize thread_id if not present
if 'current_session' not in st.session_state:
    st.session_state['current_session'] = generate_thread_id()
    add_chat_to_session(st.session_state['total_chats'], st.session_state['current_session'])

if "last_message" not in st.session_state:
    st.session_state['last_message'] = ""

st.session_state.setdefault("edit_mode", False)

# new CONFIG allows us to group traces by thread_id's
CONFIG = {'configurable':{'thread_id':st.session_state['current_session']},
          'metadata':{'thread_id':st.session_state['current_session']},
          'run_name': 'chat_turn'} # each interaction is logged by LangSmith if needed, each trace will be named "chat_turn"



############################################ 
# Sidebar UI
############################################

# disable collapsing sider and set width range
st.markdown("""
<style>
[data-testid="stSidebar"] {
    min-width: 220px;
    max-width: 300px;
}

[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapseButton"] {
    cursor: not-allowed;
    pointer-events: none;
}
</style>
""", unsafe_allow_html=True)

# expandable chat input
st.html(
    """
<style>

    .stChatInput div {
        height: auto;
    }
</style>
    """
)

#  title and caption and margins+padding
st.html(
    """
<style>

    .stHeading h1 {
        padding-top: 0
    }

    .stMarkdown p {
        margin-bottom: 0
    }
</style>
    """
)


st.sidebar.caption('Agentic RAG Chatbot')
st.sidebar.title(':red[ARTIM]')

if st.sidebar.button('New chat', width='stretch', icon='📝',type='primary'):
    thread_id = reset_chat()
    st.session_state['total_chats'] += 1
    add_chat_to_session(st.session_state['total_chats'],thread_id)
    

st.sidebar.header('Chats')

for chat in list(st.session_state['chat_threads'].keys())[::-1]:
    if st.sidebar.button(label = f"💬 {chat.replace('-',' ')}", width='stretch'):
        st.session_state['current_session'] = st.session_state['chat_threads'][chat]
        messages = list(map(message_converter,load_chat(st.session_state['chat_threads'][chat])))
        st.session_state['conversation_history'] = messages



############################################ 
# Main UI
############################################ 

# load chat history in UI
for message in st.session_state['conversation_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])


user_input = st.chat_input('Ask anything about Canvas LMS...',key="last_message") # each time the user interacts, the script is run from the top.

with st._bottom:
    cols = st.columns([1,2,1])
    with cols[1]:
        st.caption(":red[ARTIM] can make mistakes. Check important info.")

edit_prompt = None

if user_input:

    if edit_prompt is not None:
        edit_prompt.empty()
        edit_prompt = None

    # storing user input in session state
    st.session_state['conversation_history'].append({
        "role": "user",
        "content": user_input
    })

    # display user input in UI
    col1, col2 =  st.columns([0.9,0.1], gap="small", vertical_alignment="bottom", border=False)
    with col1:
        with st.chat_message("user"):
            st.markdown(user_input)
    


    # process input with the help of LLM
    # +
    # display AI reply in UI

    with st.chat_message('assistant'):

        empty_space = st.empty()
        
        with empty_space.container():
            st.status(label="")

        # custom generator to disply streamed tokens and tool updates.
        def gen():

            last_msg_id = None
            
            stream = chatbot.stream({"messages":{"op":"edit_last_msg", "text":user_input}}, config=CONFIG,stream_mode=["messages","custom"]) if st.session_state['edit_mode'] else chatbot.stream({"messages":[HumanMessage(content=user_input)]}, config=CONFIG,stream_mode=["messages","custom"])

            st.session_state['edit_mode'] = False

            for stream_mode, message_chunk in stream:

                # print(message_chunk)
                if stream_mode == "messages":
                    if not isinstance(message_chunk[0], (AIMessage, AIMessageChunk)):
                        continue
                    
                    if getattr(message_chunk[0], "chunk_position", None) == "last":
                        with empty_space.container():
                            st.status(label="",state="complete", expanded=False)
                        empty_space.empty()

                    msg_id = getattr(message_chunk[0],"id", None)
                    if msg_id and msg_id != last_msg_id:
                        yield "\n\n"
                        last_msg_id = msg_id

                    tool_calls = getattr(message_chunk[0], "tool_calls", None) or getattr(message_chunk[0], "tool_call_chunks", None)
                    if tool_calls:
                        if message_chunk[0].content[0].get("type") == "text":
                            yield message_chunk[0].text

                    if message_chunk[0].content and message_chunk[0].content[0].get('type') == 'text':
                        yield message_chunk[0].text
                else:
                    with empty_space.container():   
                        if message_chunk.lower().startswith('finished'):
                            st.status(label = message_chunk, state = "complete", expanded=False)

                        st.status(message_chunk)
                    
                    if message_chunk.lower().startswith('finished'):
                        empty_space.empty()
                
        ai_msg = st.write_stream(gen())
    
    with col2:
        edit_prompt = st.empty()
        with edit_prompt.container():
            st.button(label="",icon="✏️",width="content",type="secondary",on_click=edit_last_message)

    # storing ai reply in session state
    st.session_state['conversation_history'].append({
        "role": "assistant",
        "content": ai_msg,
    })