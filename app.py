import streamlit as st
import ollama
from openai import OpenAI
import toml
from datetime import datetime
import time
import json
from typing import Dict, List, Optional, Union
from pathlib import Path

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1000
PROMPTS_FILE = Path("data/prompts.toml")

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OLLAMA_MODEL = "llama3.2:latest"


def load_prompts() -> Dict:
    """Loads system prompts from a TOML file and returns them as a dictionary."""
    if not PROMPTS_FILE.exists():
        st.error(f"Prompts file not found: {PROMPTS_FILE}")
        return {}
    return toml.load(PROMPTS_FILE)

def get_ollama_models() -> Dict:
    """Fetches and returns available Ollama models from the local server."""
    try:
        client = ollama.Client()
        response = client.list()
        return {model['model']: model for model in response['models']}
    except Exception as e:
        st.error(f"Failed to fetch Ollama models: {e}")
        return {}

def setup_page():
    """Initializes the Streamlit page with custom configuration and sidebar settings."""
    st.set_page_config(layout="wide")
    st.sidebar.markdown("""
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] {
            min-width: 300px;
        }
        </style>
        """, unsafe_allow_html=True)
    st.sidebar.title("Settings")
    

def render_model_selection() -> tuple[str, str]:
    """Renders the LLM provider and model selection interface in the sidebar."""
    # initialise model states if not exists
    if "current_llm_interface" not in st.session_state:
        st.session_state.current_llm_interface = "OpenAI"
        st.session_state.openai_model = DEFAULT_OPENAI_MODEL
        st.session_state.ollama_model = DEFAULT_OLLAMA_MODEL
    
    llm_interface = st.sidebar.pills(
        "LLM Provider",
        ["Ollama", "OpenAI"],
        selection_mode="single",
        default="Ollama"
    )
    
    # update the interface in session state if changed
    st.session_state.current_llm_interface = llm_interface
    
    selected_model = None
    if llm_interface == "OpenAI":
        try:
            if st.secrets: pass
        except FileNotFoundError:
            st.write("You need to make a `/.streamlit/secrets.toml` file to store OpenAI API key.")
            st.stop()
        if "OPENAI_API_KEY" not in st.secrets['connections'] or st.secrets['connections']['OPENAI_API_KEY'] == "":
            st.error("OpenAI API key not set in secrets.")
            st.stop()
        selected_model = st.session_state.openai_model
        # add OpenAI model selector if you want to support multiple models
        new_model = st.sidebar.selectbox(
            "Model Name",
            options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        )
        if new_model != selected_model:
            st.session_state.openai_model = new_model
            selected_model = new_model
    elif llm_interface == "Ollama":
        ollama_models = get_ollama_models()
        if ollama_models:
            
            model_names = list(ollama_models.keys())
            default_index = (model_names.index(st.session_state.ollama_model) 
                           if st.session_state.ollama_model in model_names 
                           else 0)
            selected_model = st.sidebar.selectbox(
                "Model Name",
                options=model_names,
                index=default_index
            )
            if selected_model != st.session_state.ollama_model:
                st.session_state.ollama_model = selected_model
    
    return llm_interface, selected_model

def render_role_selection(prompts: Dict) -> tuple[str, float, int]:
    """Renders role selection interface and returns selected role configuration."""
    role_type = st.sidebar.pills(
        "Role Configuration",
        ["No Role", "Preset Role", "Custom Role"],
        selection_mode="single",
        default="No Role"
    )
    
    role = ""
    temperature = DEFAULT_TEMPERATURE
    max_tokens = DEFAULT_MAX_TOKENS
    
    if role_type == "Preset Role":
        prompt_names = [name.replace("_", " ").title() for name in prompts.keys()]
        preset_role = st.sidebar.selectbox("Available Roles", options=prompt_names, index=0)
        toml_key = preset_role.lower().replace(" ", "_")
        preset_config = prompts[toml_key]
        role = preset_config.get("role", "")
        temperature = preset_config.get("temperature", DEFAULT_TEMPERATURE)
        max_tokens = preset_config.get("max_tokens", DEFAULT_MAX_TOKENS)
        st.sidebar.code(role, wrap_lines=True, language="text")
    
    elif role_type == "Custom Role":
        role = st.sidebar.text_area(
            "Role Instructions",
            value="",
            placeholder="You are a helpful AI assistant...",
            height=250
        )
    
    return role, temperature, max_tokens

def handle_chat(client: Union[OpenAI, ollama.Client], 
                llm_interface: str, 
                messages: List[Dict],
                temperature: float,
                max_tokens: int) -> tuple[Optional[str], str, float]:
    """Processes chat interaction with selected LLM and streams the response."""
    start_time = time.time()
    try:
        if llm_interface == "OpenAI":
            stream = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[{"role": m["role"], "content": m["content"]} for m in messages],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            response = st.write_stream(stream)
            model_used = st.session_state["openai_model"]
            elapsed_time = time.time() - start_time
            return response, model_used, elapsed_time
        
        else:  # Ollama
            message_thread = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            stream = client.generate(
                model=st.session_state["ollama_model"],
                prompt=message_thread,
                options={"temperature": temperature, "num_predict": max_tokens},
                stream=True
            )
            
            response = ""
            response_container = st.empty()
            for chunk in stream:
                if 'response' in chunk:
                    response += chunk['response']
                    response_container.markdown(response)
            model_used = st.session_state["ollama_model"]
            elapsed_time = time.time() - start_time
            return response, model_used, elapsed_time
    
    except Exception as e:
        st.error(f"Chat handling error: {e}")
        return None, "", 0.0

def main():
    setup_page()
    prompts = load_prompts()
    
    # initialise session state for messages and settings if not exists
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "temperature" not in st.session_state:
        st.session_state.temperature = DEFAULT_TEMPERATURE
    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = DEFAULT_MAX_TOKENS
    
    # sidebar settings
    llm_interface, selected_model = render_model_selection()
    role, preset_temp, preset_max = render_role_selection(prompts)
    
    # update system message and settings if role changes
    if (not st.session_state.messages and role) or \
       (st.session_state.messages and 
        st.session_state.messages[0]["role"] == "system" and 
        st.session_state.messages[0]["content"] != role):
        if st.session_state.messages and st.session_state.messages[0]["role"] == "system":
            st.session_state.messages.pop(0)
        if role:
            st.session_state.messages.insert(0, {"role": "system", "content": role})
            st.session_state.temperature = preset_temp
            st.session_state.max_tokens = preset_max
    
    # advanced settings
    if st.sidebar.toggle("Show advanced model settings"):
        new_temp = st.sidebar.slider("Temperature", 0.0, 2.0, st.session_state.temperature, 0.1)
        new_max_tokens = st.sidebar.number_input("Max Tokens", 10, 4000, st.session_state.max_tokens, 50)
        
        if new_temp != st.session_state.temperature:
            st.session_state.temperature = new_temp
        if new_max_tokens != st.session_state.max_tokens:
            st.session_state.max_tokens = new_max_tokens
    
    temperature = st.session_state.temperature
    max_tokens = st.session_state.max_tokens
    
    col1, col2 = st.sidebar.columns(2)
    if col1.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    if col2.button("Prepare Chat Export"):
        if st.session_state.messages:
            chat_data = {
                "messages": st.session_state.messages,
                "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            col2.download_button(
                label="Download Chat Export",
                data=json.dumps(chat_data, indent=2),
                file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

    client = OpenAI(api_key=st.secrets["connections"]["OPENAI_API_KEY"]) if llm_interface == "OpenAI" else ollama.Client()

    # display message history with model information
    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "model_used" in message:
                    metadata = (
                        f"{message['model_used']} - "
                        f"{message['timestamp']} (took {message.get('generation_time', '??.?')}s) "
                    )
                    st.markdown(
                        f"<div style='color: grey; font-size: 0.7em; '>{metadata}</div>",
                        unsafe_allow_html=True
                    )
    
    # handle new messages from user
    if user_prompt := st.chat_input("Enter your message here.", disabled=not llm_interface):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)
        
        with st.chat_message("assistant"):
            response, model_used, generation_time = handle_chat(
                client, llm_interface, st.session_state.messages, temperature, max_tokens
            )
            if response:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                metadata = (
                    f"{model_used} - "
                    f"{timestamp} (took {generation_time:.1f}s) "
                )
                st.markdown(
                    f"<div style='color: grey; font-size: 0.7em;'>{metadata}</div>",
                    unsafe_allow_html=True
                )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "model_used": model_used,
                    "timestamp": timestamp,
                    "generation_time": round(generation_time, 1),
                    "temperature": temperature,
                    "max_tokens": max_tokens
                })

if __name__ == "__main__":
    main()