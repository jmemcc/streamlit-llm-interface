# Copyright (c) 2025 jmemcc
# MIT License - See https://mit-license.org/

import streamlit as st
import ollama
from openai import OpenAI
import toml
from datetime import datetime
import time
import json
from typing import Dict, List, Union
from pathlib import Path

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 5000
DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_OLLAMA_MODEL = "llama3.2:latest"

PROMPTS_FILE = Path("data/prompts.toml")


# feel free to remove, just star the repo first ;)
def made_by_credit():
    github_html = """
    <div style="display: flex; align-items: center; padding: 10px; background-color: #f6f8fa; border-radius: 6px; width: fit-content; cursor: pointer; text-decoration: none; margin-bottom: 10px;">
        <img src="https://github.com/jmemcc.png" style="width: 32px; height: 32px; border-radius: 50%; margin-right: 10px;">
        <a href="https://github.com/jmemcc" style="text-decoration: none; color: #24292e; font-size: 15px;">
            Made by @jmemcc
        </a>
    </div>
    """
    st.sidebar.markdown(github_html, unsafe_allow_html=True)


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
        return {model["model"]: model for model in response["models"]}
    except Exception as e:
        st.error(f"Failed to fetch Ollama models: {e}")
        return {}


def setup_page(hide_app_header=False):
    """Initialises the Streamlit page with custom configuration and sidebar settings."""
    st.set_page_config(page_title="Streamlit LLM", page_icon="✨", layout="wide")

    page_settings = ""

    if hide_app_header:
        page_settings += """
        [data-testid="stHeader"] {
            display: none;
        }
        """

    st.sidebar.markdown(
        f"""
        <style>
        {page_settings}
        </style>
        """,
        unsafe_allow_html=True,
    )
    made_by_credit()


def render_model_selection() -> tuple[str, str]:
    """Renders LLM provider and model selection interface in the sidebar."""

    # initialise model states if not exists
    if "current_llm_provider" not in st.session_state:
        st.session_state.current_llm_provider = "OpenAI"
        st.session_state.openai_model = DEFAULT_OPENAI_MODEL
        st.session_state.ollama_model = DEFAULT_OLLAMA_MODEL

    llm_provider = st.sidebar.pills(
        "LLM Provider", ["Ollama", "OpenAI"], selection_mode="single", default="Ollama"
    )

    # update the interface in session state if changed
    st.session_state.current_llm_provider = llm_provider

    selected_model = None
    if llm_provider == "OpenAI":
        try:
            if st.secrets:
                pass
        except FileNotFoundError:
            st.write(
                "You need to make a `secrets.toml` file in `.streamlit/` directory to store OpenAI API key."
            )
            st.stop()
        if (
            "OPENAI_API_KEY" not in st.secrets["connections"]
            or st.secrets["connections"]["OPENAI_API_KEY"] == ""
        ):
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
    elif llm_provider == "Ollama":
        ollama_models = get_ollama_models()
        if ollama_models:
            model_names = list(ollama_models.keys())
            default_index = (
                model_names.index(st.session_state.ollama_model)
                if st.session_state.ollama_model in model_names
                else 0
            )
            selected_model = st.sidebar.selectbox(
                "Model Name", options=model_names, index=default_index
            )
            if selected_model != st.session_state.ollama_model:
                st.session_state.ollama_model = selected_model

    return llm_provider, selected_model


def render_role_selection(prompts: Dict) -> tuple[str, float, int]:
    """Renders role selection interface and returns selected role configuration."""
    role_type = st.sidebar.pills(
        "Role Configuration",
        ["No Role", "Preset Role", "Custom Role"],
        selection_mode="single",
        default="No Role",
    )

    role = ""
    temperature = DEFAULT_TEMPERATURE
    max_tokens = DEFAULT_MAX_TOKENS

    if role_type == "Preset Role":
        prompt_names = [name.replace("_", " ").title() for name in prompts.keys()]
        preset_role = st.sidebar.selectbox(
            "Available Roles", options=prompt_names, index=0
        )
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
            height=250,
        )

    return role, temperature, max_tokens


def handle_chat(
    client: Union[OpenAI, ollama.Client],
    llm_provider: str,
    messages: List[Dict],
    temperature: float,
    max_tokens: int,
) -> tuple[str, str, float]:
    """Processes chat interaction with selected LLM and streams the response."""
    start_time = time.time()
    try:
        if llm_provider == "OpenAI":
            stream = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]} for m in messages
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            response_container = st.empty()
            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    response_container.markdown(full_response)

            model_used = st.session_state["openai_model"]
            elapsed_time = time.time() - start_time
            return full_response, model_used, elapsed_time

        elif llm_provider == "Ollama":
            message_thread = "\n".join(
                [f"{m['role']}: {m['content']}" for m in messages]
            )
            stream = client.generate(
                model=st.session_state["ollama_model"],
                prompt=message_thread,
                options={"temperature": temperature, "num_predict": max_tokens},
                stream=True,
            )

            response_container = st.empty()
            full_response = ""
            for chunk in stream:
                if "response" in chunk:
                    full_response += chunk["response"]
                    response_container.markdown(full_response)

            model_used = st.session_state["ollama_model"]
            elapsed_time = time.time() - start_time
            return full_response, model_used, elapsed_time

    except Exception as e:
        st.error(f"Chat handling error: {e}")
        return "", "", 0.0


def main():
    setup_page()
    prompts = load_prompts()
    st.sidebar.markdown("# Settings")

    # initialise session state for messages and settings if not exists
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "temperature" not in st.session_state:
        st.session_state.temperature = DEFAULT_TEMPERATURE
    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = DEFAULT_MAX_TOKENS

    # sidebar settings
    llm_provider, selected_model = render_model_selection()
    role, preset_temp, preset_max = render_role_selection(prompts)
    st.markdown(f"### {selected_model}")

    has_system_message = (
        st.session_state.messages and st.session_state.messages[0]["role"] == "system"
    )

    if role:
        # if theres already a system message, update it
        if has_system_message:
            st.session_state.messages[0] = {"role": "system", "content": role}
        # if no system message exists, insert it at the beginning
        else:
            st.session_state.messages.insert(0, {"role": "system", "content": role})

        # update temperature and max_tokens with roles presets
        st.session_state.temperature = preset_temp
        st.session_state.max_tokens = preset_max
    elif has_system_message:  # if no role but theres a system message, remove it
        st.session_state.messages.pop(0)

    with st.sidebar.expander("Advanced Settings"):
        new_temp = st.slider("Temperature", 0.0, 2.0, st.session_state.temperature, 0.1)
        new_max_tokens = st.number_input(
            "Max Tokens", 10, 100000, st.session_state.max_tokens, 50
        )

        if new_temp != st.session_state.temperature:
            st.session_state.temperature = new_temp
        if new_max_tokens != st.session_state.max_tokens:
            st.session_state.max_tokens = new_max_tokens

    temperature = st.session_state.temperature
    max_tokens = st.session_state.max_tokens

    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    if st.sidebar.button("Prepare Export"):
        if st.session_state.messages:
            chat_data = {
                "messages": st.session_state.messages,
                "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            st.sidebar.download_button(
                label="Download Chat Export",
                data=json.dumps(chat_data, indent=2),
                file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )

    client = (
        OpenAI(api_key=st.secrets["connections"]["OPENAI_API_KEY"])
        if llm_provider == "OpenAI"
        else ollama.Client()
    )

    # display message history with model information
    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(
                    message["content"]
                )  # Add this line to display message content
                if "model_used" in message:
                    metadata = (
                        f"{message['model_used']} - "
                        f"{message['timestamp']} (took {message.get('generation_time', '??.?')}s) "
                    )
                    st.markdown(
                        f"<div style='color: grey; font-size: 0.7em;'>{metadata}</div>",
                        unsafe_allow_html=True,
                    )

    # handle new messages from user
    if user_prompt := st.chat_input(
        f"Message {selected_model}...", disabled=not llm_provider
    ):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            response, model_used, generation_time = handle_chat(
                client, llm_provider, st.session_state.messages, temperature, max_tokens
            )
            if response:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                metadata = f"{model_used} - {timestamp} (took {generation_time:.1f}s) "
                st.markdown(
                    f"<div style='color: grey; font-size: 0.7em;'>{metadata}</div>",
                    unsafe_allow_html=True,
                )
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response,
                        "model_used": model_used,
                        "timestamp": timestamp,
                        "generation_time": round(generation_time, 1),
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                )


if __name__ == "__main__":
    main()
