# Streamlit Ollama Chat Example

by [@jmemcc](https://github.com/jmemcc)

A starter template for building chat interfaces with Ollama in Streamlit, created as an example for those wanting to build local LLM chat applications. 

**Note**: This is an example repository to develop from, and is not suited for production use.

### Key Features

- **Model Selection**: Choose between Ollama and OpenAI models via the settings sidebar.
- **Role Settings**: Set the assistant's behaviour using presets or custom instructions.
- **Parameters**: Adjust Temperature and Max Tokens to control responses.

## Usage

Visit the app at http://0.0.0.0:8501.

Select an LLM provider and model in the sidebar, then you can interact with the model. Roles can be selected for more tailored model responses - you can add your own in the `data/prompts.toml` file.


## Installation

You need install [Ollama](https://ollama.com/) and at least one model (e.g. [llama3.2](https://ollama.com/library/llama3.2)). 

For OpenAI models, obtain an API key from their [developer dashboard](https://platform.openai.com/api-keys) and add it to `.streamlit/secrets.toml`.

Set up the app manually or using Docker.

### Manual Setup 

1. Create a Python 3.10 environment:
    ```bash
    pyenv virtualenv 3.10 my_env 
    pyenv local my_env
    ```

2. Install requirements:
    ```bash
    pip install -r requirements.txt
    ```

3. Launch the app:
    ```bash
    streamlit run app.py --server.port=8501 --server.address=0.0.0.0
    ```

### Docker Setup

1. Build image:
    ```bash
    docker build -t streamlit_ollama_chat .
    ```

2. Run container:
    ```bash
    docker run -p 8501:8501 streamlit_ollama_chat
    ```
