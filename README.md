# Streamlit LLM Interface

Made by [@jmemcc](https://github.com/jmemcc)

A chat interface for Ollama/ChatGPT built in Streamlit, created as an example for those wanting to build local LLM chat applications.

## Prerequisites

- [Ollama](https://ollama.com/) installed with at least one model, like [llama3.2](https://ollama.com/library/llama3.2) (if installing manually)
- For OpenAI models: API key from their [developer dashboard](https://platform.openai.com/api-keys)
- Python 3.10 or higher (if installing manually)
- Docker (if using containerised deployment)

## Installation

### Option 1: Manual

Below are steps to setup using [pyenv](https://github.com/pyenv/pyenv), but an environment with venv is also fine.

1. Create a Python environment (Python 3.10):

    ```bash
    pyenv virtualenv 3.10 python3.10.14 
    pyenv local python3.10.14 
    ```

2. Install requirements:

    ```bash
    pip install -r requirements.txt
    ```

3. Launch the app:

    ```bash
    streamlit run app.py
    ```

**Note**: If you are already using port `8501`, run the `streamlit` command above with  `--server.port=XXXX` on your port of choice.

### Option 2: Docker

In the directory, build and run with the compose file:

```bash
docker compose up -d
```

or build and run with the Dockerfile:

```bash
docker build -t streamlit_llm
docker run streamlit_llm
```

**Note**: If you already are using port `8501`, you need to change the port in the Docker files to use another port.

## Usage

Select an LLM provider and model in the sidebar, then you can interact with the model.

Roles can be selected for more tailored responses - you can add your own in the `data/prompts.toml` file.

## Contributing

Contributions are welcome. Please feel free to submit a pull request. For major changes, please open an issue first to discuss the change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.
