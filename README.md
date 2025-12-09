![oreilly-logo](images/oreilly.png)

# Designing and Deploying LLM Pipelines

This repository contains code for the [O'Reilly Live Online Training for Designing and Deploying LLM Pipelines](https://learning.oreilly.com/live-events/designing-and-deploying-llm-pipelines/0642572014796)

In this comprehensive course, machine learning engineers and software developers learn how to transition large language model (LLM) prototypes into fully deployed production systems. Through detailed instruction and real-world case studies, you explore the best practices for integrating LLMs into diverse workflows, ensuring that your models perform effectively in practical applications.

## Repository Structure

- `data/` — Example CSV datasets used in some notebooks
- `deploy/` — Minimal FastAPI service for model inference, with `Dockerfile` and runtime `requirements.txt`
- `images/` — Course/README images
- `notebooks/` — Jupyter notebooks used throughout the course
	- `crewai_streamlit/` — Streamlit app demonstrating CrewAI with `app.py`, its own `requirements.txt`, and an example `secrets.toml`
- `requirements.txt` — Primary Python dependencies for running notebooks locally

## Setup Instructions

### Using Python 3.11 Virtual Environment

At the time of writing, we need a Python virtual environment with Python 3.11.

#### Option 1: Python 3.11 is Already Installed

##### Step 1: Verify Python 3.11 Installation

```bash
python3.11 --version
```

##### Step 2: Create a Virtual Environment

```bash
python3.11 -m venv .venv
```

This creates a `.venv` folder in your current directory.

##### Step 3: Activate the Virtual Environment

- **macOS/Linux:**
  
  ```bash
  source .venv/bin/activate
  ```

- **Windows:**
  
  ```cmd
  .venv\Scripts\activate
  ```

You should see `(.venv)` in your terminal prompt.

##### Step 4: Verify the Python Version

```bash
python --version
```

##### Step 5: Install Packages

```bash
pip install -r requirements.txt
```

##### Step 6: Deactivate the Virtual Environment

```bash
deactivate
```

---

#### Option 2: Install Python 3.11

If you don’t have Python 3.11, follow the steps below for your OS.

##### **macOS (Using Homebrew)**

```bash
brew install python@3.11
```

##### **Ubuntu/Debian**

```bash
sudo apt update
sudo apt install python3.11 python3.11-venv
```

##### **Windows (Using Windows Installer)**

1. Go to [Python Downloads](https://www.python.org/downloads/release/python-3110/).
2. Download the installer for Python 3.11.
3. Run the installer and ensure **"Add Python 3.11 to PATH"** is checked.

### Verify Installation

```bash
python3.11 --version
```

You might need to run this command to make the venv findable in jupyter

```bash
python -m ipykernel install --user --name=oreilly-ai-pipelines --display-name "Python (oreilly-ai-pipelines)"
```

### Installing Course Dependencies

Install the main dependencies for notebooks:

```bash
pip install -r requirements.txt
```

Note: The `deploy/` service has its own minimal `requirements.txt` optimized for the API runtime. See the Deployment section below if you plan to run the FastAPI service.


## Notebooks



#### Agents and Workflows

- **[Introduction to LangGraph](notebooks/LangGraph_Hello_World.ipynb)** - Building workflows and agents with LangGraph
	- **[Reflection to LangGraph](notebooks/LangGraph_Reflect.ipynb)** - Reflection with LangGraph
	- **[Planning/Executing with LangGraph](notebooks/LangGraph_Plan_Execute.ipynb)** - Planning/Executing with LangGraph
	- **[ReAct with LangGraph](notebooks/LangGraph_React.ipynb)** - ReAct Agents with LangGraph
		- **[Using Local LLMs](notebooks/LangGraph_React_Local_LLMs.ipynb)** - ReAct Agents with LangGraph with local llms
	- **[Evaluating LangGraph](notebooks/LangGraph_Workfow_Eval.ipynb)** - Evaluating LangGraph
- **[Introduction to CrewAI](notebooks/CrewAI_Hello_World.ipynb)** - CrewAI 101
- **[Introduction to OpenAI Agents](notebooks/OpenAI%20Agents.ipynb)** - OpenAI Agents 101
- **[Introduction to SmolAgents](notebooks/SmolAgents.ipynb)** - HuggingFace's SmolAgents 101

#### Deployment + Production
	
- **[Model Training/Serving with BERT](notebooks/model_serving.ipynb)** - Fine-tuning and running batch data loads with BERT
- **[Deploying models with FastAPI](deploy/)** - using FastAPI to deploy a model
- **[Third party model inference](notebooks/third_party_inference.ipynb)** - Using different types of LLM providers
- **[Using Evaluation to combat AI drift](https://colab.research.google.com/drive/14E6DMP_RGctUPqjI6VMa8EFlggXR7fat?usp=sharing)** - See how drift affects ML models

#### Distillation + Quantization

- **[Quantizing Llama-3 dynamically](https://colab.research.google.com/drive/12RTnrcaXCeAqyGQNbWsrvcqKyOdr0NSm?usp=sharing)** - Using bitsandbytes to quantize a model in real-time on load. We will investigate the differences before and after quantization

	- **[Working with llama.cpp and GGUF (no GPU)](https://colab.research.google.com/drive/15IC5cI-aFbpND5GrvKjAMas1Hmc7M6Rg?usp=sharing)** - See how to load a pre-quantized version of Llama to compare speed and memory usage
	- **[Working with llama.cpp and GGUF (with a GPU)](https://colab.research.google.com/drive/1D6k-BeuF8YRTR8BGi2YYJrSOAZ6cYX8Y?usp=sharing)**

## Running the FastAPI Inference Service (`deploy/`)

The `deploy/` folder contains a minimal FastAPI app that loads a DistilBERT sequence classification model and exposes a `/predict` endpoint.

### Local (Uvicorn)

```bash
cd deploy
pip install -r requirements.txt
uvicorn api:app --reload
```

Then open `http://localhost:8000/docs` to try the API.

Notes:
- The example model currently loads from the Hugging Face Hub. If you need access to a private model, make sure you are authenticated (e.g., `huggingface-cli login`) or configure a `HUGGING_FACE_HUB_TOKEN` in your environment.
- For fully offline usage, you can modify `deploy/api.py` to load a local model directory instead of pulling from the Hub.

### Docker

```bash
cd deploy
docker build . --tag fastapi-demo:1
# On Apple Silicon, you may need a specific platform:
# docker build . --tag fastapi-demo:1 --platform linux/amd64

docker run -p 80:8000 fastapi-demo:1
# If you built with a custom platform, include it on run:
# docker run -p 80:8000 --platform linux/amd64 fastapi-demo:1
```

Open `http://localhost/docs` to access the API docs.

For more details (including Heroku container registry notes), see `deploy/README.md`.

## Streamlit CrewAI Demo (`notebooks/crewai_streamlit/`)

This small Streamlit app demonstrates building agents and tasks with CrewAI.

### Setup

```bash
cd notebooks/crewai_streamlit
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Add your OpenAI key:
- Preferred: create `.streamlit/secrets.toml` in this directory with:
	```toml
	[general]
	OPENAI_API_KEY = "your-openai-api-key"
	```
- Alternatively, you can input the key in the UI at runtime.

Run the app:

```bash
streamlit run app.py
```

An example `secrets.toml` is included for reference in `notebooks/crewai_streamlit/`.

## Data

Sample datasets used by notebooks are provided in `data/`. Not all notebooks require these files; consult each notebook’s instructions for expected inputs.

## References

- O’Reilly Live Training: Designing and Deploying LLM Pipelines — details, schedule, and outcomes:
	- https://learning.oreilly.com/live-events/designing-and-deploying-llm-pipelines/0642572014796/

## Instructor

**Sinan Ozdemir** Sinan is a former lecturer of Data Science at Johns Hopkins University and the author of multiple textbooks on data science and machine learning. Additionally, he is the founder of the recently acquired Kylie.ai, an enterprise-grade conversational AI platform with RPA capabilities. He holds a master’s degree in Pure Mathematics from Johns Hopkins University and is based in San Francisco, CA.