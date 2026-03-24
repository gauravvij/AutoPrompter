# AutoPrompter: Autonomous Prompt Optimization System

<p align="center">
  <a href="https://heyneo.so" target="_blank">
    <img src="https://img.shields.io/badge/Made%20by-NEO-ff3b30?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJjdXJyZW50Q29sb3IiIHN0cm9rZS13aWR0aD0iMiI+PHBhdGggZD0iTTEyIDJMNCA3djZsOCA1IDgtNXYtNmwtOC01eiIvPjxwYXRoIGQ9Ik00IDEzbDggNSA4LTUiLz48L3N2Zz4=&logoColor=white" alt="Made by NEO">
  </a>
</p>

AutoPrompter is an autonomous system designed to iteratively improve LLM prompts through a closed-loop optimization process. It merges the validation and metrics capabilities of tools like `promptfoo` with the iterative improvement logic of `autoresearch`.

## System Architecture

The system operates in a continuous loop where an **Optimizer LLM** refines prompts for a **Target LLM** based on empirical performance data.

1.  **Dataset Generation**: The Optimizer LLM (Gemini 3.1 Flash - customizable through config.yaml) generates a synthetic dataset of input/output pairs based on the task description.
2.  **Iterative Improvement**:
    *   The Target LLM (Qwen 3.5 9b) is tested against the current prompt using the generated dataset.
    *   Performance is measured using a defined metric (Accuracy, F1, Semantic Similarity, etc.).
    *   The Optimizer LLM analyzes failures and successes to generate a refined prompt.
3.  **Experiment Ledger**: Every iteration is recorded in a persistent ledger to prevent duplicate experiments and track progress.
4.  **Context Management**: The system manages the history of experiments to provide the Optimizer LLM with relevant context without exceeding window limits.

### Core Components
- **Optimizer LLM**: `google/gemini-3.1-flash-lite-preview` (via OpenRouter)
- **Target LLM**: `qwen/qwen3.5-9b` (via OpenRouter)
- **Metrics**: Automated evaluation against expected outputs.
- **Ledger**: JSON-based tracking of all experiments.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/gauravvij/AutoPrompter.git
    cd AutoPrompter
    ```

2.  Create and activate a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4.  Configure API Key (for OpenRouter backend):
    Set the `OPENROUTER_API_KEY` environment variable or add it to `/root/.config/openrouter/config`.

    **Note:** API key is only required when using OpenRouter backend. Local backends (Ollama, llama.cpp) do not require API keys.

## Configuration

The system is configured via YAML files. Key fields include:

- `optimizer_llm`: Model ID and parameters for the optimizer.
- `target_llm`: Model ID and parameters for the target.
- `experiment`: `max_iterations`, `batch_size`, and convergence thresholds.
- `task`: `name`, `description`, and `initial_prompt`.
- `metric`: `type` (e.g., `accuracy`, `semantic_similarity`) and `target_score`.
- `storage`: Paths for the ledger, dataset, and results.

### Supported Backends

AutoPrompter supports multiple LLM backends:

1. **OpenRouter** (default): Cloud-based LLM access via OpenRouter API
2. **Ollama**: Local LLM inference using Ollama server
3. **llama.cpp**: Local LLM inference using llama.cpp server

#### OpenRouter Configuration (Default)

```yaml
optimizer_llm:
  backend: "openrouter"
  model: "google/gemini-3.1-flash-lite-preview"
  api_base: "https://openrouter.ai/api/v1"
  temperature: 0.7
  max_tokens: 4096
```

#### Ollama Configuration

```yaml
optimizer_llm:
  backend: "ollama"
  model: "llama3.2"
  host: "http://localhost"
  port: 11434
  temperature: 0.7
  max_tokens: 4096
```

#### llama.cpp Configuration

```yaml
optimizer_llm:
  backend: "llama_cpp"
  model: "llama-3.2-3b"
  host: "http://localhost"
  port: 8080
  temperature: 0.7
  max_tokens: 4096
```

### Setting Up Local Backends

#### Ollama Setup

1. Install Ollama: https://ollama.ai
2. Pull a model:
   ```bash
   ollama pull llama3.2
   ```
3. Start the server:
   ```bash
   ollama serve
   ```
4. Use `config_ollama.yaml` or configure your own with `backend: "ollama"`

#### llama.cpp Setup

1. Build llama.cpp from source: https://github.com/ggerganov/llama.cpp
2. Download a GGUF model (e.g., from https://huggingface.co/TheBloke)
3. Start the server:
   ```bash
   ./llama-server -m llama-3.2-3b.Q4_K_M.gguf -c 4096 --host 0.0.0.0 --port 8080
   ```
4. Use `config_llama_cpp.yaml` or configure your own with `backend: "llama_cpp"`

#### Auto-Detection

You can also use `backend: "auto"` to automatically detect the available backend:

```yaml
optimizer_llm:
  backend: "auto"
  model: "llama3.2"
  host: "http://localhost"
  port: 11434
```

The system will probe both Ollama and llama.cpp endpoints to determine which is available.

## Usage

Run the optimization process using the main entry point:

```bash
python main.py --config config.yaml
```

### Specific Task Examples

The repository includes pre-configured tasks:

- **Blogging**: Optimize prompts for high-quality blog post generation.
  ```bash
  python main.py --config config_blogging.yaml
  ```
- **Math**: Optimize prompts for solving complex mathematical problems.
  ```bash
  python main.py --config config_math.yaml
  ```
- **Reasoning**: Optimize prompts for logical reasoning and chain-of-thought tasks.
  ```bash
  python main.py --config config_reasoning.yaml
  ```

### Using Local Backends

- **Ollama**: Use the Ollama backend for local inference
  ```bash
  python main.py --config config_ollama.yaml
  ```

- **llama.cpp**: Use the llama.cpp backend for local inference
  ```bash
  python main.py --config config_llama_cpp.yaml
  ```

**Note:** Make sure your local backend server is running before starting the optimization process.

### Command Line Overrides

You can override configuration values directly from the CLI:

```bash
python main.py --config config.yaml --max-iterations 50 --override experiment.batch_size=10
```

## License

MIT

---

<p align="center">
  <a href="https://heyneo.so" target="_blank">
    <img src="https://img.shields.io/badge/Made%20by-NEO-ff3b30?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJjdXJyZW50Q29sb3IiIHN0cm9rZS13aWR0aD0iMiI+PHBhdGggZD0iTTEyIDJMNCA3djZsOCA1IDgtNXYtNmwtOC01eiIvPjxwYXRoIGQ9Ik00IDEzbDggNSA4LTUiLz48L3N2Zz4=&logoColor=white" alt="Made by NEO">
  </a>
</p>

<p align="center">
  <em>NEO - A fully autonomous AI Engineer</em>
</p>
