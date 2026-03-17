# AutoPrompter: Autonomous Prompt Optimization System

<p align="center">
  <a href="https://heyneo.so" target="_blank">
    <img src="https://img.shields.io/badge/Made%20by-NEO-ff3b30?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJjdXJyZW50Q29sb3IiIHN0cm9rZS13aWR0aD0iMiI+PHBhdGggZD0iTTEyIDJMNCA3djZsOCA1IDgtNXYtNmwtOC01eiIvPjxwYXRoIGQ9Ik00IDEzbDggNSA4LTUiLz48L3N2Zz4=&logoColor=white" alt="Made by NEO">
  </a>
</p>

AutoPrompter is an autonomous system designed to iteratively improve LLM prompts through a closed-loop optimization process. It merges the validation and metrics capabilities of tools like `promptfoo` with the iterative improvement logic of `autoresearch`.

## System Architecture

The system operates in a continuous loop where an **Optimizer LLM** refines prompts for a **Target LLM** based on empirical performance data.

1.  **Dataset Generation**: The Optimizer LLM (Gemini 3.1 Flash Lite) generates a synthetic dataset of input/output pairs based on the task description.
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

4.  Configure OpenRouter API Key:
    Set the `OPENROUTER_API_KEY` environment variable or add it to `/root/.config/openrouter/config`.

## Configuration

The system is configured via YAML files. Key fields include:

- `optimizer_llm`: Model ID and parameters for the optimizer.
- `target_llm`: Model ID and parameters for the target.
- `experiment`: `max_iterations`, `batch_size`, and convergence thresholds.
- `task`: `name`, `description`, and `initial_prompt`.
- `metric`: `type` (e.g., `accuracy`, `semantic_similarity`) and `target_score`.
- `storage`: Paths for the ledger, dataset, and results.

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
