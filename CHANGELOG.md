# Changelog

All notable changes to AutoPrompter will be documented in this file.

## [Unreleased] - 2025-04-08

### Added

#### Web UI Dashboard
- **New Flask-based Web Interface** (`web_ui.py`)
  - Real-time optimization monitoring via Server-Sent Events (SSE)
  - Interactive configuration builder with backend-specific options
  - Live score history charting with Chart.js
  - Log streaming for debugging and monitoring
  - Checkpoint management (save/load optimization states)
  
- **Prompt Diff Visualization**
  - Side-by-side comparison of prompt iterations
  - Highlighted additions, removals, and changes
  - Modal interface for detailed inspection
  
- **Export/Import Functionality**
  - Save complete optimization runs to JSON format
  - Restore saved runs including config, score history, and best prompts
  - Download/upload via web interface

#### Parallel Execution System
- **Parallel Experiment Executor** (`src/parallel_executor.py`)
  - ThreadPoolExecutor for simultaneous candidate evaluation
  - Configurable worker pool (default: max_workers=3)
  - Statistical result aggregation with significance testing
  - Thread-safe experiment tracking

#### Robustness Testing Framework
- **Adversarial Input Generation** (`src/dataset_generator.py`)
  - Automatic generation of robustness variants:
    - Typo injection (character swaps, missing letters)
    - Ambiguous phrasing substitutions
    - Format variations (uppercase, no punctuation)
    - Noise injection
  - Consistency scoring across variants to detect overfitting

### Fixed

#### Optimization System Stability
- **Baseline Evaluation Fix** (`src/optimization_system.py`)
  - `best_score` now initializes from actual baseline prompt evaluation
  - Added distinct "BASELINE ESTABLISHED" logging (separate from "NEW BEST")
  - Previous score properly set from baseline for convergence detection
  
- **Statistical Significance Testing** (`src/optimization_system.py`)
  - Implemented Welch's t-test (unequal variances) for score comparison
  - Bootstrap confidence interval fallback when t-test assumptions fail
  - Only declares "NEW BEST" when improvement is statistically significant (p<0.05)
  - Prevents noise-driven prompt switches

#### Prompt Optimizer Robustness
- **Stagnation Threshold Increase** (`src/prompt_optimizer.py`)
  - Raised from 2 to 5 iterations before triggering diversification
  - Reduces premature exploration due to score noise
  
- **Prompt Complexity Tracking** (`src/prompt_optimizer.py`)
  - New metrics: character length, word count, instruction count
  - Detection of structural elements (numbered steps, bullet points)
  - Composite complexity score (0-1 scale)
  - Warning logs for "longer but not better" patterns (length > 500, score < 0.5)

#### Experiment Ledger
- **Semantic Duplicate Detection** (`src/experiment_ledger.py`)
  - Embedding-based similarity using SentenceTransformer (`all-MiniLM-L6-v2`)
  - Cosine similarity threshold of 0.95 for near-duplicate detection
  - Integrated into `is_duplicate()` as fallback after exact hash check
  - Embeddings stored with each ledger entry

### Changed

- **Web UI Performance Optimizations**
  - Chart data point limit (max 50 points) with throttled updates (2-second intervals)
  - SSE status stream with change detection to reduce unnecessary updates
  - Memory leak fixes: proper EventSource cleanup, bounded log buffers (100 entries)
  - CSS containment for improved rendering performance

- **Model Selection UI**
  - Dynamic backend-specific model inputs:
    - OpenRouter: Dropdown with 25+ current models (Claude 4.6, GPT 5.4, Gemini 3.1, Llama 4, etc.)
    - Ollama: Text input for local model names
    - Llama.cpp: Text input for GGUF file paths

### Technical Details

#### Files Added
- `web_ui.py` - Flask web application
- `templates/index.html` - Web UI template
- `static/js/app.js` - Frontend JavaScript
- `static/css/style.css` - Custom styles
- `src/parallel_executor.py` - Parallel execution system
- `tests/test_optimization_fixes.py` - Integration test suite
- `tests/test_parallel_executor.py` - Parallel execution tests
- `tests/test_robustness.py` - Robustness testing tests

#### Files Modified
- `src/optimization_system.py` - Baseline evaluation, statistical testing, progress callbacks
- `src/prompt_optimizer.py` - Stagnation threshold, complexity tracking
- `src/experiment_ledger.py` - Semantic duplicate detection
- `src/dataset_generator.py` - Robustness variant generation

#### Test Coverage
- 7 integration tests for optimization fixes (100% pass rate)
- Tests for parallel execution
- Tests for robustness framework

---

## [1.0.0] - Initial Release

### Features
- Closed-loop prompt optimization system
- Multiple LLM backend support (OpenRouter, Ollama, llama.cpp)
- Synthetic dataset generation
- Multiple metric types (accuracy, F1, semantic similarity)
- JSON-based experiment ledger
- Context management for optimizer LLM
- YAML configuration system
- Command-line interface

### Core Components
- `main.py` - CLI entry point
- `src/optimization_system.py` - Core optimization loop
- `src/prompt_optimizer.py` - Prompt refinement logic
- `src/experiment_ledger.py` - Experiment tracking
- `src/dataset_generator.py` - Dataset creation
- `src/metrics.py` - Evaluation metrics
- `src/llm_client.py` - LLM API client
- `src/config_manager.py` - Configuration handling
- `src/context_manager.py` - Context window management
