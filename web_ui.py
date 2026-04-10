#!/usr/bin/env python3
"""
AutoPrompter Web UI - Flask Backend with Real-time Updates

Provides:
- REST API for configuration, status, and checkpoint management
- Server-Sent Events (SSE) for real-time log streaming
- Web interface for visual config builder, dashboard, and results
"""

import os
import sys
import json
import time
import logging
import queue
import threading
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import asdict

from flask import Flask, render_template, jsonify, request, Response, stream_with_context
from flask_cors import CORS

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config_manager import Config, load_config, ExperimentConfig, TaskConfig, MetricConfig, ContextConfig, StorageConfig, LLMConfig
from optimization_system import PromptOptimizationSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('web_ui.log')
    ]
)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Global state
class AppState:
    def __init__(self):
        self.current_config: Optional[Config] = None
        self.optimization_system: Optional[PromptOptimizationSystem] = None
        self.optimization_thread: Optional[threading.Thread] = None
        self.is_running: bool = False
        self.log_queue: queue.Queue = queue.Queue()
        self.current_iteration: int = 0
        self.max_iterations: int = 0
        self.best_score: float = 0.0
        self.best_prompt: str = ""
        self.best_prompt_iteration: int = 0      # Which iteration produced the best prompt
        self.current_test_prompt: str = ""        # Prompt being evaluated RIGHT NOW
        self.previous_best_prompt: str = ""  # For diff visualization
        self.score_history: List[Dict[str, Any]] = []
        self.all_iterations: List[Dict[str, Any]] = []  # For export
        self.start_time: Optional[float] = None
        self.error_message: Optional[str] = None
        self.final_report: Optional[Dict[str, Any]] = None
        self.recent_logs: List[Dict[str, Any]] = []  # Rolling buffer for polling fallback

    def reset(self):
        self.is_running = False
        self.current_iteration = 0
        self.max_iterations = 0
        self.best_score = 0.0
        self.best_prompt = ""
        self.best_prompt_iteration = 0
        self.current_test_prompt = ""
        self.previous_best_prompt = ""
        self.score_history = []
        self.all_iterations = []
        self.start_time = None
        self.error_message = None
        self.final_report = None
        self.recent_logs = []
        # Clear log queue
        while not self.log_queue.empty():
            try:
                self.log_queue.get_nowait()
            except queue.Empty:
                break

app_state = AppState()

# Custom log handler to capture logs for SSE
class SSELogHandler(logging.Handler):
    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue
        
    def emit(self, record):
        try:
            # Skip noisy werkzeug HTTP request logs (200/304/etc.) — keep errors
            if record.name == 'werkzeug' and record.levelno < logging.WARNING:
                return

            # Strip ANSI escape codes from the message
            import re
            raw_msg = self.format(record)
            clean_msg = re.sub(r'\x1b\[[0-9;]*[mGKHF]', '', raw_msg)

            log_entry = {
                'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                'level': record.levelname,
                'message': clean_msg,
                'logger': record.name
            }
            self.log_queue.put(log_entry)
            # Also keep in rolling buffer for HTTP polling fallback
            app_state.recent_logs.append(log_entry)
            if len(app_state.recent_logs) > 200:
                app_state.recent_logs = app_state.recent_logs[-200:]
        except Exception:
            self.handleError(record)

# Add SSE log handler to root logger
sse_handler = SSELogHandler(app_state.log_queue)
sse_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
sse_handler.setFormatter(formatter)
logging.getLogger().addHandler(sse_handler)


def progress_callback(iteration: int, best_score: float, best_prompt: str, current_score, current_prompt: str = ""):
    """Callback called by the optimizer.

    When current_score is None it is a 'pre-run' signal (iteration just started,
    experiment not yet evaluated) — we update current_test_prompt but do NOT add
    to score_history so the chart gets no duplicate points.
    When current_score is a float the experiment is finished and we record results.
    """
    # Update current prompt being tested (always)
    if current_prompt:
        app_state.current_test_prompt = current_prompt
    app_state.current_iteration = iteration

    # Pre-run ping — only update the "testing now" display, nothing else
    if current_score is None:
        return

    # Detect when the best prompt actually improved
    if best_prompt != app_state.best_prompt and app_state.best_prompt:
        app_state.previous_best_prompt = app_state.best_prompt
    if best_prompt != app_state.best_prompt:
        app_state.best_prompt_iteration = iteration

    app_state.best_score = best_score
    app_state.best_prompt = best_prompt

    # Add to score history (one entry per completed experiment)
    history_entry = {
        'iteration': iteration,
        'best_score': best_score,
        'current_score': float(current_score),
        'timestamp': time.time()
    }
    app_state.score_history.append(history_entry)

    # Add to all iterations for export
    app_state.all_iterations.append({
        'iteration': iteration,
        'best_score': best_score,
        'current_score': float(current_score),
        'best_prompt': best_prompt,
        'timestamp': time.time()
    })

    if len(app_state.score_history) > 100:
        app_state.score_history = app_state.score_history[-100:]

    # Push a score event immediately into the log queue so the log SSE
    # delivers it to the client in real-time (chart syncs with logs).
    app_state.log_queue.put({
        'type': 'score',
        'iteration': iteration,
        'best_score': best_score,
        'current_score': float(current_score),
        'timestamp': time.time()
    })

def run_optimization_in_thread(config: Config):
    """Run optimization in background thread."""
    try:
        app_state.is_running = True
        app_state.start_time = time.time()
        app_state.error_message = None
        app_state.final_report = None
        app_state.current_iteration = 0
        app_state.max_iterations = config.experiment.max_iterations
        app_state.best_score = 0.0
        app_state.best_prompt = config.task.initial_prompt  # Show initial prompt immediately
        app_state.previous_best_prompt = ""
        app_state.score_history = []
        app_state.all_iterations = []
        app_state.recent_logs = []
        
        logger.info("Starting optimization in background thread...")
        
        system = PromptOptimizationSystem(config, progress_callback=progress_callback)
        app_state.optimization_system = system
        
        # Run optimization
        report = system.run()
        
        app_state.final_report = report
        
        # Final state update from report
        if report:
            app_state.best_score = report.get('final_score', 0)
            app_state.best_prompt = report.get('best_prompt', '')
            app_state.current_iteration = report.get('total_iterations', 0)
        
        app_state.is_running = False
        
        logger.info("Optimization completed successfully")
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}", exc_info=True)
        app_state.error_message = str(e)
        app_state.is_running = False



def load_state_from_files():
    """
    Reconstruct dashboard state from checkpoint/report files written by main.py.
    Called when no live web-started run is active in app_state.
    """
    try:
        results_dirs = [
            d for d in os.listdir('.')
            if os.path.isdir(d) and d.startswith('results')
        ]
        if not results_dirs:
            return None

        results_dirs.sort(key=lambda d: os.path.getmtime(d), reverse=True)
        results_dir = results_dirs[0]

        state = {
            'is_running': False,
            'current_iteration': 0,
            'max_iterations': 0,
            'best_score': 0.0,
            'best_prompt': '',
            'previous_best_prompt': '',
            'elapsed_time': 0,
            'score_history': [],
            'error_message': None,
            'final_report': None,
            'has_final_report': False,
            'recent_logs': [],  # No historical logs available from files
        }

        # Final report is the most authoritative source
        report_path = os.path.join(results_dir, 'final_report.json')
        if os.path.exists(report_path):
            with open(report_path) as f:
                report = json.load(f)
            state['final_report'] = report
            state['has_final_report'] = True
            state['best_score'] = float(report.get('final_score', 0) or report.get('best_score', 0))
            state['best_prompt'] = report.get('best_prompt', '')
            total = report.get('total_iterations', 0)
            state['current_iteration'] = total
            state['max_iterations'] = total  # Completed run: cur == max

        # Build score history from experiment ledger (most complete source).
        # Filter to the most recent run by grouping records that share contiguous
        # timestamps (records within 6 hours of the most-recent record).
        ledger_path = 'experiment_ledger.json'
        if os.path.exists(ledger_path):
            try:
                with open(ledger_path) as f:
                    ledger = json.load(f)
                records = ledger.get('records', [])
                if records:
                    # Find most-recent record time, keep records within 6h
                    max_ts = max(float(r.get('timestamp', 0)) for r in records)
                    recent = [r for r in records
                              if max_ts - float(r.get('timestamp', 0)) < 6 * 3600]
                    recent.sort(key=lambda r: (r.get('iteration', 0), float(r.get('timestamp', 0))))
                    # Deduplicate: keep last record per iteration
                    seen_iters = {}
                    for rec in recent:
                        seen_iters[rec.get('iteration', 0)] = rec
                    best_so_far = 0.0
                    for it, rec in sorted(seen_iters.items()):
                        score = float(rec.get('mean_score', 0))
                        if score > best_so_far:
                            best_so_far = score
                        state['score_history'].append({
                            'iteration': it,
                            'best_score': best_so_far,
                            'current_score': score,
                            'timestamp': time.time(),
                        })
            except Exception as e:
                logger.warning(f"Could not read ledger for score history: {e}")

        # Fall back to checkpoints if ledger gave nothing
        if not state['score_history']:
            checkpoints = []
            for filename in os.listdir(results_dir):
                if filename.startswith('checkpoint_') and filename.endswith('.json'):
                    path = os.path.join(results_dir, filename)
                    try:
                        with open(path) as f:
                            cp = json.load(f)
                        checkpoints.append(cp)
                    except Exception:
                        pass
            checkpoints.sort(key=lambda x: x.get('iteration', 0))
            for i, cp in enumerate(checkpoints):
                prev_score = checkpoints[i - 1].get('best_score', 0) if i > 0 else 0
                state['score_history'].append({
                    'iteration': cp.get('iteration', i + 1),
                    'best_score': float(cp.get('best_score', 0)),
                    'current_score': float(prev_score),
                    'timestamp': time.time(),
                })
            # Fallback: if report had no prompt, take from last checkpoint
            if checkpoints and not state['best_prompt']:
                last = checkpoints[-1]
                state['best_score'] = float(last.get('best_score', 0))
                state['best_prompt'] = last.get('best_prompt', '')
                state['current_iteration'] = last.get('iteration', 0)

        return state
    except Exception as e:
        logger.warning(f"Could not load state from files: {e}")
        return None

def generate_status_stream():
    """Generate SSE stream for status updates."""
    while True:
        try:
            elapsed = 0
            if app_state.start_time:
                elapsed = time.time() - app_state.start_time

            # Use live state if available; otherwise read from checkpoint files
            if app_state.is_running or app_state.best_prompt or app_state.final_report:
                data = {
                    'type': 'status',
                    'is_running': app_state.is_running,
                    'current_iteration': app_state.current_iteration,
                    'max_iterations': app_state.max_iterations,
                    'best_score': app_state.best_score,
                    'best_prompt': app_state.best_prompt,
                    'best_prompt_iteration': app_state.best_prompt_iteration,
                    'current_test_prompt': app_state.current_test_prompt,
                    'previous_best_prompt': app_state.previous_best_prompt,
                    'elapsed_time': round(elapsed, 2),
                    'score_history': app_state.score_history,
                    'error_message': app_state.error_message,
                    'final_report': app_state.final_report,
                    'has_final_report': app_state.final_report is not None,
                    'recent_logs': app_state.recent_logs[-50:]
                }
            else:
                file_state = load_state_from_files()
                if file_state:
                    data = {'type': 'status', **file_state}
                else:
                    data = {
                        'type': 'status',
                        'is_running': False,
                        'current_iteration': 0,
                        'best_score': 0.0,
                        'best_prompt': '',
                        'previous_best_prompt': '',
                        'elapsed_time': 0,
                        'score_history': [],
                        'error_message': None,
                        'final_report': None,
                        'has_final_report': False,
                        'recent_logs': [],
                    }

            yield f"data: {json.dumps(data)}\n\n"
            time.sleep(2)

        except GeneratorExit:
            break
        except Exception as e:
            logger.error(f"Error in status stream: {e}")
            time.sleep(1)

def generate_log_stream():
    """Generate SSE stream for log entries."""
    # Send initial heartbeat
    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
    
    while True:
        try:
            # Check for new log entries
            try:
                entry = app_state.log_queue.get(timeout=1)
                entry['type'] = 'log'
                yield f"data: {json.dumps(entry)}\n\n"
            except queue.Empty:
                # Send heartbeat to keep connection alive
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
                
        except GeneratorExit:
            break
        except Exception as e:
            logger.error(f"Error in log stream: {e}")
            time.sleep(0.1)


# ==================== API Routes ====================

@app.route('/')
def index():
    """Main page - redirects to dashboard."""
    return render_template('index.html')


@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration."""
    if app_state.current_config:
        return jsonify({
            'status': 'success',
            'config': asdict(app_state.current_config)
        })
    
    # Return default config
    default_config = {
        'optimizer_llm': {
            'model': 'meta-llama/llama-3.1-8b-instruct',
            'backend': 'openrouter',
            'temperature': 0.7,
            'max_tokens': 2048
        },
        'target_llm': {
            'model': 'mistralai/mistral-nemo',
            'backend': 'openrouter',
            'temperature': 0.1,
            'max_tokens': 128
        },
        'experiment': {
            'max_iterations': 5,
            'convergence_threshold': 0.95,
            'min_improvement': 0.01,
            'batch_size': 5,
            'parallel_enabled': False,
            'parallel_workers': 3,
            'parallel_candidates': 3
        },
        'task': {
            'name': 'text_classification',
            'description': 'Classify text into positive or negative sentiment',
            'initial_prompt': 'Analyze the sentiment of the following text and respond with only "positive" or "negative".'
        },
        'metric': {
            'type': 'accuracy',
            'target_score': 0.95
        },
        'context': {
            'max_experiments_in_context': 20,
            'compression_threshold': 50
        },
        'storage': {
            'ledger_file': 'experiment_ledger.json',
            'dataset_file': 'generated_dataset.json',
            'results_dir': 'results',
            'checkpoint_interval': 10
        },
        'robustness': {
            'enabled': False,
            'num_variants': 3,
            'score_threshold': 0.9,
            'strategies': ['typos', 'paraphrase', 'format_variation', 'ambiguous']
        }
    }
    
    return jsonify({
        'status': 'success',
        'config': default_config
    })


@app.route('/api/config', methods=['POST'])
def save_config():
    """Save configuration from form data."""
    try:
        data = request.get_json()
        
        # Build config from form data
        config = Config(
            optimizer_llm=LLMConfig(**data.get('optimizer_llm', {})),
            target_llm=LLMConfig(**data.get('target_llm', {})),
            experiment=ExperimentConfig(**data.get('experiment', {})),
            task=TaskConfig(**data.get('task', {})),
            metric=MetricConfig(**data.get('metric', {})),
            context=ContextConfig(**data.get('context', {})),
            storage=StorageConfig(**data.get('storage', {}))
        )
        
        # Validate
        errors = config.validate()
        if errors:
            return jsonify({
                'status': 'error',
                'message': 'Validation failed',
                'errors': errors
            }), 400
        
        app_state.current_config = config
        
        # Save to file
        config_path = data.get('config_path', 'config_web.yaml')
        config.to_yaml(config_path)
        
        return jsonify({
            'status': 'success',
            'message': f'Configuration saved to {config_path}'
        })
        
    except Exception as e:
        logger.error(f"Failed to save config: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current optimization status (HTTP polling fallback)."""
    elapsed = 0
    if app_state.start_time:
        elapsed = time.time() - app_state.start_time

    # If no live data, read from checkpoint/report files (e.g. run via CLI)
    if not app_state.is_running and not app_state.best_prompt and not app_state.final_report:
        file_state = load_state_from_files()
        if file_state:
            return jsonify({'status': 'success', 'data': file_state})

    return jsonify({
        'status': 'success',
        'data': {
            'is_running': app_state.is_running,
            'current_iteration': app_state.current_iteration,
            'max_iterations': app_state.max_iterations,
            'best_score': app_state.best_score,
            'best_prompt': app_state.best_prompt,
            'best_prompt_iteration': app_state.best_prompt_iteration,
            'current_test_prompt': app_state.current_test_prompt,
            'previous_best_prompt': app_state.previous_best_prompt,
            'elapsed_time': round(elapsed, 2),
            'score_history': app_state.score_history,
            'error_message': app_state.error_message,
            'final_report': app_state.final_report,
            'has_final_report': app_state.final_report is not None,
            'recent_logs': app_state.recent_logs[-50:]
        }
    })


@app.route('/api/status/stream')
def status_stream():
    """Server-Sent Events stream for real-time status updates."""
    return Response(
        stream_with_context(generate_status_stream()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


@app.route('/api/logs/stream')
def logs_stream():
    """Server-Sent Events stream for real-time log updates."""
    return Response(
        stream_with_context(generate_log_stream()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )



@app.route('/api/logs', methods=['GET'])
def get_logs():
    """Return recent log entries for polling fallback (when SSE is unavailable)."""
    since = request.args.get('since', 0, type=float)
    logs = [l for l in app_state.recent_logs if l.get('timestamp', '') > str(since)[:23]] if since else app_state.recent_logs[-100:]
    return jsonify({'status': 'success', 'logs': logs})


@app.route('/api/start', methods=['POST'])
def start_optimization():
    """Start optimization with current config."""
    if app_state.is_running:
        return jsonify({
            'status': 'error',
            'message': 'Optimization already running'
        }), 400
    
    if not app_state.current_config:
        return jsonify({
            'status': 'error',
            'message': 'No configuration loaded. Please configure first.'
        }), 400
    
    try:
        app_state.reset()
        # Mark as running synchronously BEFORE the thread starts so the SSE
        # stream immediately sees is_running=True and doesn't fall through to
        # load_state_from_files() which would return the previous run's
        # has_final_report=True and cause the client to kill the stream.
        app_state.is_running = True
        app_state.start_time = time.time()
        app_state.max_iterations = app_state.current_config.experiment.max_iterations

        # Start optimization in background thread
        thread = threading.Thread(
            target=run_optimization_in_thread,
            args=(app_state.current_config,),
            daemon=True
        )
        app_state.optimization_thread = thread
        thread.start()
        
        return jsonify({
            'status': 'success',
            'message': 'Optimization started'
        })
        
    except Exception as e:
        logger.error(f"Failed to start optimization: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/stop', methods=['POST'])
def stop_optimization():
    """Stop current optimization."""
    if not app_state.is_running:
        return jsonify({
            'status': 'error',
            'message': 'No optimization running'
        }), 400
    
    # Note: Python threads can't be forcefully stopped
    # We can only signal to stop gracefully
    app_state.is_running = False
    
    return jsonify({
        'status': 'success',
        'message': 'Stop signal sent (will stop after current iteration)'
    })


@app.route('/api/checkpoints', methods=['GET'])
def list_checkpoints():
    """List available checkpoints from all results directories."""
    try:
        # Collect all results-like directories to search
        search_dirs = set()
        if app_state.current_config:
            search_dirs.add(app_state.current_config.storage.results_dir)
        # Also scan the working directory for any results* folders
        for entry in os.listdir('.'):
            if os.path.isdir(entry) and (entry == 'results' or entry.startswith('results')):
                search_dirs.add(entry)

        checkpoints = []
        for results_dir in search_dirs:
            if not os.path.exists(results_dir):
                continue
            for filename in os.listdir(results_dir):
                if filename.startswith('checkpoint_') and filename.endswith('.json'):
                    path = os.path.join(results_dir, filename)
                    try:
                        with open(path, 'r') as f:
                            data = json.load(f)
                        checkpoints.append({
                            'filename': f"{results_dir}/{filename}",
                            'path': path,
                            'iteration': data.get('iteration', 0),
                            'best_score': float(data.get('best_score', 0) or 0),
                            'timestamp': os.path.getmtime(path)
                        })
                    except Exception:
                        pass

        checkpoints.sort(key=lambda x: x['timestamp'], reverse=True)

        return jsonify({
            'status': 'success',
            'checkpoints': checkpoints
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/checkpoints/load', methods=['POST'])
def load_checkpoint():
    """Load a checkpoint and resume from it."""
    try:
        data = request.get_json()
        checkpoint_path = data.get('checkpoint_path')
        
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            return jsonify({
                'status': 'error',
                'message': 'Checkpoint not found'
            }), 404
        
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)
        
        # Restore state
        app_state.current_iteration = checkpoint_data.get('iteration', 0)
        app_state.best_score = float(checkpoint_data.get('best_score', 0) or 0)
        app_state.best_prompt = checkpoint_data.get('best_prompt', '')
        app_state.best_prompt_iteration = checkpoint_data.get('iteration', 0)

        # Reconstruct Config from the dict stored in the checkpoint (same logic as from_yaml)
        config_dict = checkpoint_data.get('config', {})
        if config_dict:
            try:
                opt_data = config_dict.get('optimizer_llm', {})
                tgt_data = config_dict.get('target_llm', {})
                opt_backend = opt_data.get('backend', 'openrouter')
                tgt_backend = tgt_data.get('backend', 'openrouter')

                from config_manager import LocalLLMConfig
                optimizer_llm = LocalLLMConfig(**opt_data) if opt_backend in ('ollama', 'llama_cpp', 'auto') else LLMConfig(**opt_data)
                target_llm   = LocalLLMConfig(**tgt_data) if tgt_backend in ('ollama', 'llama_cpp', 'auto') else LLMConfig(**tgt_data)

                app_state.current_config = Config(
                    optimizer_llm=optimizer_llm,
                    target_llm=target_llm,
                    experiment=ExperimentConfig(**config_dict.get('experiment', {})),
                    task=TaskConfig(**config_dict.get('task', {})),
                    metric=MetricConfig(**config_dict.get('metric', {})),
                    context=ContextConfig(**config_dict.get('context', {})),
                    storage=StorageConfig(**config_dict.get('storage', {}))
                )
            except Exception as cfg_err:
                logger.warning(f"Could not reconstruct config from checkpoint: {cfg_err}")

        # Rebuild score history from the checkpoint's ledger snapshot if available
        score_history = checkpoint_data.get('score_history', [])
        if score_history:
            app_state.score_history = score_history

        return jsonify({
            'status': 'success',
            'message': f'Checkpoint loaded: iteration {app_state.current_iteration}, score {app_state.best_score:.3f}',
            'data': {
                'iteration': app_state.current_iteration,
                'best_score': app_state.best_score,
                'best_prompt': app_state.best_prompt
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/results', methods=['GET'])
def get_results():
    """Get final results."""
    if not app_state.final_report:
        return jsonify({
            'status': 'error',
            'message': 'No results available yet'
        }), 404
    
    return jsonify({
        'status': 'success',
        'report': app_state.final_report
    })


@app.route('/api/export', methods=['POST'])
def export_results():
    """Export results to file."""
    try:
        data = request.get_json()
        format_type = data.get('format', 'json')
        
        if not app_state.final_report:
            return jsonify({
                'status': 'error',
                'message': 'No results to export'
            }), 400
        
        export_path = f"results/export.{format_type}"
        
        if format_type == 'json':
            with open(export_path, 'w') as f:
                json.dump(app_state.final_report, f, indent=2)
        elif format_type == 'txt':
            with open(export_path, 'w') as f:
                f.write("AUTOPROMPTER RESULTS\n")
                f.write("=" * 70 + "\n\n")
                f.write(f"Task: {app_state.final_report.get('task', 'N/A')}\n")
                f.write(f"Iterations: {app_state.final_report.get('total_iterations', 0)}\n")
                f.write(f"Initial Score: {app_state.final_report.get('initial_score', 0):.3f}\n")
                f.write(f"Final Score: {app_state.final_report.get('final_score', 0):.3f}\n")
                f.write(f"Improvement: {app_state.final_report.get('improvement', 0):.3f}\n\n")
                f.write("BEST PROMPT:\n")
                f.write("-" * 70 + "\n")
                f.write(app_state.final_report.get('best_prompt', 'N/A'))
        
        return jsonify({
            'status': 'success',
            'message': f'Results exported to {export_path}',
            'path': export_path
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/export/full', methods=['GET'])
def export_full_state():
    """Export full optimization state including all iterations."""
    try:
        export_data = {
            'version': '1.0',
            'exported_at': datetime.now().isoformat(),
            'config': asdict(app_state.current_config) if app_state.current_config else None,
            'state': {
                'current_iteration': app_state.current_iteration,
                'best_score': app_state.best_score,
                'best_prompt': app_state.best_prompt,
                'previous_best_prompt': app_state.previous_best_prompt,
                'score_history': app_state.score_history,
                'all_iterations': app_state.all_iterations,
                'start_time': app_state.start_time,
                'error_message': app_state.error_message
            },
            'final_report': app_state.final_report
        }
        
        return jsonify({
            'status': 'success',
            'data': export_data
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/import', methods=['POST'])
def import_full_state():
    """Import full optimization state."""
    try:
        data = request.get_json()
        import_data = data.get('data')
        
        if not import_data:
            return jsonify({
                'status': 'error',
                'message': 'No data provided'
            }), 400
        
        # Restore config
        config_dict = import_data.get('config')
        if config_dict:
            try:
                opt_data = config_dict.get('optimizer_llm', {})
                tgt_data = config_dict.get('target_llm', {})
                opt_backend = opt_data.get('backend', 'openrouter')
                tgt_backend = tgt_data.get('backend', 'openrouter')
                from config_manager import LocalLLMConfig
                optimizer_llm = LocalLLMConfig(**opt_data) if opt_backend in ('ollama', 'llama_cpp', 'auto') else LLMConfig(**opt_data)
                target_llm   = LocalLLMConfig(**tgt_data) if tgt_backend in ('ollama', 'llama_cpp', 'auto') else LLMConfig(**tgt_data)
                app_state.current_config = Config(
                    optimizer_llm=optimizer_llm,
                    target_llm=target_llm,
                    experiment=ExperimentConfig(**config_dict.get('experiment', {})),
                    task=TaskConfig(**config_dict.get('task', {})),
                    metric=MetricConfig(**config_dict.get('metric', {})),
                    context=ContextConfig(**config_dict.get('context', {})),
                    storage=StorageConfig(**config_dict.get('storage', {}))
                )
            except Exception as cfg_err:
                logger.warning(f"Could not reconstruct config from import: {cfg_err}")
        
        # Restore state
        state = import_data.get('state', {})
        app_state.current_iteration = state.get('current_iteration', 0)
        app_state.best_score = state.get('best_score', 0)
        app_state.best_prompt = state.get('best_prompt', '')
        app_state.previous_best_prompt = state.get('previous_best_prompt', '')
        app_state.score_history = state.get('score_history', [])
        app_state.all_iterations = state.get('all_iterations', [])
        app_state.start_time = state.get('start_time')
        app_state.error_message = state.get('error_message')
        
        # Restore final report
        app_state.final_report = import_data.get('final_report')
        
        return jsonify({
            'status': 'success',
            'message': 'Optimization state imported successfully',
            'data': {
                'iteration': app_state.current_iteration,
                'best_score': app_state.best_score,
                'best_prompt': app_state.best_prompt
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=int(os.getenv('PORT', 7860)))
    args = parser.parse_args()
    app.run(debug=False, host='0.0.0.0', port=args.port, threaded=True)
