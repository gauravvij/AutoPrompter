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
        self.best_score: float = 0.0
        self.best_prompt: str = ""
        self.previous_best_prompt: str = ""  # For diff visualization
        self.score_history: List[Dict[str, Any]] = []
        self.all_iterations: List[Dict[str, Any]] = []  # For export
        self.start_time: Optional[float] = None
        self.error_message: Optional[str] = None
        self.final_report: Optional[Dict[str, Any]] = None
        
    def reset(self):
        self.is_running = False
        self.current_iteration = 0
        self.best_score = 0.0
        self.best_prompt = ""
        self.previous_best_prompt = ""
        self.score_history = []
        self.all_iterations = []
        self.start_time = None
        self.error_message = None
        self.final_report = None
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
            log_entry = {
                'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                'level': record.levelname,
                'message': self.format(record),
                'logger': record.name
            }
            self.log_queue.put(log_entry)
        except Exception:
            self.handleError(record)

# Add SSE log handler to root logger
sse_handler = SSELogHandler(app_state.log_queue)
sse_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
sse_handler.setFormatter(formatter)
logging.getLogger().addHandler(sse_handler)


def progress_callback(iteration: int, best_score: float, best_prompt: str, current_score: float):
    """Callback to update app_state during optimization."""
    # Track previous best prompt for diff visualization
    if best_prompt != app_state.best_prompt and app_state.best_prompt:
        app_state.previous_best_prompt = app_state.best_prompt
    
    app_state.current_iteration = iteration
    app_state.best_score = best_score
    app_state.best_prompt = best_prompt
    
    # Add to score history
    history_entry = {
        'iteration': iteration,
        'best_score': best_score,
        'current_score': current_score,
        'timestamp': time.time()
    }
    app_state.score_history.append(history_entry)
    
    # Add to all iterations for export
    app_state.all_iterations.append({
        'iteration': iteration,
        'best_score': best_score,
        'current_score': current_score,
        'best_prompt': best_prompt,
        'timestamp': time.time()
    })
    
    # Limit history size to prevent memory bloat
    if len(app_state.score_history) > 100:
        app_state.score_history = app_state.score_history[-100:]

def run_optimization_in_thread(config: Config):
    """Run optimization in background thread."""
    try:
        app_state.is_running = True
        app_state.start_time = time.time()
        app_state.error_message = None
        app_state.final_report = None
        app_state.current_iteration = 0
        app_state.best_score = 0.0
        app_state.best_prompt = ""
        app_state.previous_best_prompt = ""
        app_state.score_history = []
        app_state.all_iterations = []
        
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


def generate_status_stream():
    """Generate SSE stream for status updates."""
    while True:
        try:
            elapsed = 0
            if app_state.start_time:
                elapsed = time.time() - app_state.start_time
            
            data = {
                'type': 'status',
                'is_running': app_state.is_running,
                'current_iteration': app_state.current_iteration,
                'best_score': app_state.best_score,
                'best_prompt': app_state.best_prompt,
                'previous_best_prompt': app_state.previous_best_prompt,
                'elapsed_time': round(elapsed, 2),
                'score_history': app_state.score_history,
                'error_message': app_state.error_message,
                'final_report': app_state.final_report,
                'has_final_report': app_state.final_report is not None
            }
            
            yield f"data: {json.dumps(data)}\n\n"
            
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
            'model': 'anthropic/claude-opus-4.6',
            'backend': 'openrouter',
            'temperature': 0.7,
            'max_tokens': 4096
        },
        'target_llm': {
            'model': 'anthropic/claude-sonnet-4.6',
            'backend': 'openrouter',
            'temperature': 0.7,
            'max_tokens': 4096
        },
        'experiment': {
            'max_iterations': 20,
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
    
    return jsonify({
        'status': 'success',
        'data': {
            'is_running': app_state.is_running,
            'current_iteration': app_state.current_iteration,
            'best_score': app_state.best_score,
            'best_prompt': app_state.best_prompt,
            'previous_best_prompt': app_state.previous_best_prompt,
            'elapsed_time': round(elapsed, 2),
            'score_history': app_state.score_history,
            'error_message': app_state.error_message,
            'final_report': app_state.final_report,
            'has_final_report': app_state.final_report is not None
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
    """List available checkpoints."""
    try:
        results_dir = 'results'
        if app_state.current_config:
            results_dir = app_state.current_config.storage.results_dir
        
        checkpoints = []
        if os.path.exists(results_dir):
            for filename in os.listdir(results_dir):
                if filename.startswith('checkpoint_') and filename.endswith('.json'):
                    path = os.path.join(results_dir, filename)
                    try:
                        with open(path, 'r') as f:
                            data = json.load(f)
                        checkpoints.append({
                            'filename': filename,
                            'path': path,
                            'iteration': data.get('iteration', 0),
                            'best_score': data.get('best_score', 0),
                            'timestamp': os.path.getmtime(path)
                        })
                    except:
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
        app_state.best_score = checkpoint_data.get('best_score', 0)
        app_state.best_prompt = checkpoint_data.get('best_prompt', '')
        
        # Reconstruct config
        config_dict = checkpoint_data.get('config', {})
        app_state.current_config = Config.from_dict(config_dict)
        
        return jsonify({
            'status': 'success',
            'message': f'Checkpoint loaded from iteration {app_state.current_iteration}',
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
            app_state.current_config = Config.from_dict(config_dict)
        
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
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
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
        self.best_score: float = 0.0
        self.best_prompt: str = ""
        self.previous_best_prompt: str = ""  # For diff visualization
        self.score_history: List[Dict[str, Any]] = []
        self.all_iterations: List[Dict[str, Any]] = []  # For export
        self.start_time: Optional[float] = None
        self.error_message: Optional[str] = None
        self.final_report: Optional[Dict[str, Any]] = None
        
    def reset(self):
        self.is_running = False
        self.current_iteration = 0
        self.best_score = 0.0
        self.best_prompt = ""
        self.previous_best_prompt = ""
        self.score_history = []
        self.all_iterations = []
        self.start_time = None
        self.error_message = None
        self.final_report = None
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
            log_entry = {
                'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                'level': record.levelname,
                'message': self.format(record),
                'logger': record.name
            }
            self.log_queue.put(log_entry)
        except Exception:
            self.handleError(record)

# Add SSE log handler to root logger
sse_handler = SSELogHandler(app_state.log_queue)
sse_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
sse_handler.setFormatter(formatter)
logging.getLogger().addHandler(sse_handler)


def progress_callback(iteration: int, best_score: float, best_prompt: str, current_score: float):
    """Callback to update app_state during optimization."""
    # Track previous best prompt for diff visualization
    if best_prompt != app_state.best_prompt and app_state.best_prompt:
        app_state.previous_best_prompt = app_state.best_prompt
    
    app_state.current_iteration = iteration
    app_state.best_score = best_score
    app_state.best_prompt = best_prompt
    
    # Add to score history
    history_entry = {
        'iteration': iteration,
        'best_score': best_score,
        'current_score': current_score,
        'timestamp': time.time()
    }
    app_state.score_history.append(history_entry)
    
    # Add to all iterations for export
    app_state.all_iterations.append({
        'iteration': iteration,
        'best_score': best_score,
        'current_score': current_score,
        'best_prompt': best_prompt,
        'timestamp': time.time()
    })
    
    # Limit history size to prevent memory bloat
    if len(app_state.score_history) > 100:
        app_state.score_history = app_state.score_history[-100:]

def run_optimization_in_thread(config: Config):
    """Run optimization in background thread."""
    try:
        app_state.is_running = True
        app_state.start_time = time.time()
        app_state.error_message = None
        app_state.final_report = None
        app_state.current_iteration = 0
        app_state.best_score = 0.0
        app_state.best_prompt = ""
        app_state.previous_best_prompt = ""
        app_state.score_history = []
        app_state.all_iterations = []
        
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


def generate_status_stream():
    """Generate SSE stream for status updates."""
    while True:
        try:
            elapsed = 0
            if app_state.start_time:
                elapsed = time.time() - app_state.start_time
            
            data = {
                'type': 'status',
                'is_running': app_state.is_running,
                'current_iteration': app_state.current_iteration,
                'best_score': app_state.best_score,
                'best_prompt': app_state.best_prompt,
                'previous_best_prompt': app_state.previous_best_prompt,
                'elapsed_time': round(elapsed, 2),
                'score_history': app_state.score_history,
                'error_message': app_state.error_message,
                'final_report': app_state.final_report,
                'has_final_report': app_state.final_report is not None
            }
            
            yield f"data: {json.dumps(data)}\n\n"
            time.sleep(1)  # Send update every second
            
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
            'model': 'anthropic/claude-opus-4.6',
            'backend': 'openrouter',
            'temperature': 0.7,
            'max_tokens': 4096
        },
        'target_llm': {
            'model': 'anthropic/claude-sonnet-4.6',
            'backend': 'openrouter',
            'temperature': 0.7,
            'max_tokens': 4096
        },
        'experiment': {
            'max_iterations': 20,
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
    
    return jsonify({
        'status': 'success',
        'data': {
            'is_running': app_state.is_running,
            'current_iteration': app_state.current_iteration,
            'best_score': app_state.best_score,
            'best_prompt': app_state.best_prompt,
            'previous_best_prompt': app_state.previous_best_prompt,
            'elapsed_time': round(elapsed, 2),
            'score_history': app_state.score_history,
            'error_message': app_state.error_message,
            'final_report': app_state.final_report,
            'has_final_report': app_state.final_report is not None
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
    """List available checkpoints."""
    try:
        results_dir = 'results'
        if app_state.current_config:
            results_dir = app_state.current_config.storage.results_dir
        
        checkpoints = []
        if os.path.exists(results_dir):
            for filename in os.listdir(results_dir):
                if filename.startswith('checkpoint_') and filename.endswith('.json'):
                    path = os.path.join(results_dir, filename)
                    try:
                        with open(path, 'r') as f:
                            data = json.load(f)
                        checkpoints.append({
                            'filename': filename,
                            'path': path,
                            'iteration': data.get('iteration', 0),
                            'best_score': data.get('best_score', 0),
                            'timestamp': os.path.getmtime(path)
                        })
                    except:
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
        app_state.best_score = checkpoint_data.get('best_score', 0)
        app_state.best_prompt = checkpoint_data.get('best_prompt', '')
        
        # Reconstruct config
        config_dict = checkpoint_data.get('config', {})
        app_state.current_config = Config.from_dict(config_dict)
        
        return jsonify({
            'status': 'success',
            'message': f'Checkpoint loaded from iteration {app_state.current_iteration}',
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
            app_state.current_config = Config.from_dict(config_dict)
        
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
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
