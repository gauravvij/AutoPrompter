/**
 * AutoPrompter Web UI - Frontend JavaScript (Performance Optimized)
 * Handles real-time updates, SSE streaming, and user interactions
 * 
 * PERFORMANCE FIXES:
 * - Chart limited to 50 data points maximum
 * - Chart updates throttled to once per 2 seconds
 * - Log entry limit reduced from 500 to 100
 * - EventSource connections properly cleaned up
 * - Score history limited in memory
 * - Debounced dashboard updates
 */

// Global state
let eventSource = null;
let statusEventSource = null;
let scoreChart = null;
let isRunning = false;
let statusPollInterval = null;
let logPollInterval = null;
let lastSeenLogTimestamp = '';
let sseWatchdogTimer = null;
let sseReceivedData = false;

// Client-side elapsed time counter
let elapsedTimerInterval = null;
let elapsedStartTime = null;   // local Date.now() when run started
let elapsedBaseSeconds = 0;    // server-reported elapsed at last status update

// Performance optimization state
const MAX_CHART_POINTS = 50;
const MAX_LOG_ENTRIES = 200;
const CHART_UPDATE_INTERVAL = 1000;
const STATUS_UPDATE_INTERVAL = 2000;
const SSE_WATCHDOG_MS = 5000; // If no SSE message in 5s, fall back to polling
let lastChartUpdate = 0;
let pendingChartUpdate = false;
let scoreHistoryBuffer = [];

// DOM Elements
const elements = {
    // Tabs
    tabBtns: document.querySelectorAll('.tab-btn'),
    tabContents: document.querySelectorAll('.tab-content'),
    
    // Config form
    configForm: document.getElementById('configForm'),
    parallelEnabled: document.getElementById('parallelEnabled'),
    parallelOptions: document.getElementById('parallelOptions'),
    robustnessEnabled: document.getElementById('robustnessEnabled'),
    robustnessOptions: document.getElementById('robustnessOptions'),
    
    // LLM tabs
    llmTabBtns: document.querySelectorAll('.llm-tab-btn'),
    llmContents: document.querySelectorAll('.llm-content'),
    
    // Dashboard
    currentIteration: document.getElementById('currentIteration'),
    bestScore: document.getElementById('bestScore'),
    elapsedTime: document.getElementById('elapsedTime'),
    runStatus: document.getElementById('runStatus'),
    bestPromptBox: document.getElementById('bestPromptBox'),
    bestPromptMeta: document.getElementById('bestPromptMeta'),
    currentTestPromptBox: document.getElementById('currentTestPromptBox'),
    currentPromptMeta: document.getElementById('currentPromptMeta'),
    logsBox: document.getElementById('logsBox'),
    
    // Buttons
    startBtn: document.getElementById('startBtn'),
    stopBtn: document.getElementById('stopBtn'),
    refreshBtn: document.getElementById('refreshBtn'),
    clearLogsBtn: document.getElementById('clearLogsBtn'),
    saveConfigBtn: document.getElementById('saveConfigBtn'),
    loadConfigBtn: document.getElementById('loadConfigBtn'),
    exportStateBtn: document.getElementById('exportStateBtn'),
    importStateBtn: document.getElementById('importStateBtn'),
    showDiffBtn: document.getElementById('showDiffBtn'),
    
    // Checkpoints
    checkpointsList: document.getElementById('checkpointsList'),
    refreshCheckpointsBtn: document.getElementById('refreshCheckpointsBtn'),
    
    // Results
    resultsContent: document.getElementById('resultsContent'),
    resultsActions: document.getElementById('resultsActions'),
    exportJsonBtn: document.getElementById('exportJsonBtn'),
    exportTxtBtn: document.getElementById('exportTxtBtn'),
    
    // Status
    statusDot: document.getElementById('statusDot'),
    statusText: document.getElementById('statusText'),
    
    // Toast
    toastContainer: document.getElementById('toastContainer'),
    
    // Diff modal
    diffModal: document.getElementById('diffModal'),
    diffContent: document.getElementById('diffContent'),
    closeDiffBtn: document.getElementById('closeDiffBtn'),

    // Error banner
    errorBanner: document.getElementById('errorBanner')
};

// Initialize
function init() {
    setupTabs();
    setupLLMTabs();
    setupFormHandlers();
    setupButtonHandlers();
    initChart();
    loadConfig();
    startStatusStream();
    // loadCheckpoints is called via debounce after init completes
    setTimeout(() => loadCheckpoints(), 100);
    
    // Cleanup on page unload
    window.addEventListener('beforeunload', cleanupEventSources);
    
    // Cleanup on visibility change (tab switch)
    document.addEventListener('visibilitychange', handleVisibilityChange);
    
    // Initialize model options based on default backend
    updateModelOptions('optimizer');
    updateModelOptions('target');
}

// Update model options based on backend selection
function updateModelOptions(llmType) {
    const backendSelect = document.getElementById(llmType === 'optimizer' ? 'optBackend' : 'targetBackend');
    const modelSelect = document.getElementById(llmType === 'optimizer' ? 'optModel' : 'targetModel');
    const modelInput = document.getElementById(llmType === 'optimizer' ? 'optModelInput' : 'targetModelInput');
    const helpText = document.getElementById(llmType === 'optimizer' ? 'optBackendHelp' : 'targetBackendHelp');
    
    const backend = backendSelect.value;
    
    if (backend === 'openrouter') {
        // Show dropdown, hide text input
        modelSelect.style.display = 'block';
        modelInput.style.display = 'none';
        modelSelect.disabled = false;
        modelInput.disabled = true;
        helpText.textContent = 'Uses OpenRouter API with 25+ cloud models';
    } else if (backend === 'ollama') {
        // Hide dropdown, show text input for local model name
        modelSelect.style.display = 'none';
        modelInput.style.display = 'block';
        modelSelect.disabled = true;
        modelInput.disabled = false;
        modelInput.placeholder = 'Enter model name (e.g., llama3.1, mistral, codellama)';
        helpText.textContent = 'Requires Ollama running locally (ollama.com)';
    } else if (backend === 'llama_cpp') {
        // Hide dropdown, show text input for GGUF path
        modelSelect.style.display = 'none';
        modelInput.style.display = 'block';
        modelSelect.disabled = true;
        modelInput.disabled = false;
        modelInput.placeholder = 'Enter path to .gguf file or model name';
        helpText.textContent = 'Requires llama-cpp-python installed';
    }
}

// Cleanup EventSource connections
function cleanupEventSources() {
    if (eventSource) { eventSource.close(); eventSource = null; }
    if (statusEventSource) { statusEventSource.close(); statusEventSource = null; }
    if (statusPollInterval) { clearInterval(statusPollInterval); statusPollInterval = null; }
    if (logPollInterval) { clearInterval(logPollInterval); logPollInterval = null; }
    if (sseWatchdogTimer) { clearTimeout(sseWatchdogTimer); sseWatchdogTimer = null; }
    stopElapsedTimer();
}

// Handle tab visibility change
function handleVisibilityChange() {
    if (document.hidden) {
        // Page hidden - close connections to save resources
        cleanupEventSources();
    } else {
        // Page visible - reconnect
        startStatusStream();
        // Log stream will be started by status stream if running
    }
}

// Tab Navigation
function setupTabs() {
    elements.tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.dataset.tab;
            
            // Update buttons
            elements.tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Update content
            elements.tabContents.forEach(c => c.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            
            // Refresh data if needed (debounced)
            if (tabId === 'checkpoints') {
                debounce(loadCheckpoints, 100)();
            } else if (tabId === 'results') {
                debounce(loadResults, 100)();
            }
        });
    });
}

// LLM Tabs
function setupLLMTabs() {
    elements.llmTabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const llmType = btn.dataset.llm;
            
            elements.llmTabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            elements.llmContents.forEach(c => c.classList.remove('active'));
            document.getElementById(`${llmType}-llm`).classList.add('active');
        });
    });
}

// Form Handlers
function setupFormHandlers() {
    // Parallel execution toggle
    elements.parallelEnabled.addEventListener('change', (e) => {
        if (e.target.checked) {
            elements.parallelOptions.classList.add('visible');
        } else {
            elements.parallelOptions.classList.remove('visible');
        }
    });
    
    // Robustness toggle
    elements.robustnessEnabled.addEventListener('change', (e) => {
        if (e.target.checked) {
            elements.robustnessOptions.classList.add('visible');
        } else {
            elements.robustnessOptions.classList.remove('visible');
        }
    });
}

// Button Handlers
function setupButtonHandlers() {
    // Start optimization
    elements.startBtn.addEventListener('click', startOptimization);
    
    // Stop optimization
    elements.stopBtn.addEventListener('click', stopOptimization);
    
    // Refresh status
    elements.refreshBtn.addEventListener('click', refreshStatus);
    
    // Clear logs
    elements.clearLogsBtn.addEventListener('click', () => {
        elements.logsBox.innerHTML = '<p class="placeholder">Logs cleared...</p>';
        scoreHistoryBuffer = []; // Clear memory buffer too
    });
    
    // Save config
    elements.saveConfigBtn.addEventListener('click', saveConfig);
    
    // Load config
    elements.loadConfigBtn.addEventListener('click', () => {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.yaml,.yml,.json';
        input.onchange = (e) => {
            const file = e.target.files[0];
            if (file) {
                loadConfigFromFile(file);
            }
        };
        input.click();
    });
    
    // Refresh checkpoints
    elements.refreshCheckpointsBtn.addEventListener('click', loadCheckpoints);
    
    // Export results
    elements.exportJsonBtn.addEventListener('click', () => exportResults('json'));
    elements.exportTxtBtn.addEventListener('click', () => exportResults('txt'));
    
    // Export/Import full state
    if (elements.exportStateBtn) {
        elements.exportStateBtn.addEventListener('click', exportFullState);
    }
    if (elements.importStateBtn) {
        elements.importStateBtn.addEventListener('click', importFullState);
    }
    
    // Show diff
    if (elements.showDiffBtn) {
        elements.showDiffBtn.addEventListener('click', showPromptDiff);
    }
    if (elements.closeDiffBtn) {
        elements.closeDiffBtn.addEventListener('click', closePromptDiff);
    }
    
    // Close modal on outside click
    if (elements.diffModal) {
        elements.diffModal.addEventListener('click', (e) => {
            if (e.target === elements.diffModal) {
                closePromptDiff();
            }
        });
    }
}

// Chart.js initialization
function initChart() {
    const ctx = document.getElementById('scoreChart').getContext('2d');
    scoreChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Best Score',
                data: [],
                backgroundColor: 'rgba(79, 70, 229, 0.75)',
                borderColor: '#4f46e5',
                borderWidth: 1,
                borderRadius: 4,
                borderSkipped: false,
                maxBarThickness: 28
            }, {
                label: 'Current Score',
                data: [],
                backgroundColor: 'rgba(16, 185, 129, 0.75)',
                borderColor: '#10b981',
                borderWidth: 1,
                borderRadius: 4,
                borderSkipped: false,
                maxBarThickness: 28
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1
                },
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        autoSkip: false
                    }
                }
            },
            animation: {
                duration: 200
            }
        }
    });
}

// Clear all chart data
function clearChart() {
    if (!scoreChart) return;
    scoreChart.data.labels = [];
    scoreChart.data.datasets[0].data = [];
    scoreChart.data.datasets[1].data = [];
    scoreChart.update('none');
    scoreHistoryBuffer = [];
    lastChartUpdate = 0;
}

// Throttled chart update
function updateChartThrottled(labels, bestScores, currentScores) {
    const now = Date.now();
    
    // Store data for later update
    scoreHistoryBuffer = {
        labels: labels.slice(-MAX_CHART_POINTS),
        bestScores: bestScores.slice(-MAX_CHART_POINTS),
        currentScores: currentScores.slice(-MAX_CHART_POINTS)
    };
    
    // Check if we should update now
    if (now - lastChartUpdate >= CHART_UPDATE_INTERVAL) {
        updateChartImmediate();
    } else if (!pendingChartUpdate) {
        // Schedule update for later
        pendingChartUpdate = true;
        setTimeout(() => {
            updateChartImmediate();
            pendingChartUpdate = false;
        }, CHART_UPDATE_INTERVAL - (now - lastChartUpdate));
    }
}

// Immediate chart update (internal)
function updateChartImmediate() {
    if (!scoreChart || !scoreHistoryBuffer.labels) return;
    
    scoreChart.data.labels = scoreHistoryBuffer.labels;
    scoreChart.data.datasets[0].data = scoreHistoryBuffer.bestScores;
    scoreChart.data.datasets[1].data = scoreHistoryBuffer.currentScores;
    scoreChart.update('none'); // Update without animation
    
    lastChartUpdate = Date.now();
}

// Load configuration
async function loadConfig() {
    try {
        const response = await fetch('/api/config');
        const data = await response.json();
        
        if (data.status === 'success' && data.config) {
            populateForm(data.config);
        }
    } catch (error) {
        showToast('Failed to load configuration', 'error');
    }
}

// Populate form with config data
function populateForm(config) {
    // Helper to set nested values
    const setValue = (name, value) => {
        const input = document.querySelector(`[name="${name}"]`);
        if (input) {
            if (input.type === 'checkbox') {
                input.checked = value;
                // Trigger change event for visibility
                input.dispatchEvent(new Event('change'));
            } else {
                input.value = value;
            }
        }
    };
    
    // Set values
    if (config.task) {
        setValue('task.name', config.task.name);
        setValue('task.description', config.task.description);
        setValue('task.initial_prompt', config.task.initial_prompt);
    }
    
    if (config.optimizer_llm) {
        setValue('optimizer_llm.model', config.optimizer_llm.model);
        setValue('optimizer_llm.backend', config.optimizer_llm.backend);
        setValue('optimizer_llm.temperature', config.optimizer_llm.temperature);
        setValue('optimizer_llm.max_tokens', config.optimizer_llm.max_tokens);
    }
    
    if (config.target_llm) {
        setValue('target_llm.model', config.target_llm.model);
        setValue('target_llm.backend', config.target_llm.backend);
        setValue('target_llm.temperature', config.target_llm.temperature);
        setValue('target_llm.max_tokens', config.target_llm.max_tokens);
    }
    
    if (config.experiment) {
        setValue('experiment.max_iterations', config.experiment.max_iterations);
        setValue('experiment.batch_size', config.experiment.batch_size);
        setValue('experiment.parallel_enabled', config.experiment.parallel_enabled);
        setValue('experiment.parallel_workers', config.experiment.parallel_workers);
        setValue('experiment.parallel_candidates', config.experiment.parallel_candidates);
    }
    
    if (config.metric) {
        setValue('metric.type', config.metric.type);
        setValue('metric.target_score', config.metric.target_score);
    }
    
    if (config.storage) {
        setValue('storage.results_dir', config.storage.results_dir);
        setValue('storage.checkpoint_interval', config.storage.checkpoint_interval);
    }
    
    if (config.robustness) {
        setValue('robustness.enabled', config.robustness.enabled);
        setValue('robustness.num_variants', config.robustness.num_variants);
        setValue('robustness.score_threshold', config.robustness.score_threshold);
    }
}

// Get form data as config object
function getFormData() {
    const formData = new FormData(elements.configForm);
    const config = {
        optimizer_llm: {},
        target_llm: {},
        experiment: {},
        task: {},
        metric: {},
        context: {
            max_experiments_in_context: 20,
            compression_threshold: 50
        },
        storage: {},
        robustness: {}
    };
    
    for (const [key, value] of formData.entries()) {
        const parts = key.split('.');
        let current = config;
        
        for (let i = 0; i < parts.length - 1; i++) {
            current = current[parts[i]];
        }
        
        const lastKey = parts[parts.length - 1];
        
        // Type conversion
        if (value === 'on') {
            current[lastKey] = true;
        } else if (!isNaN(value) && value !== '') {
            current[lastKey] = Number(value);
        } else {
            current[lastKey] = value;
        }
    }
    
    // Handle unchecked checkboxes
    if (!config.experiment.parallel_enabled) config.experiment.parallel_enabled = false;
    if (!config.robustness.enabled) config.robustness.enabled = false;
    
    return config;
}

// Save configuration
async function saveConfig() {
    const config = getFormData();
    
    try {
        const response = await fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            showToast(data.message, 'success');
        } else {
            showToast(data.message || 'Failed to save configuration', 'error');
        }
    } catch (error) {
        showToast('Failed to save configuration', 'error');
    }
}

// Load config from file
function loadConfigFromFile(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        try {
            const content = e.target.result;
            // Simple YAML/JSON parsing
            let config;
            if (file.name.endsWith('.json')) {
                config = JSON.parse(content);
            } else {
                // Basic YAML parsing (simplified)
                config = parseYAML(content);
            }
            populateForm(config);
            showToast('Configuration loaded', 'success');
        } catch (error) {
            showToast('Failed to parse configuration file', 'error');
        }
    };
    reader.readAsText(file);
}

// Simple YAML parser
function parseYAML(content) {
    const lines = content.split('\n');
    const result = {};
    let current = result;
    const stack = [];
    
    for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith('#')) continue;
        
        const indent = line.search(/\S/);
        const [key, ...valueParts] = trimmed.split(':');
        const value = valueParts.join(':').trim();
        
        if (!value && trimmed.endsWith(':')) {
            // New section
            result[key] = {};
            current = result[key];
        } else if (value) {
            // Key-value pair
            // Remove quotes if present
            const cleanValue = value.replace(/^["']|["']$/g, '');
            
            // Try to parse as number
            if (!isNaN(cleanValue) && cleanValue !== '') {
                current[key] = Number(cleanValue);
            } else if (cleanValue === 'true' || cleanValue === 'false') {
                current[key] = cleanValue === 'true';
            } else {
                current[key] = cleanValue;
            }
        }
    }
    
    return result;
}

// Start optimization
async function startOptimization() {
    // Prevent double-clicks
    if (elements.startBtn.disabled) return;
    
    elements.startBtn.disabled = true;
    elements.startBtn.classList.add('loading');
    
    const config = getFormData();
    
    // Save config first
    try {
        await fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
    } catch (error) {
        showToast('Failed to save configuration', 'error');
        elements.startBtn.disabled = false;
        elements.startBtn.classList.remove('loading');
        return;
    }
    
    // Start optimization
    try {
        const response = await fetch('/api/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            isRunning = true;
            elements.stopBtn.disabled = false;
            updateStatusIndicator('running');
            showToast('Optimization started', 'success');

            // Reset all dashboard displays immediately on new run
            if (elements.currentIteration) elements.currentIteration.textContent = '0';
            if (elements.bestScore) elements.bestScore.textContent = '0.000';
            if (elements.elapsedTime) elements.elapsedTime.textContent = '00:00:00';
            if (elements.runStatus) elements.runStatus.textContent = 'Running';
            if (elements.errorBanner) elements.errorBanner.style.display = 'none';
            if (elements.bestPromptBox) elements.bestPromptBox.innerHTML = '<p class="placeholder">Waiting for first iteration...</p>';
            if (elements.currentTestPromptBox) elements.currentTestPromptBox.innerHTML = '<p class="placeholder">Waiting for first iteration...</p>';
            if (elements.bestPromptMeta) elements.bestPromptMeta.textContent = '—';
            if (elements.currentPromptMeta) elements.currentPromptMeta.textContent = '—';
            if (elements.logsBox) elements.logsBox.innerHTML = '<p class="placeholder">Waiting for logs...</p>';
            _seenLogs.clear();
            elapsedBaseSeconds = 0;
            elapsedStartTime = Date.now();
            startElapsedTimer();
            // Hide and clear chart until run completes
            const chartCard = document.getElementById('scoreChartCard');
            if (chartCard) chartCard.style.display = 'none';
            clearChart();

            // Switch to dashboard
            document.querySelector('[data-tab="dashboard"]').click();

            // (Re)start status stream — it may have been closed after previous run completed.
            // startStatusStream also kicks off log streaming when it detects is_running=true.
            startStatusStream();
            startLogStream();
        } else {
            showToast(data.message, 'error');
            elements.startBtn.disabled = false;
        }
    } catch (error) {
        showToast('Failed to start optimization', 'error');
        elements.startBtn.disabled = false;
    } finally {
        elements.startBtn.classList.remove('loading');
    }
}

// Stop optimization
async function stopOptimization() {
    try {
        const response = await fetch('/api/stop', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            showToast(data.message, 'info');
            elements.stopBtn.disabled = true;
        } else {
            showToast(data.message, 'error');
        }
    } catch (error) {
        showToast('Failed to stop optimization', 'error');
    }
}

// Refresh status (debounced)
const refreshStatus = debounce(async function() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        
        if (data.status === 'success') {
            updateDashboard(data.data);
        }
    } catch (error) {
        showToast('Failed to refresh status', 'error');
    }
}, 500);

// Debounce helper
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Update dashboard with status data
function updateDashboard(data) {
    // Update stats
    if (elements.currentIteration) {
        const cur = data.current_iteration || 0;
        const max = data.max_iterations || 0;
        if (data.is_running) {
            elements.currentIteration.textContent = max > 0 ? `${cur} / ${max}` : cur;
        } else if (data.has_final_report && max > 0) {
            elements.currentIteration.textContent = `${cur} / ${max}`;
        } else if (cur > 0 && !data.is_running) {
            // Historical state from a previous run — label it clearly
            elements.currentIteration.textContent = `${cur} (last)`;
        } else {
            elements.currentIteration.textContent = '—';
        }
    }
    if (elements.bestScore) {
        elements.bestScore.textContent = (data.best_score || 0).toFixed(3);
    }

    // Sync client-side elapsed timer from server
    if (data.is_running) {
        elapsedBaseSeconds = data.elapsed_time || 0;
        elapsedStartTime = Date.now();
        startElapsedTimer();
    } else {
        stopElapsedTimer();
        if (elements.elapsedTime) {
            elements.elapsedTime.textContent = formatElapsedTime(data.elapsed_time || 0);
        }
    }

    if (elements.runStatus) {
        if (data.error_message) {
            elements.runStatus.textContent = 'Error';
        } else {
            elements.runStatus.textContent = data.is_running ? 'Running' : (data.has_final_report ? 'Complete' : 'Idle');
        }
    }
    
    // Show/hide error banner
    if (elements.errorBanner) {
        if (data.error_message) {
            elements.errorBanner.textContent = 'Error: ' + data.error_message;
            elements.errorBanner.style.display = 'block';
        } else {
            elements.errorBanner.style.display = 'none';
        }
    }

    // Update best prompt with meta label
    if (elements.bestPromptBox && data.best_prompt) {
        elements.bestPromptBox.innerHTML = `<pre>${escapeHtml(data.best_prompt)}</pre>`;
    }
    if (elements.bestPromptMeta) {
        const bestIter = data.best_prompt_iteration || 0;
        const bestScore = typeof data.best_score === 'number' ? data.best_score.toFixed(3) : '—';
        elements.bestPromptMeta.textContent = bestIter > 0
            ? `score ${bestScore} · iteration ${bestIter}`
            : (data.best_prompt ? `score ${bestScore} · baseline` : '—');
    }

    // Update currently-testing prompt
    if (elements.currentTestPromptBox) {
        const cur = data.current_test_prompt;
        if (cur) {
            elements.currentTestPromptBox.innerHTML = `<pre>${escapeHtml(cur)}</pre>`;
        }
    }
    if (elements.currentPromptMeta) {
        const cur = data.current_iteration || 0;
        const max = data.max_iterations || 0;
        elements.currentPromptMeta.textContent = data.is_running
            ? `iteration ${cur}${max > 0 ? ' / ' + max : ''}`
            : (data.has_final_report ? 'run complete' : '—');
    }
    
    // Store previous best prompt for diff
    if (data.previous_best_prompt) {
        elements.bestPromptBox.dataset.previousPrompt = data.previous_best_prompt;
    }
    
    // Update status indicator
    if (data.is_running) {
        updateStatusIndicator('running');
        isRunning = true;
        elements.startBtn.disabled = true;
        elements.stopBtn.disabled = false;
    } else if (data.has_final_report) {
        updateStatusIndicator('complete');
        isRunning = false;
        elements.startBtn.disabled = false;
        elements.stopBtn.disabled = true;
        // Keep the status stream alive — closing it here causes a race condition
        // where the next Start click sees stale "complete" data and kills the
        // new stream before any real updates arrive.
    } else {
        updateStatusIndicator('idle');
        isRunning = false;
        elements.startBtn.disabled = false;
        elements.stopBtn.disabled = true;
    }
    
    // Show chart only on completion with full history
    const chartCard = document.getElementById('scoreChartCard');
    if (data.has_final_report && data.score_history && data.score_history.length > 0) {
        // Pad to max_iterations — skipped/duplicate iterations leave gaps in score_history
        const maxIter = data.max_iterations || data.score_history.length;
        const history = [...data.score_history];
        while (history.length < maxIter) {
            const last = history[history.length - 1];
            history.push({ best_score: last.best_score, current_score: last.current_score });
        }
        if (history.length > 0) {
            const labels = history.map((_, i) => `Iter ${i + 1}`);
            const bestScores = history.map(h => h.best_score || 0);
            const currentScores = history.map(h => h.current_score || 0);
            updateChartThrottled(labels, bestScores, currentScores);
            if (chartCard) chartCard.style.display = '';
        }
    } else {
        if (chartCard) chartCard.style.display = 'none';
        clearChart();
    }
}

// Format elapsed time
function formatElapsedTime(seconds) {
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

// Start client-side elapsed timer (ticks every second while running)
function startElapsedTimer() {
    if (elapsedTimerInterval) return; // already running
    elapsedTimerInterval = setInterval(() => {
        if (!elapsedStartTime) return;
        const localElapsed = (Date.now() - elapsedStartTime) / 1000;
        const total = elapsedBaseSeconds + localElapsed;
        if (elements.elapsedTime) {
            elements.elapsedTime.textContent = formatElapsedTime(total);
        }
    }, 1000);
}

function stopElapsedTimer() {
    if (elapsedTimerInterval) {
        clearInterval(elapsedTimerInterval);
        elapsedTimerInterval = null;
    }
    elapsedStartTime = null;
}

// Escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Update status indicator
function updateStatusIndicator(status) {
    if (!elements.statusDot || !elements.statusText) return;
    
    elements.statusDot.className = 'status-dot';
    
    switch (status) {
        case 'running':
            elements.statusDot.classList.add('running');
            elements.statusText.textContent = 'Running';
            break;
        case 'complete':
            elements.statusDot.classList.add('complete');
            elements.statusText.textContent = 'Complete';
            break;
        case 'error':
            elements.statusDot.classList.add('error');
            elements.statusText.textContent = 'Error';
            break;
        default:
            elements.statusText.textContent = 'Idle';
    }
}

// Start status stream — SSE with aggressive polling fallback
function startStatusStream() {
    cleanupEventSources();
    sseReceivedData = false;

    // Watchdog: if no real data arrives via SSE within 5s, switch to polling
    sseWatchdogTimer = setTimeout(() => {
        if (!sseReceivedData) {
            console.warn('SSE watchdog: no data received, switching to HTTP polling');
            if (statusEventSource) { statusEventSource.close(); statusEventSource = null; }
            startStatusPolling();
            startLogPolling();
        }
    }, SSE_WATCHDOG_MS);

    try {
        statusEventSource = new EventSource('/api/status/stream');

        statusEventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.type === 'heartbeat') return;
                sseReceivedData = true;
                if (sseWatchdogTimer) { clearTimeout(sseWatchdogTimer); sseWatchdogTimer = null; }
                updateDashboard(data);
                // Render any logs bundled in the status update
                if (data.recent_logs && data.recent_logs.length) renderRecentLogs(data.recent_logs);
                if (data.is_running && !eventSource) startLogStream();
            } catch (error) {
                console.error('Failed to parse status update:', error);
            }
        };

        statusEventSource.onerror = () => {
            if (statusEventSource) { statusEventSource.close(); statusEventSource = null; }
            if (sseWatchdogTimer) { clearTimeout(sseWatchdogTimer); sseWatchdogTimer = null; }
            startStatusPolling();
            startLogPolling();
        };

        statusEventSource.onopen = () => {
            if (statusPollInterval) { clearInterval(statusPollInterval); statusPollInterval = null; }
        };

    } catch (error) {
        if (sseWatchdogTimer) { clearTimeout(sseWatchdogTimer); sseWatchdogTimer = null; }
        startStatusPolling();
        startLogPolling();
    }
}

// Start status polling (fallback when SSE fails)
function startStatusPolling() {
    if (statusPollInterval) return;
    console.log('Polling /api/status every', STATUS_UPDATE_INTERVAL, 'ms');

    const poll = async () => {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();
            if (data.status === 'success') {
                updateDashboard(data.data);
                if (data.data.recent_logs && data.data.recent_logs.length) {
                    renderRecentLogs(data.data.recent_logs);
                }
            }
        } catch (error) {
            console.error('Status poll error:', error);
        }
    };

    poll(); // immediate first call
    statusPollInterval = setInterval(poll, STATUS_UPDATE_INTERVAL);
}

// Log polling fallback (when SSE log stream unavailable)
function startLogPolling() {
    if (logPollInterval) return;
    console.log('Polling /api/logs');

    const poll = async () => {
        try {
            const response = await fetch('/api/logs');
            const data = await response.json();
            if (data.status === 'success' && data.logs) {
                renderRecentLogs(data.logs);
            }
        } catch (e) { /* ignore */ }
    };

    poll();
    logPollInterval = setInterval(poll, 2000);
}

// Start log stream (SSE) — falls back to polling if not delivering
function startLogStream() {
    if (eventSource) { eventSource.close(); eventSource = null; }

    let sseLogGotData = false;
    const logWatchdog = setTimeout(() => {
        if (!sseLogGotData) {
            console.warn('Log SSE stalled, switching to log polling');
            if (eventSource) { eventSource.close(); eventSource = null; }
            startLogPolling();
        }
    }, SSE_WATCHDOG_MS);

    eventSource = new EventSource('/api/logs/stream');

    eventSource.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            // Heartbeat confirms SSE is alive — reset watchdog even for heartbeats
            sseLogGotData = true;
            clearTimeout(logWatchdog);
            if (data.type === 'heartbeat') return;
            // Score events are used for chart data at completion — ignore during run
            if (data.type === 'score') return;
            appendLogEntry(data);
        } catch (e) {}
    };

    eventSource.onerror = () => {
        clearTimeout(logWatchdog);
        if (eventSource) { eventSource.close(); eventSource = null; }
        startLogPolling();
    };
}

// Render a batch of log entries (dedup by timestamp+message)
const _seenLogs = new Set();
function renderRecentLogs(logs) {
    if (!elements.logsBox || !logs || !logs.length) return;
    const placeholder = elements.logsBox.querySelector('.placeholder');
    if (placeholder) placeholder.remove();

    let added = 0;
    for (const entry of logs) {
        const key = (entry.timestamp || '') + '|' + (entry.message || '');
        if (_seenLogs.has(key)) continue;
        _seenLogs.add(key);
        appendLogEntry(entry);
        added++;
    }

    // Trim set size
    if (_seenLogs.size > 500) {
        const arr = [..._seenLogs];
        arr.splice(0, 200).forEach(k => _seenLogs.delete(k));
    }
}

// Strip ANSI escape codes from a string
function stripAnsi(str) {
    // eslint-disable-next-line no-control-regex
    return str.replace(/\x1b\[[0-9;]*[mGKHF]/g, '');
}

// Append log entry (throttled and limited)
function appendLogEntry(entry) {
    if (!elements.logsBox) return;

    // Defensive guard: skip entries missing required fields
    if (!entry || !entry.level || !entry.message) return;

    // Remove placeholder if present
    const placeholder = elements.logsBox.querySelector('.placeholder');
    if (placeholder) {
        placeholder.remove();
    }

    // Create log entry element
    const logEntry = document.createElement('div');
    logEntry.className = `log-entry log-${entry.level.toLowerCase()}`;

    const timestamp = document.createElement('span');
    timestamp.className = 'log-timestamp';
    timestamp.textContent = entry.timestamp ? new Date(entry.timestamp).toLocaleTimeString() : '';

    const level = document.createElement('span');
    level.className = `log-level log-level-${entry.level.toLowerCase()}`;
    level.textContent = entry.level;
    
    const message = document.createElement('span');
    message.className = 'log-message';
    message.textContent = stripAnsi(entry.message);
    
    logEntry.appendChild(timestamp);
    logEntry.appendChild(level);
    logEntry.appendChild(message);
    
    elements.logsBox.appendChild(logEntry);
    
    // Limit to MAX_LOG_ENTRIES (remove oldest)
    const entries = elements.logsBox.querySelectorAll('.log-entry');
    if (entries.length > MAX_LOG_ENTRIES) {
        for (let i = 0; i < entries.length - MAX_LOG_ENTRIES; i++) {
            entries[i].remove();
        }
    }
    
    // Auto-scroll to bottom
    elements.logsBox.scrollTop = elements.logsBox.scrollHeight;
}

// Load checkpoints (debounced)
const loadCheckpoints = debounce(async function() {
    try {
        const response = await fetch('/api/checkpoints');
        const data = await response.json();
        
        if (data.status === 'success') {
            renderCheckpoints(data.checkpoints);
        } else {
            elements.checkpointsList.innerHTML = '<p class="placeholder">Failed to load checkpoints</p>';
        }
    } catch (error) {
        elements.checkpointsList.innerHTML = '<p class="placeholder">Error loading checkpoints</p>';
    }
}, 300);

// Render checkpoints list
function renderCheckpoints(checkpoints) {
    if (!checkpoints || checkpoints.length === 0) {
        elements.checkpointsList.innerHTML = '<div class="empty-state"><p>No checkpoints found</p><span>Run an optimization to create checkpoints</span></div>';
        return;
    }

    elements.checkpointsList.innerHTML = '';

    checkpoints.forEach(cp => {
        const item = document.createElement('div');
        item.className = 'checkpoint-item';

        const date = cp.timestamp ? new Date(cp.timestamp * 1000).toLocaleString() : 'Unknown';
        const score = (typeof cp.best_score === 'number') ? cp.best_score.toFixed(3) : '—';
        const safePath = escapeHtml(cp.path || '');

        item.innerHTML = `
            <div class="checkpoint-info">
                <div class="checkpoint-name">${escapeHtml(cp.filename || 'checkpoint')}</div>
                <div class="checkpoint-meta">Iteration ${cp.iteration || '?'} &bull; Score: ${score} &bull; ${date}</div>
            </div>
            <button class="btn btn-small" onclick="loadCheckpoint('${safePath}')">Load</button>
        `;

        elements.checkpointsList.appendChild(item);
    });
}

// Load checkpoint
async function loadCheckpoint(path) {
    try {
        const response = await fetch('/api/checkpoints/load', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ checkpoint_path: path })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            showToast(data.message, 'success');
            // Refresh status
            refreshStatus();
        } else {
            showToast(data.message || 'Failed to load checkpoint', 'error');
        }
    } catch (error) {
        showToast('Failed to load checkpoint', 'error');
    }
}

// Load results (debounced)
const loadResults = debounce(async function() {
    try {
        const response = await fetch('/api/results');
        const data = await response.json();
        
        if (data.status === 'success') {
            renderResults(data.report);
        } else {
            elements.resultsContent.innerHTML = '<p class="placeholder">No results available yet. Complete an optimization to see results.</p>';
            elements.resultsActions.style.display = 'none';
        }
    } catch (error) {
        elements.resultsContent.innerHTML = '<p class="placeholder">Error loading results</p>';
        elements.resultsActions.style.display = 'none';
    }
}, 300);

// Render results
function renderResults(report) {
    if (!report) {
        elements.resultsContent.innerHTML = '<p class="placeholder">No results available</p>';
        elements.resultsActions.style.display = 'none';
        return;
    }
    
    const html = `
        <div class="results-summary">
            <div class="result-item">
                <span class="result-label">Task:</span>
                <span class="result-value">${escapeHtml(report.task || 'N/A')}</span>
            </div>
            <div class="result-item">
                <span class="result-label">Total Iterations:</span>
                <span class="result-value">${report.total_iterations || 0}</span>
            </div>
            <div class="result-item">
                <span class="result-label">Initial Score:</span>
                <span class="result-value">${(report.initial_score || 0).toFixed(3)}</span>
            </div>
            <div class="result-item">
                <span class="result-label">Final Score:</span>
                <span class="result-value">${(report.final_score || 0).toFixed(3)}</span>
            </div>
            <div class="result-item">
                <span class="result-label">Improvement:</span>
                <span class="result-value">+${(report.improvement || 0).toFixed(3)}</span>
            </div>
        </div>
        <div class="results-prompt">
            <h4>Best Prompt</h4>
            <pre>${escapeHtml(report.best_prompt || 'N/A')}</pre>
        </div>
    `;
    
    elements.resultsContent.innerHTML = html;
    elements.resultsActions.style.display = 'flex';
}

// Export results
async function exportResults(format) {
    try {
        const response = await fetch('/api/export', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ format })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            showToast(data.message, 'success');
        } else {
            showToast(data.message || 'Failed to export results', 'error');
        }
    } catch (error) {
        showToast('Failed to export results', 'error');
    }
}

// Export full optimization state
async function exportFullState() {
    try {
        const response = await fetch('/api/export/full');
        const data = await response.json();
        
        if (data.status === 'success') {
            // Create download
            const blob = new Blob([JSON.stringify(data.data, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `autoprompter_state_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
            showToast('Optimization state exported', 'success');
        } else {
            showToast(data.message || 'Failed to export state', 'error');
        }
    } catch (error) {
        showToast('Failed to export state', 'error');
    }
}

// Import full optimization state
function importFullState() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        
        const reader = new FileReader();
        reader.onload = async (event) => {
            try {
                const importData = JSON.parse(event.target.result);
                
                const response = await fetch('/api/import', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ data: importData })
                });
                
                const data = await response.json();
                
                if (data.status === 'success') {
                    showToast(data.message, 'success');
                    // Refresh dashboard
                    refreshStatus();
                    // Switch to dashboard tab
                    document.querySelector('[data-tab="dashboard"]').click();
                } else {
                    showToast(data.message || 'Failed to import state', 'error');
                }
            } catch (error) {
                showToast('Failed to parse import file', 'error');
            }
        };
        reader.readAsText(file);
    };
    input.click();
}

// Show prompt diff visualization
function showPromptDiff() {
    const currentPrompt = elements.bestPromptBox.querySelector('pre')?.textContent || '';
    const previousPrompt = elements.bestPromptBox.dataset.previousPrompt || '';
    
    if (!currentPrompt || !previousPrompt) {
        showToast('No previous prompt available for comparison', 'info');
        return;
    }
    
    const diffHtml = computeDiff(previousPrompt, currentPrompt);
    elements.diffContent.innerHTML = diffHtml;
    elements.diffModal.classList.add('active');
}

// Close prompt diff modal
function closePromptDiff() {
    elements.diffModal.classList.remove('active');
}

// Compute diff between two prompts
function computeDiff(oldText, newText) {
    // Simple word-level diff
    const oldWords = oldText.split(/(\s+)/);
    const newWords = newText.split(/(\s+)/);
    
    // Use a simple LCS (Longest Common Subsequence) approach
    const result = [];
    let i = 0, j = 0;
    
    // Find common prefix
    while (i < oldWords.length && j < newWords.length && oldWords[i] === newWords[j]) {
        result.push(`<span class="diff-unchanged">${escapeHtml(oldWords[i])}</span>`);
        i++;
        j++;
    }
    
    // Find removed words
    while (i < oldWords.length) {
        // Check if this word appears later in newWords
        const indexInNew = newWords.indexOf(oldWords[i], j);
        if (indexInNew === -1) {
            result.push(`<span class="diff-removed">${escapeHtml(oldWords[i])}</span>`);
            i++;
        } else {
            // Add words that were inserted before this match
            while (j < indexInNew) {
                result.push(`<span class="diff-added">${escapeHtml(newWords[j])}</span>`);
                j++;
            }
            // Add the matching word
            result.push(`<span class="diff-unchanged">${escapeHtml(oldWords[i])}</span>`);
            i++;
            j++;
        }
    }
    
    // Add remaining new words
    while (j < newWords.length) {
        result.push(`<span class="diff-added">${escapeHtml(newWords[j])}</span>`);
        j++;
    }
    
    return `
        <div class="diff-container">
            <div class="diff-header">
                <span class="diff-legend">
                    <span class="diff-legend-item"><span class="diff-legend-color diff-legend-added"></span> Added</span>
                    <span class="diff-legend-item"><span class="diff-legend-color diff-legend-removed"></span> Removed</span>
                </span>
            </div>
            <div class="diff-content">${result.join('')}</div>
        </div>
    `;
}

// Show toast notification
function showToast(message, type = 'info') {
    if (!elements.toastContainer) return;
    
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    
    elements.toastContainer.appendChild(toast);
    
    // Remove after 3 seconds
    setTimeout(() => {
        toast.classList.add('fade-out');
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// Start the app when DOM is ready
document.addEventListener('DOMContentLoaded', init);
