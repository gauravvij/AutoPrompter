#!/usr/bin/env python3
"""
Autonomous Prompt Optimization System

A system that autonomously optimizes prompts for LLMs using iterative improvement.
Uses an Optimizer LLM (Gemini 3.1 Flash Lite Preview) to improve prompts for a 
Target LLM (Qwen 3.5 9b) across multiple experimental iterations.

Usage:
    python main.py --config config.yaml
    python main.py --config config.yaml --max-iterations 50
    python main.py --config config.yaml --override experiment.max_iterations=50
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config_manager import load_config, Config
from optimization_system import PromptOptimizationSystem


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Autonomous Prompt Optimization System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --config config.yaml
  %(prog)s --config config.yaml --max-iterations 50
  %(prog)s --config config.yaml --override experiment.batch_size=10
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--max-iterations', '-i',
        type=int,
        help='Override maximum number of iterations'
    )
    
    parser.add_argument(
        '--override', '-o',
        action='append',
        default=[],
        help='Override config values (format: key=value, e.g., experiment.max_iterations=50)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def parse_overrides(override_list):
    """Parse override arguments into dictionary."""
    overrides = {}
    
    for override in override_list:
        if '=' not in override:
            print(f"Warning: Invalid override format '{override}', expected key=value")
            continue
        
        key, value = override.split('=', 1)
        
        # Try to convert value to appropriate type
        try:
            # Try int
            value = int(value)
        except ValueError:
            try:
                # Try float
                value = float(value)
            except ValueError:
                # Keep as string
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
        
        overrides[key] = value
    
    return overrides


def main():
    """Main entry point."""
    args = parse_args()
    
    print("=" * 70)
    print("AUTONOMOUS PROMPT OPTIMIZATION SYSTEM")
    print("=" * 70)
    print()
    
    # Parse overrides
    overrides = parse_overrides(args.override)
    
    # Add max_iterations override if provided
    if args.max_iterations is not None:
        overrides['experiment.max_iterations'] = args.max_iterations
    
    # Load configuration
    try:
        print(f"Loading configuration from: {args.config}")
        config = load_config(args.config, overrides if overrides else None)
        print(f"Configuration loaded successfully")
        print()
    except FileNotFoundError as e:
        print(f"Error: Configuration file not found: {args.config}")
        print(f"Please create a config.yaml file or specify a different path.")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: Configuration validation failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Print configuration summary
    print("Configuration Summary:")
    print(f"  Task: {config.task.name}")
    print(f"  Optimizer LLM: {config.optimizer_llm.model}")
    print(f"  Target LLM: {config.target_llm.model}")
    print(f"  Metric: {config.metric.type} (target: {config.metric.target_score})")
    print(f"  Max Iterations: {config.experiment.max_iterations}")
    print(f"  Batch Size: {config.experiment.batch_size}")
    print()
    
    # Initialize and run optimization system
    try:
        print("Initializing optimization system...")
        system = PromptOptimizationSystem(config)
        print("System initialized successfully")
        print()
        
        print("Starting optimization...")
        print("-" * 70)
        report = system.run()
        print("-" * 70)
        print()
        
        # Print summary
        if report['status'] == 'success':
            print("OPTIMIZATION SUMMARY")
            print("=" * 70)
            print(f"Task: {report['task']}")
            print(f"Total Iterations: {report['total_iterations']}")
            print(f"Initial Score: {report['initial_score']:.3f}")
            print(f"Final Score: {report['final_score']:.3f}")
            print(f"Improvement: {report['improvement']:.3f} ({report['improvement_percent']:.1f}%)")
            print(f"Target Reached: {'Yes' if report['target_reached'] else 'No'}")
            print()
            print("Best Prompt:")
            print("-" * 70)
            print(report['best_prompt'])
            print("-" * 70)
            print()
            print(f"Full report saved to: {config.storage.results_dir}/final_report.json")
            print(f"Experiment ledger: {config.storage.ledger_file}")
            print()
            
            return 0
        else:
            print(f"Optimization failed: {report.get('reason', 'Unknown error')}")
            return 1
            
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user")
        return 130
    except Exception as e:
        print(f"\nError during optimization: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
