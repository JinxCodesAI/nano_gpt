#!/usr/bin/env python3
"""
Comprehensive experiment runner for diffusion training.

This script runs multiple experiments sequentially, handling errors gracefully
and organizing results in separate directories. It supports easy override of
global settings like batch size, output directory, and wandb configuration.

Usage:
    python run_experiments.py --configs config1.py config2.py --batch_size 8 --base_out_dir experiments
"""

import os
import sys
import time
import shutil
import subprocess
import traceback
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
import json


class ExperimentRunner:
    def __init__(self, base_out_dir: str = "experiments", 
                 batch_size: Optional[int] = None,
                 wandb_project: Optional[str] = None,
                 wandb_enabled: Optional[bool] = None,
                 device: Optional[str] = None,
                 compile: Optional[bool] = None,
                 max_iters: Optional[int] = None,
                 learning_rate: Optional[float] = None):
        self.base_out_dir = Path(base_out_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Global overrides
        self.global_overrides = {}
        if batch_size is not None:
            self.global_overrides['batch_size'] = batch_size
        if wandb_project is not None:
            self.global_overrides['wandb_project'] = wandb_project
        if wandb_enabled is not None:
            self.global_overrides['wandb_log'] = wandb_enabled
        if device is not None:
            self.global_overrides['device'] = device
        if compile is not None:
            self.global_overrides['compile'] = compile
        if max_iters is not None:
            self.global_overrides['max_iters'] = max_iters
        if learning_rate is not None:
            self.global_overrides['learning_rate'] = learning_rate
            
        # Create base experiment directory
        self.experiment_dir = self.base_out_dir / f"experiment_{self.timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup master logging
        self.setup_master_logging()
        
        # Results tracking
        self.results = {}
        
        self.logger.info(f"Experiment runner initialized")
        self.logger.info(f"Base directory: {self.experiment_dir}")
        self.logger.info(f"Global overrides: {self.global_overrides}")
        
    def setup_master_logging(self):
        """Setup master log file for the entire experiment batch"""
        log_file = self.experiment_dir / "master_log.txt"
        
        # Create logger
        self.logger = logging.getLogger('ExperimentRunner')
        self.logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
    def get_config_name(self, config_path: str) -> str:
        """Extract clean config name from path"""
        path = Path(config_path)
        # Remove .py extension and any path components
        return path.stem
        
    def create_experiment_dir(self, config_name: str) -> Path:
        """Create directory for specific experiment"""
        exp_dir = self.experiment_dir / config_name
        exp_dir.mkdir(exist_ok=True)
        return exp_dir
        
    def build_command(self, config_path: str, experiment_dir: Path) -> List[str]:
        """Build training command with overrides"""
        cmd = [
            sys.executable, "train_run.py",
            config_path,  # Config file as positional argument (no --config=)
            f"--out_dir={experiment_dir}"
        ]
        
        # Add global overrides
        for key, value in self.global_overrides.items():
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}=True")
                else:
                    cmd.append(f"--{key}=False")
            else:
                cmd.append(f"--{key}={value}")
                
        return cmd
        
    def run_single_experiment(self, config_path: str) -> Dict[str, Any]:
        """Run a single experiment with comprehensive logging and error handling"""
        config_name = self.get_config_name(config_path)
        experiment_dir = self.create_experiment_dir(config_name)
        
        self.logger.info(f"=" * 80)
        self.logger.info(f"STARTING EXPERIMENT: {config_name}")
        self.logger.info(f"Config: {config_path}")
        self.logger.info(f"Output directory: {experiment_dir}")
        self.logger.info(f"=" * 80)
        
        # Prepare result dictionary
        result = {
            'config_name': config_name,
            'config_path': str(config_path),
            'experiment_dir': str(experiment_dir),
            'start_time': datetime.now().isoformat(),
            'status': 'running',
            'error': None,
            'duration_seconds': None,
            'final_checkpoint': None,
            'log_files': {
                'stdout': str(experiment_dir / 'stdout.log'),
                'stderr': str(experiment_dir / 'stderr.log'),
                'combined': str(experiment_dir / 'combined.log')
            }
        }
        
        start_time = time.time()
        
        try:
            # Build command
            cmd = self.build_command(config_path, experiment_dir)
            self.logger.info(f"Command: {' '.join(cmd)}")
            
            # Create log files
            stdout_log = experiment_dir / 'stdout.log'
            stderr_log = experiment_dir / 'stderr.log' 
            combined_log = experiment_dir / 'combined.log'
            
            # Save experiment metadata
            metadata = {
                'config_path': str(config_path),
                'command': cmd,
                'global_overrides': self.global_overrides,
                'start_time': result['start_time'],
                'python_executable': sys.executable,
                'working_directory': os.getcwd()
            }
            
            with open(experiment_dir / 'experiment_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Copy config file to experiment directory
            if os.path.exists(config_path):
                shutil.copy2(config_path, experiment_dir / 'config_used.py')
            
            # Run the training process
            with open(stdout_log, 'w') as stdout_f, \
                 open(stderr_log, 'w') as stderr_f, \
                 open(combined_log, 'w') as combined_f:
                
                # Set environment to force unbuffered output from child process
                env = os.environ.copy()
                env['PYTHONUNBUFFERED'] = '1'  # Force unbuffered Python output
                
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    bufsize=1,  # Line buffered
                    env=env
                )
                
                self.logger.info(f"Process started with PID: {process.pid}")
                
                # Real-time log streaming with proper flushing
                import select
                import sys
                
                # Use select for non-blocking I/O on Unix-like systems
                use_select = hasattr(select, 'select') and sys.platform != 'win32'
                
                while True:
                    # Check if process has terminated
                    if process.poll() is not None:
                        break
                    
                    lines_read = False
                    
                    # Read stdout (non-blocking on Unix, blocking with timeout on Windows)
                    if use_select:
                        # Unix: use select for non-blocking I/O
                        ready, _, _ = select.select([process.stdout, process.stderr], [], [], 0.1)
                        
                        if process.stdout in ready:
                            stdout_line = process.stdout.readline()
                            if stdout_line:
                                lines_read = True
                                stdout_line = stdout_line.rstrip()
                                stdout_f.write(stdout_line + '\n')
                                combined_f.write(f"[STDOUT] {stdout_line}\n")
                                stdout_f.flush()
                                combined_f.flush()
                                # Also log to master log (with prefix to avoid spam)
                                if any(keyword in stdout_line.lower() for keyword in 
                                       ['step', 'iter', 'loss', 'validation', 'stage', 'error', 'warning']):
                                    self.logger.info(f"[{config_name}] {stdout_line}")
                        
                        if process.stderr in ready:
                            stderr_line = process.stderr.readline()
                            if stderr_line:
                                lines_read = True
                                stderr_line = stderr_line.rstrip()
                                stderr_f.write(stderr_line + '\n')
                                combined_f.write(f"[STDERR] {stderr_line}\n")
                                stderr_f.flush()
                                combined_f.flush()
                                self.logger.warning(f"[{config_name}] STDERR: {stderr_line}")
                    else:
                        # Windows: use blocking readline (child process should flush stdout)
                        stdout_line = process.stdout.readline()
                        if stdout_line:
                            lines_read = True
                            stdout_line = stdout_line.rstrip()
                            stdout_f.write(stdout_line + '\n')
                            combined_f.write(f"[STDOUT] {stdout_line}\n")
                            stdout_f.flush()
                            combined_f.flush()
                            # Also log to master log (with prefix to avoid spam)
                            if any(keyword in stdout_line.lower() for keyword in 
                                   ['step', 'iter', 'loss', 'validation', 'stage', 'error', 'warning']):
                                self.logger.info(f"[{config_name}] {stdout_line}")
                        
                        stderr_line = process.stderr.readline()
                        if stderr_line:
                            lines_read = True
                            stderr_line = stderr_line.rstrip()
                            stderr_f.write(stderr_line + '\n')
                            combined_f.write(f"[STDERR] {stderr_line}\n")
                            stderr_f.flush()
                            combined_f.flush()
                            self.logger.warning(f"[{config_name}] STDERR: {stderr_line}")
                    
                    # Small sleep to prevent busy waiting when no data is available
                    if not lines_read:
                        time.sleep(0.01)
                
                # Get final output and flush everything
                remaining_stdout, remaining_stderr = process.communicate()
                if remaining_stdout:
                    stdout_f.write(remaining_stdout)
                    combined_f.write(f"[STDOUT] {remaining_stdout}")
                    stdout_f.flush()
                    combined_f.flush()
                if remaining_stderr:
                    stderr_f.write(remaining_stderr)
                    combined_f.write(f"[STDERR] {remaining_stderr}")
                    stderr_f.flush()
                    combined_f.flush()
                
                # Ensure all file handles are flushed before closing
                stdout_f.flush()
                stderr_f.flush()
                combined_f.flush()
                    
                # Get return code
                return_code = process.returncode
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Update result
            result['duration_seconds'] = duration
            result['end_time'] = datetime.now().isoformat()
            result['return_code'] = return_code
            
            if return_code == 0:
                result['status'] = 'completed'
                self.logger.info(f"Experiment {config_name} completed successfully in {duration:.1f}s")
            else:
                result['status'] = 'failed'
                result['error'] = f"Process exited with code {return_code}"
                self.logger.error(f"Experiment {config_name} failed with return code {return_code}")
            
            # Find final checkpoint
            checkpoint_pattern = experiment_dir / "ckpt_*.pt"
            checkpoints = list(experiment_dir.glob("ckpt_*.pt"))
            if checkpoints:
                # Find the latest checkpoint
                latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
                result['final_checkpoint'] = str(latest_checkpoint)
                self.logger.info(f"Final checkpoint: {latest_checkpoint.name}")
            else:
                self.logger.warning(f"No checkpoints found for {config_name}")
                
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            result['status'] = 'error'
            result['error'] = str(e)
            result['duration_seconds'] = duration
            result['end_time'] = datetime.now().isoformat()
            result['traceback'] = traceback.format_exc()
            
            self.logger.error(f"Exception in experiment {config_name}: {e}")
            self.logger.error(f"Traceback:\n{traceback.format_exc()}")
            
            # Save error details
            with open(experiment_dir / 'error_details.txt', 'w') as f:
                f.write(f"Error: {e}\n\n")
                f.write(f"Traceback:\n{traceback.format_exc()}\n")
        
        # Save individual experiment result
        with open(experiment_dir / 'experiment_result.json', 'w') as f:
            json.dump(result, f, indent=2)
            
        self.logger.info(f"Experiment {config_name} finished: {result['status']}")
        return result
    
    def run_experiments(self, config_paths: List[str]) -> Dict[str, Dict[str, Any]]:
        """Run all experiments sequentially"""
        self.logger.info(f"Starting batch of {len(config_paths)} experiments")
        self.logger.info(f"Configs: {config_paths}")
        
        total_start = time.time()
        
        for i, config_path in enumerate(config_paths, 1):
            self.logger.info(f"Running experiment {i}/{len(config_paths)}: {config_path}")
            
            result = self.run_single_experiment(config_path)
            self.results[result['config_name']] = result
            
            # Save intermediate results
            self.save_summary()
            
            self.logger.info(f"Completed experiment {i}/{len(config_paths)}")
        
        total_duration = time.time() - total_start
        
        self.logger.info(f"=" * 80)
        self.logger.info(f"ALL EXPERIMENTS COMPLETED")
        self.logger.info(f"Total time: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
        self.logger.info(f"Results saved to: {self.experiment_dir}")
        self.logger.info(f"=" * 80)
        
        # Print final summary
        self.print_summary()
        
        return self.results
    
    def save_summary(self):
        """Save experiment summary to JSON file"""
        summary_file = self.experiment_dir / 'experiment_summary.json'
        
        summary = {
            'timestamp': self.timestamp,
            'base_out_dir': str(self.base_out_dir),
            'experiment_dir': str(self.experiment_dir),
            'global_overrides': self.global_overrides,
            'total_experiments': len(self.results),
            'completed': len([r for r in self.results.values() if r['status'] == 'completed']),
            'failed': len([r for r in self.results.values() if r['status'] == 'failed']),
            'errors': len([r for r in self.results.values() if r['status'] == 'error']),
            'results': self.results
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
    def print_summary(self):
        """Print summary of all experiments"""
        completed = [r for r in self.results.values() if r['status'] == 'completed']
        failed = [r for r in self.results.values() if r['status'] == 'failed']
        errors = [r for r in self.results.values() if r['status'] == 'error']
        
        print("\n" + "=" * 80)
        print("EXPERIMENT BATCH SUMMARY")
        print("=" * 80)
        print(f"Total experiments: {len(self.results)}")
        print(f"Completed successfully: {len(completed)}")
        print(f"Failed (non-zero exit): {len(failed)}")
        print(f"Errors (exceptions): {len(errors)}")
        print()
        
        if completed:
            print("✅ COMPLETED:")
            for result in completed:
                duration = result.get('duration_seconds', 0)
                print(f"  {result['config_name']:25} ({duration:.1f}s)")
        
        if failed:
            print("\n❌ FAILED:")
            for result in failed:
                duration = result.get('duration_seconds', 0)
                print(f"  {result['config_name']:25} ({duration:.1f}s) - {result.get('error', 'Unknown error')}")
        
        if errors:
            print("\n💥 ERRORS:")
            for result in errors:
                duration = result.get('duration_seconds', 0)
                print(f"  {result['config_name']:25} ({duration:.1f}s) - {result.get('error', 'Unknown error')}")
        
        print(f"\nResults directory: {self.experiment_dir}")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Run multiple diffusion training experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all configs in a directory with custom batch size
  python run_experiments.py --configs config/shkspr_char_diff/*.py --batch_size 8
  
  # Run specific configs with wandb disabled
  python run_experiments.py --configs easy.py moderate.py --wandb_enabled False
  
  # Run with custom output directory and device
  python run_experiments.py --configs config/*.py --base_out_dir my_experiments --device cuda:1
  
  # Run with custom learning rate and max iterations
  python run_experiments.py --configs *.py --learning_rate 5e-4 --max_iters 10000
        """
    )
    
    parser.add_argument('--configs', nargs='+', required=True,
                       help='Config files to run (supports glob patterns)')
    parser.add_argument('--base_out_dir', default='experiments',
                       help='Base output directory (default: experiments)')
    parser.add_argument('--batch_size', type=int,
                       help='Override batch size for all configs')
    parser.add_argument('--wandb_project', type=str,
                       help='Override wandb project name')
    parser.add_argument('--wandb_enabled', type=lambda x: x.lower() in ['true', '1', 'yes'],
                       help='Enable/disable wandb logging (true/false)')
    parser.add_argument('--device', type=str,
                       help='Override device (cuda, cpu, mps, cuda:0, etc.)')
    parser.add_argument('--compile', type=lambda x: x.lower() in ['true', '1', 'yes'],
                       help='Enable/disable torch.compile (true/false)')
    parser.add_argument('--max_iters', type=int,
                       help='Override maximum training iterations')
    parser.add_argument('--learning_rate', type=float,
                       help='Override learning rate')
    
    args = parser.parse_args()
    
    # Expand glob patterns in config paths
    import glob
    config_paths = []
    for pattern in args.configs:
        matches = glob.glob(pattern)
        if matches:
            config_paths.extend(matches)
        else:
            # If no glob matches, treat as literal filename
            config_paths.append(pattern)
    
    if not config_paths:
        print("Error: No config files found matching the patterns")
        sys.exit(1)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_configs = []
    for config in config_paths:
        if config not in seen:
            unique_configs.append(config)
            seen.add(config)
    
    print(f"Found {len(unique_configs)} config files:")
    for config in unique_configs:
        print(f"  {config}")
    print()
    
    # Create experiment runner
    runner = ExperimentRunner(
        base_out_dir=args.base_out_dir,
        batch_size=args.batch_size,
        wandb_project=args.wandb_project,
        wandb_enabled=args.wandb_enabled,
        device=args.device,
        compile=args.compile,
        max_iters=args.max_iters,
        learning_rate=args.learning_rate
    )
    
    # Run experiments
    try:
        results = runner.run_experiments(unique_configs)
        
        # Exit with error code if any experiments failed
        failed_count = len([r for r in results.values() if r['status'] != 'completed'])
        if failed_count > 0:
            print(f"\n{failed_count} experiments failed or had errors")
            sys.exit(1)
        else:
            print(f"\nAll {len(results)} experiments completed successfully!")
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n\nExperiment batch interrupted by user")
        runner.logger.info("Experiment batch interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nFatal error in experiment runner: {e}")
        runner.logger.error(f"Fatal error: {e}")
        runner.logger.error(f"Traceback:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == '__main__':
    main()