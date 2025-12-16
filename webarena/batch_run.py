#!/usr/bin/env python3
"""
Batch runner for WebArena tasks.

Usage:
    # Run tasks by site with parallel execution (RECOMMENDED)
    python batch_run.py --sites shopping_admin --workers 8

    # Run tasks by multiple sites
    python batch_run.py --sites shopping_admin,gitlab --workers 4

    # Run each task multiple times
    python batch_run.py --sites shopping_admin --workers 8 --repeat 3

    # Run tasks by range
    python batch_run.py --start 700 --end 710

    # Run specific tasks
    python batch_run.py --tasks 700,701,702

    # Run tasks by website
    python batch_run.py --website shopping_admin

    # Resume from last failed/incomplete task
    python batch_run.py --sites shopping_admin --workers 8 --resume

    # Specify model and max steps
    python batch_run.py --sites shopping_admin --workers 8 --model google/gemini-2.5-flash-preview-09-2025 --max_steps 30
"""

import os
import json
import argparse
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from collections import defaultdict
from concurrent.futures import as_completed
from threading import Lock
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class BatchRunner:
    def __init__(
        self,
        model: str = "google/gemini-2.5-flash-preview-09-2025",
        max_steps: int = 20,
        headless: bool = True,
        results_dir: str = "results",
        workflow_dir: str = "workflow",
        batch_log_dir: str = None,
        use_llm_eval: bool = True,
        llm_eval_model: str = "google/gemini-2.5-flash-preview-09-2025",
        use_autoeval: bool = False,
        update_workflow: bool = False,
        induce_model: str = "google/gemini-2.5-flash-preview-09-2025",
        num_samples: int = 1,
        repeat: int = 1,
    ):
        self.model = model
        self.max_steps = max_steps
        self.headless = headless
        self.results_dir = Path(results_dir)
        self.workflow_dir = Path(workflow_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.use_llm_eval = use_llm_eval
        self.llm_eval_model = llm_eval_model
        self.use_autoeval = use_autoeval
        self.update_workflow = update_workflow
        self.induce_model = induce_model
        self.num_samples = num_samples
        self.repeat = repeat

        # Create batch log directory with timestamp
        if batch_log_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.batch_log_dir = Path("batch_logs") / f"batch_{timestamp}"
        else:
            self.batch_log_dir = Path(batch_log_dir)

        self.batch_log_dir.mkdir(parents=True, exist_ok=True)
        print(f"Batch logs will be saved to: {self.batch_log_dir}")

        # Initialize incremental result files
        self.incremental_json = self.batch_log_dir / "results_incremental.json"
        self.incremental_summary = self.batch_log_dir / "results_summary.txt"

        # Load all task configs
        self.config_dir = Path("config_files")
        self.all_configs = self._load_all_configs()

        # Initialize LLM evaluator if enabled
        if self.use_llm_eval:
            try:
                from autoeval.string_match_evaluator import StringMatchEvaluator
                # Use OpenRouter for all models
                use_openrouter = True
                self.llm_evaluator = StringMatchEvaluator(
                    model_name=self.llm_eval_model,
                    use_openrouter=use_openrouter
                )
                print(f"LLM evaluator initialized with model: {self.llm_eval_model} (via OpenRouter)")
            except Exception as e:
                print(f"Warning: Failed to initialize LLM evaluator: {e}")
                self.llm_evaluator = None
        else:
            self.llm_evaluator = None

    def _load_all_configs(self) -> dict:
        """Load all task configurations from test.raw.json"""
        test_raw = self.config_dir / "test.raw.json"
        if not test_raw.exists():
            raise FileNotFoundError(f"Cannot find {test_raw}")

        with open(test_raw, 'r') as f:
            configs = json.load(f)

        # Create a mapping from task_id to config
        return {conf["task_id"]: conf for conf in configs}

    def get_tasks_by_range(self, start: int, end: int) -> List[int]:
        """Get task IDs in the range [start, end]"""
        return [tid for tid in self.all_configs.keys() if start <= tid <= end]

    def get_tasks_by_website(self, website: str, start: int = 0, end: int = 999999) -> List[int]:
        """Get task IDs for a specific website"""
        return [
            tid for tid, conf in self.all_configs.items()
            if website in conf["sites"] and start <= tid <= end
        ]

    def get_tasks_by_sites(self, sites: List[str]) -> List[int]:
        """Get task IDs for specific sites"""
        return [
            tid for tid, conf in self.all_configs.items()
            if any(site in conf["sites"] for site in sites)
        ]

    def _calculate_cumulative_auc(self, task_results: List[dict]) -> float:
        """
        Calculate cumulative success rate AUC for a single task.

        Args:
            task_results: List of results for a single task (sorted by repeat number)

        Returns:
            AUC value normalized to [0, 1]
        """
        if not task_results or len(task_results) == 1:
            # If only one run, return 1.0 if success, 0.0 if failure
            if len(task_results) == 1:
                return 1.0 if task_results[0].get("status") == "success" else 0.0
            return 0.0

        n = len(task_results)
        cumulative_success_rates = []

        # Calculate cumulative success rate at each step
        successes = 0
        for i, result in enumerate(task_results, 1):
            if result.get("status") == "success":
                successes += 1
            cumulative_rate = successes / i
            cumulative_success_rates.append(cumulative_rate)

        # Calculate AUC using trapezoidal rule
        auc = 0.0
        for i in range(len(cumulative_success_rates) - 1):
            # Trapezoidal rule: (y1 + y2) / 2 * dx
            # Here dx = 1 (each step)
            auc += (cumulative_success_rates[i] + cumulative_success_rates[i+1]) / 2

        # Normalize to [0, 1] by dividing by (n-1)
        # Maximum possible AUC is n-1 (all success from start)
        if n > 1:
            auc = auc / (n - 1)

        return auc

    def _save_incremental_results(self, all_results: List[dict], metadata: dict):
        """Save incremental results to JSON and summary text files"""
        # Calculate statistics
        total_runs = len(all_results)
        successful_runs = sum(1 for r in all_results if r.get("status") == "success")
        failed_runs = sum(1 for r in all_results if r.get("status") == "failure")
        error_runs = sum(1 for r in all_results if r.get("status") == "error")

        overall_accuracy = (successful_runs / total_runs * 100) if total_runs > 0 else 0

        # Group by task_id for task-level statistics
        tasks_results = {}
        for r in all_results:
            tid = r["task_id"]
            if tid not in tasks_results:
                tasks_results[tid] = []
            tasks_results[tid].append(r)

        # Sort each task's results by repeat number
        for tid in tasks_results:
            tasks_results[tid].sort(key=lambda x: x.get("repeat", 1))

        # Calculate task-level accuracy
        task_accuracies = {}
        task_aucs = {}
        tasks_with_at_least_one_success = 0
        tasks_with_all_success = 0

        for tid, results in tasks_results.items():
            successes = sum(1 for r in results if r.get("status") == "success")
            task_accuracies[tid] = (successes / len(results) * 100) if results else 0

            # Count tasks with at least one success / all success
            if successes > 0:
                tasks_with_at_least_one_success += 1
            if successes == len(results):
                tasks_with_all_success += 1

            # Calculate AUC only if repeat > 1
            if len(results) > 1:
                task_aucs[tid] = self._calculate_cumulative_auc(results)

        # Calculate average AUC
        average_auc = sum(task_aucs.values()) / len(task_aucs) if task_aucs else None

        # Build statistics
        statistics = {
            "completed_runs": total_runs,
            "total_success": successful_runs,
            "total_failure": failed_runs,
            "total_error": error_runs,
            "overall_accuracy": round(overall_accuracy, 2),
            "completed_tasks": len(tasks_results),
        }

        # Add task-level statistics only if repeat > 1
        repeats_per_task = metadata.get("repeats_per_task", 1)
        if repeats_per_task > 1:
            total_tasks = len(tasks_results)
            statistics["tasks_with_at_least_one_success"] = tasks_with_at_least_one_success
            statistics["tasks_with_at_least_one_success_percent"] = round(
                (tasks_with_at_least_one_success / total_tasks * 100) if total_tasks > 0 else 0, 2
            )
            statistics["tasks_with_all_success"] = tasks_with_all_success
            statistics["tasks_with_all_success_percent"] = round(
                (tasks_with_all_success / total_tasks * 100) if total_tasks > 0 else 0, 2
            )
            if average_auc is not None:
                statistics["average_cumulative_auc"] = round(average_auc, 4)

        # Save JSON file
        incremental_data = {
            "metadata": metadata,
            "results": all_results,
            "statistics": statistics
        }

        with open(self.incremental_json, 'w') as f:
            json.dump(incremental_data, f, indent=2)

        # Save summary text file
        with open(self.incremental_summary, 'w') as f:
            f.write("="*70 + "\n")
            f.write("INCREMENTAL RESULTS SUMMARY\n")
            f.write("="*70 + "\n\n")

            f.write("Configuration:\n")
            f.write(f"  Model: {metadata.get('model', 'N/A')}\n")
            f.write(f"  Total Tasks: {metadata.get('total_tasks', 'N/A')}\n")
            f.write(f"  Repeats per Task: {metadata.get('repeats_per_task', 1)}\n")
            f.write(f"  Workers: {metadata.get('workers', 1)}\n\n")

            f.write("Overall Statistics:\n")
            f.write(f"  Completed Runs: {total_runs}\n")
            f.write(f"  Successful: {successful_runs} ({overall_accuracy:.2f}%)\n")
            f.write(f"  Failed: {failed_runs}\n")
            f.write(f"  Errors: {error_runs}\n")
            f.write(f"  Completed Tasks: {len(tasks_results)}\n")

            # Add task-level statistics if repeat > 1
            if repeats_per_task > 1 and len(tasks_results) > 0:
                f.write(f"\nTask-Level Statistics (repeat > 1):\n")
                f.write(f"  Tasks with ≥1 success: {tasks_with_at_least_one_success}/{len(tasks_results)} ")
                f.write(f"({statistics.get('tasks_with_at_least_one_success_percent', 0):.2f}%)\n")
                f.write(f"  Tasks with all success: {tasks_with_all_success}/{len(tasks_results)} ")
                f.write(f"({statistics.get('tasks_with_all_success_percent', 0):.2f}%)\n")
                if average_auc is not None:
                    f.write(f"  Average Cumulative AUC: {average_auc:.4f}\n")

            f.write("\nTask Status:\n")
            f.write("-"*70 + "\n")
            for tid in sorted(tasks_results.keys()):
                results = tasks_results[tid]
                accuracy = task_accuracies[tid] / 100.0  # Convert to 0-1 range

                # Create success sequence: 1 for success, 0 for failure/error
                success_sequence = ','.join(['1' if r.get("status") == "success" else '0' for r in results])

                # Get AUC (or 0 if not available)
                auc = task_aucs.get(tid, 0.0)

                # Format: Task ID | sequence | AUC: X.XXXX, ACC: X.XX
                f.write(f"Task {tid:>4} | {success_sequence:<20} | AUC: {auc:.4f}, ACC: {accuracy:.2f}\n")

    def group_tasks_by_site_and_consecutive(self, task_ids: List[int]) -> List[Dict]:
        """
        Group tasks by site and consecutive task IDs.

        Returns a list of groups, where each group is a dict with:
        - 'site': the site name
        - 'tasks': list of consecutive task IDs
        - 'start': first task ID in the group
        - 'end': last task ID in the group
        """
        if not task_ids:
            return []

        # First, organize tasks by site
        tasks_by_site = defaultdict(list)
        for tid in task_ids:
            if tid in self.all_configs:
                site = self.all_configs[tid]["sites"][0]
                tasks_by_site[site].append(tid)

        # Sort tasks within each site
        for site in tasks_by_site:
            tasks_by_site[site].sort()

        # Now group consecutive tasks within each site
        groups = []
        for site, tasks in sorted(tasks_by_site.items()):
            if not tasks:
                continue

            current_group = [tasks[0]]

            for i in range(1, len(tasks)):
                # Check if consecutive
                if tasks[i] == tasks[i-1] + 1:
                    current_group.append(tasks[i])
                else:
                    # Save current group and start new one
                    groups.append({
                        'site': site,
                        'tasks': current_group,
                        'start': current_group[0],
                        'end': current_group[-1]
                    })
                    current_group = [tasks[i]]

            # Don't forget the last group
            groups.append({
                'site': site,
                'tasks': current_group,
                'start': current_group[0],
                'end': current_group[-1]
            })

        return groups

    def is_task_completed(self, task_id: int) -> bool:
        """Check if a task has been completed successfully"""
        # Find the most recent result directory for this task
        pattern = f"*webarena.{task_id}"
        matching_dirs = sorted(self.results_dir.glob(pattern))

        if not matching_dirs:
            return False

        # Check the most recent result
        latest_dir = matching_dirs[-1]
        summary_file = latest_dir / "summary_info.json"

        if not summary_file.exists():
            return False

        try:
            with open(summary_file, 'r') as f:
                summary = json.load(f)

            # Check if task completed successfully
            return summary.get("cum_reward", 0) > 0 and summary.get("err_msg") is None
        except Exception as e:
            print(f"Warning: Could not read summary for task {task_id}: {e}")
            return False

    def generate_config_file(self, task_id: int) -> Path:
        """Generate config file for a specific task"""
        if task_id not in self.all_configs:
            raise ValueError(f"Task {task_id} not found in test.raw.json")

        config = self.all_configs[task_id]
        config_file = self.config_dir / f"{task_id}.json"

        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        return config_file

    def get_workflow_file(self, task_id: int) -> Path:
        """Get or create workflow file for a task"""
        config = self.all_configs[task_id]
        website = config["sites"][0]
        workflow_file = self.workflow_dir / f"{website}.txt"

        # Create empty workflow file if it doesn't exist
        if not workflow_file.exists():
            workflow_file.parent.mkdir(exist_ok=True)
            workflow_file.touch()

        return workflow_file

    def run_task(self, task_id: int, run_number: int = 1) -> dict:
        """Run a single task and return the result"""
        print(f"\n{'='*60}")
        print(f"Starting Task {task_id} (Run {run_number}/{self.repeat})")
        print(f"{'='*60}")

        # Generate config file
        config_file = self.generate_config_file(task_id)
        workflow_file = self.get_workflow_file(task_id)

        config = self.all_configs[task_id]
        print(f"Website: {config['sites'][0]}")
        print(f"Intent: {config['intent']}")
        print(f"Model: {self.model}")
        print(f"Max steps: {self.max_steps}")

        # Create log file for this task (include run number if repeat > 1)
        if self.repeat > 1:
            task_log_file = self.batch_log_dir / f"task_{task_id}_run{run_number}.log"
        else:
            task_log_file = self.batch_log_dir / f"task_{task_id}.log"

        # Build command
        cmd = [
            "python", "run.py",
            "--task_name", f"webarena.{task_id}",
            "--workflow_path", str(workflow_file),
            "--max_steps", str(self.max_steps),
            "--headless", str(self.headless).lower(),
            "--model_name", self.model,
        ]

        # Run the task and save output to log file
        start_time = time.time()
        try:
            with open(task_log_file, 'w') as log_f:
                # Write command and metadata
                log_f.write(f"Task ID: {task_id}\n")
                log_f.write(f"Command: {' '.join(cmd)}\n")
                log_f.write(f"Start time: {datetime.now().isoformat()}\n")
                log_f.write("="*60 + "\n\n")
                log_f.flush()

                result = subprocess.run(
                    cmd,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=600  # 10 minutes timeout
                )

            elapsed_time = time.time() - start_time

            # Find the result directory
            pattern = f"*webarena.{task_id}*"
            matching_dirs = sorted(self.results_dir.glob(pattern))

            if matching_dirs:
                latest_dir = matching_dirs[-1]
                summary_file = latest_dir / "summary_info.json"

                if summary_file.exists():
                    with open(summary_file, 'r') as f:
                        summary = json.load(f)

                    success = summary.get("cum_reward", 0) > 0

                    # Determine status
                    if success:
                        status = "success"
                    elif summary.get("err_msg"):
                        status = "error"
                    else:
                        status = "failure"

                    result_data = {
                        "task_id": task_id,
                        "repeat": run_number,
                        "success": success,
                        "status": status,
                        "cum_reward": summary.get("cum_reward", 0),
                        "n_steps": summary.get("n_steps", 0),
                        "error": summary.get("err_msg"),
                        "duration": round(elapsed_time, 2),
                        "elapsed_time": elapsed_time,  # Keep for backward compatibility
                        "result_dir": str(latest_dir),
                        "log_file": str(task_log_file),
                        "intent": config["intent"],
                        "website": config["sites"][0],
                        "timestamp": datetime.now().isoformat(),
                    }

                    # LLM evaluation for string_match tasks
                    eval_config = config.get("eval", {})
                    eval_types = eval_config.get("eval_types", [])

                    if "string_match" in eval_types and self.llm_evaluator:
                        print(f"  Running LLM evaluation for string_match task...")
                        try:
                            import re
                            # Extract agent response from log
                            with open(task_log_file, 'r') as f:
                                log_content = f.read()

                            matches = re.findall(r"send_msg_to_user\(['\"](.+?)['\"]\)", log_content, re.DOTALL)
                            if matches:
                                agent_response = matches[-1]

                                # Get reference answer
                                reference_answers = eval_config.get("reference_answers", {})
                                reference_answer = (
                                    reference_answers.get("exact_match") or
                                    reference_answers.get("must_include") or
                                    reference_answers.get("fuzzy_match") or
                                    ""
                                )

                                if reference_answer:
                                    llm_eval = self.llm_evaluator.evaluate(
                                        config["intent"],
                                        agent_response,
                                        reference_answer
                                    )

                                    result_data["llm_eval"] = llm_eval
                                    result_data["llm_reward"] = llm_eval.get("reward", 0.0)

                                    # If GT eval failed but LLM eval succeeded, note it
                                    if not success and llm_eval.get("success", False):
                                        result_data["llm_overrides_gt"] = True
                                        print(f"  ✅ LLM evaluation: SUCCESS (GT eval was FAIL)")
                                    elif success and not llm_eval.get("success", False):
                                        result_data["llm_disagrees_with_gt"] = True
                                        print(f"  ❌ LLM evaluation: FAIL (GT eval was SUCCESS)")
                                    else:
                                        print(f"  LLM evaluation agrees with GT eval")
                        except Exception as e:
                            print(f"  Warning: LLM evaluation failed: {e}")
                            result_data["llm_eval_error"] = str(e)

                    # Append completion info to log
                    with open(task_log_file, 'a') as log_f:
                        log_f.write("\n" + "="*60 + "\n")
                        log_f.write(f"End time: {datetime.now().isoformat()}\n")
                        log_f.write(f"Elapsed time: {elapsed_time:.2f}s\n")
                        log_f.write(f"Success: {success}\n")
                        log_f.write(f"Cum reward: {summary.get('cum_reward', 0)}\n")
                        log_f.write(f"Steps: {summary.get('n_steps', 0)}\n")

                        # Append LLM evaluation results
                        if "llm_eval" in result_data:
                            log_f.write("\n" + "-"*60 + "\n")
                            log_f.write("LLM Evaluation (string_match):\n")
                            log_f.write(f"LLM Success: {result_data['llm_eval'].get('success', False)}\n")
                            log_f.write(f"LLM Reward: {result_data['llm_reward']}\n")
                            log_f.write(f"LLM Thoughts: {result_data['llm_eval'].get('thoughts', '')}\n")

                    # Step 2: Run autoeval (like pipeline.py)
                    if self.use_autoeval:
                        print(f"  Running autoeval for task {task_id}...")
                        try:
                            autoeval_result = self._run_autoeval(task_id, latest_dir, task_log_file)
                            result_data["autoeval"] = autoeval_result
                        except Exception as e:
                            print(f"  Warning: Autoeval failed: {e}")
                            result_data["autoeval_error"] = str(e)

                    # Step 3: Update workflow (like pipeline.py)
                    if self.update_workflow:
                        print(f"  Updating workflow for {config['sites'][0]}...")
                        try:
                            workflow_update_result = self._update_workflow(config['sites'][0], task_log_file)
                            result_data["workflow_updated"] = workflow_update_result
                        except Exception as e:
                            print(f"  Warning: Workflow update failed: {e}")
                            result_data["workflow_update_error"] = str(e)

                    return result_data

            # No result directory found
            with open(task_log_file, 'a') as log_f:
                log_f.write("\n" + "="*60 + "\n")
                log_f.write("ERROR: No result directory found\n")

            return {
                "task_id": task_id,
                "repeat": run_number,
                "success": False,
                "status": "error",
                "error": "No result directory found",
                "duration": round(elapsed_time, 2),
                "elapsed_time": elapsed_time,
                "log_file": str(task_log_file),
                "intent": config["intent"],
                "website": config["sites"][0],
                "timestamp": datetime.now().isoformat(),
            }

        except subprocess.TimeoutExpired:
            with open(task_log_file, 'a') as log_f:
                log_f.write("\n" + "="*60 + "\n")
                log_f.write("ERROR: Timeout (10 minutes)\n")

            return {
                "task_id": task_id,
                "repeat": run_number,
                "success": False,
                "status": "error",
                "error": "Timeout (10 minutes)",
                "duration": 600,
                "elapsed_time": 600,
                "log_file": str(task_log_file),
                "intent": config["intent"],
                "website": config["sites"][0],
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            with open(task_log_file, 'a') as log_f:
                log_f.write("\n" + "="*60 + "\n")
                log_f.write(f"ERROR: {str(e)}\n")

            elapsed = time.time() - start_time
            return {
                "task_id": task_id,
                "repeat": run_number,
                "success": False,
                "status": "error",
                "error": str(e),
                "duration": round(elapsed, 2),
                "elapsed_time": elapsed,
                "log_file": str(task_log_file),
                "intent": config["intent"],
                "website": config["sites"][0],
                "timestamp": datetime.now().isoformat(),
            }

    def _run_autoeval(self, task_id: int, result_dir: Path, task_log_file: Path) -> dict:
        """Run autoeval on a task result (Step 2 of pipeline.py)"""
        cmd = [
            "python", "-m", "autoeval.evaluate_trajectory",
            "--result_dir", str(result_dir)
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120  # 2 minutes timeout
            )

            # Append autoeval output to log
            with open(task_log_file, 'a') as log_f:
                log_f.write("\n" + "="*60 + "\n")
                log_f.write("Autoeval Output:\n")
                log_f.write("-"*60 + "\n")
                log_f.write(result.stdout)
                if result.stderr:
                    log_f.write("\nAutoeval Errors:\n")
                    log_f.write(result.stderr)

            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout[:500],  # Truncate for storage
            }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Autoeval timeout"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _update_workflow(self, website: str, task_log_file: Path) -> dict:
        """Update workflow using induce_prompt.py (Step 3 of pipeline.py)"""
        workflow_file = self.workflow_dir / f"{website}.txt"

        cmd = [
            "python", "induce_prompt.py",
            "--result_dir", str(self.results_dir),
            "--output_path", str(workflow_file),
            "--criteria", "gt",  # Use GT reward (cum_reward > 0)
            "--model", self.induce_model,
            "--num_samples", str(self.num_samples),
        ]

        try:
            # Pass environment variables to subprocess
            env = os.environ.copy()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes timeout
                env=env
            )

            # Append workflow update output to log
            with open(task_log_file, 'a') as log_f:
                log_f.write("\n" + "="*60 + "\n")
                log_f.write(f"Workflow Update ({website}):\n")
                log_f.write("-"*60 + "\n")
                log_f.write(result.stdout)
                if result.stderr:
                    log_f.write("\nWorkflow Update Errors:\n")
                    log_f.write(result.stderr)

            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "workflow_file": str(workflow_file),
                "stdout": result.stdout[:500],  # Truncate for storage
            }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Workflow update timeout"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def run_group(self, group: Dict, group_idx: int, total_groups: int, on_task_complete=None) -> List[dict]:
        """
        Run a single group of tasks (tasks within a group are executed serially).

        Args:
            group: Dict with 'site', 'tasks', 'start', 'end'
            group_idx: Index of this group (for display)
            total_groups: Total number of groups (for display)
            on_task_complete: Optional callback function called after each task completes

        Returns:
            List of result dicts for all tasks in this group
        """
        results = []
        site = group['site']
        tasks = group['tasks']

        print(f"\n{'='*60}")
        print(f"Group {group_idx}/{total_groups}: {site} [Tasks {group['start']}-{group['end']}]")
        print(f"Running {len(tasks)} tasks serially...")
        print(f"{'='*60}\n")

        for task_num, task_id in enumerate(tasks, 1):
            # Run each task repeat times
            for run_num in range(1, self.repeat + 1):
                if self.repeat > 1:
                    print(f"\n  [{task_num}/{len(tasks)}] Task {task_id} (run {run_num}/{self.repeat})...")
                else:
                    print(f"\n  [{task_num}/{len(tasks)}] Task {task_id}...")

                result = self.run_task(task_id, run_number=run_num)
                results.append(result)

                # Call the callback for incremental save
                if on_task_complete:
                    on_task_complete(result)

                # Print summary
                status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
                print(f"  {status}: cum_reward={result.get('cum_reward', 0)}, "
                      f"steps={result.get('n_steps', 0)}, "
                      f"time={result.get('elapsed_time', 0):.1f}s")

                if result.get("error"):
                    print(f"  Error: {result['error'][:100]}")

        print(f"\nGroup {group_idx} completed: {len(results)} runs finished\n")
        return results

    def run_batch_parallel(
        self,
        task_ids: List[int],
        workers: int = 1,
        resume: bool = False,
        save_report: bool = True,
    ) -> List[dict]:
        """
        Run a batch of tasks with parallel group execution.

        Groups are formed by site + consecutive IDs.
        Groups run in parallel (up to 'workers' groups at once).
        Tasks within each group run serially.
        """
        # Filter out completed tasks if resume is enabled
        if resume:
            original_count = len(task_ids)
            task_ids = [tid for tid in task_ids if not self.is_task_completed(tid)]
            skipped = original_count - len(task_ids)
            if skipped > 0:
                print(f"\nResuming: Skipping {skipped} completed tasks")

        # Group tasks by site and consecutive IDs
        groups = self.group_tasks_by_site_and_consecutive(task_ids)

        total_tasks = len(task_ids)
        total_groups = len(groups)
        total_runs = total_tasks * self.repeat

        # Create metadata for results
        metadata = {
            "benchmark": "webarena",
            "model": self.model,
            "eval_model": self.llm_eval_model if self.use_llm_eval else None,
            "total_tasks": total_tasks,
            "repeats_per_task": self.repeat,
            "workers": workers,
            "max_steps": self.max_steps,
            "start_time": datetime.now().isoformat(),
        }

        print(f"\n{'='*60}")
        print(f"Batch Run (Parallel Groups)")
        print(f"{'='*60}")
        print(f"Total tasks: {total_tasks}")
        print(f"Total groups (by site + consecutive IDs): {total_groups}")
        if self.repeat > 1:
            print(f"Repeat: {self.repeat} times per task ({total_runs} total runs)")
        print(f"Workers: {workers} (groups run in parallel)")
        print(f"Model: {self.model}")
        print(f"Max steps: {self.max_steps}")
        print(f"{'='*60}\n")

        # Print group details
        print("Group details:")
        for i, group in enumerate(groups, 1):
            task_range = f"{group['start']}-{group['end']}" if group['start'] != group['end'] else str(group['start'])
            print(f"  Group {i}: {group['site']} [Tasks {task_range}] ({len(group['tasks'])} tasks)")
        print()

        all_results = []
        results_lock = Lock()  # Thread-safe lock for results list

        # Define callback for incremental save
        def on_task_complete(result):
            with results_lock:
                all_results.append(result)
                # Save incremental results
                self._save_incremental_results(all_results, metadata)

        if workers == 1:
            # Sequential execution
            for i, group in enumerate(groups, 1):
                self.run_group(group, i, total_groups, on_task_complete=on_task_complete)
                # Results already added via callback, no need to extend
        else:
            # Parallel execution using ThreadPoolExecutor
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=workers) as executor:
                # Submit all groups
                future_to_group = {
                    executor.submit(self.run_group, group, i, total_groups, on_task_complete): (i, group)
                    for i, group in enumerate(groups, 1)
                }

                # Collect results as they complete
                for future in as_completed(future_to_group):
                    group_idx, group = future_to_group[future]
                    try:
                        future.result()
                        # Results already added via callback, no need to extend
                    except Exception as e:
                        print(f"Group {group_idx} generated an exception: {e}")

        # Print final summary
        self._print_summary(all_results)

        # Final save with end time
        metadata["end_time"] = datetime.now().isoformat()
        metadata["total_duration_seconds"] = sum(r.get("duration", 0) for r in all_results)
        if save_report:
            self._save_incremental_results(all_results, metadata)

        print(f"\n📊 Results saved:")
        print(f"  - Incremental JSON: {self.incremental_json}")
        print(f"  - Summary TXT: {self.incremental_summary}")
        print(f"  - Logs directory: {self.batch_log_dir}")

        return all_results

    def run_batch(
        self,
        task_ids: List[int],
        resume: bool = False,
        save_report: bool = True,
    ) -> List[dict]:
        """Run a batch of tasks (legacy sequential mode)"""
        results = []

        # Filter out completed tasks if resume is enabled
        if resume:
            original_count = len(task_ids)
            task_ids = [tid for tid in task_ids if not self.is_task_completed(tid)]
            skipped = original_count - len(task_ids)
            if skipped > 0:
                print(f"\nResuming: Skipping {skipped} completed tasks")

        total = len(task_ids)
        total_runs = total * self.repeat

        # Create metadata for results
        metadata = {
            "benchmark": "webarena",
            "model": self.model,
            "eval_model": self.llm_eval_model if self.use_llm_eval else None,
            "total_tasks": total,
            "repeats_per_task": self.repeat,
            "workers": 1,
            "max_steps": self.max_steps,
            "start_time": datetime.now().isoformat(),
        }

        print(f"\n{'='*60}")
        print(f"Batch Run: {total} tasks")
        if self.repeat > 1:
            print(f"Repeat: {self.repeat} times per task ({total_runs} total runs)")
        print(f"Model: {self.model}")
        print(f"Max steps: {self.max_steps}")
        print(f"{'='*60}\n")

        run_counter = 0
        for task_id in task_ids:
            # Run each task repeat times
            for run_num in range(1, self.repeat + 1):
                run_counter += 1
                if self.repeat > 1:
                    print(f"\n[{run_counter}/{total_runs}] Running task {task_id} (run {run_num}/{self.repeat})...")
                else:
                    print(f"\n[{run_counter}/{total_runs}] Running task {task_id}...")

                result = self.run_task(task_id, run_number=run_num)
                results.append(result)

                # Save incrementally after each task
                if save_report:
                    self._save_incremental_results(results, metadata)

                # Print summary
                status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
                print(f"\n{status}")
                print(f"Task {task_id} (run {run_num}): cum_reward={result.get('cum_reward', 0)}, "
                      f"steps={result.get('n_steps', 0)}, "
                      f"time={result.get('elapsed_time', 0):.1f}s")

                if result.get("error"):
                    print(f"Error: {result['error'][:200]}")

        # Print final summary
        self._print_summary(results)

        # Final save with end time
        metadata["end_time"] = datetime.now().isoformat()
        metadata["total_duration_seconds"] = sum(r.get("duration", 0) for r in results)
        if save_report:
            self._save_incremental_results(results, metadata)

        print(f"\n📊 Results saved:")
        print(f"  - Incremental JSON: {self.incremental_json}")
        print(f"  - Summary TXT: {self.incremental_summary}")
        print(f"  - Logs directory: {self.batch_log_dir}")

        return results

    def _print_summary(self, results: List[dict]):
        """Print summary of batch run"""
        total = len(results)
        successful = sum(1 for r in results if r["success"])
        failed = total - successful

        total_time = sum(r.get("elapsed_time", 0) for r in results)
        avg_time = total_time / total if total > 0 else 0

        print(f"\n{'='*60}")
        print(f"Batch Run Summary")
        print(f"{'='*60}")
        print(f"Total tasks: {total}")
        print(f"Successful: {successful} ({successful/total*100:.1f}%)")
        print(f"Failed: {failed} ({failed/total*100:.1f}%)")
        print(f"Total time: {total_time:.1f}s")
        print(f"Average time per task: {avg_time:.1f}s")
        print(f"{'='*60}")

        # Print failed tasks
        if failed > 0:
            print(f"\nFailed tasks:")
            for r in results:
                if not r["success"]:
                    error_msg = r.get("error", "Unknown error")
                    if error_msg is None:
                        error_msg = "Unknown error"
                    error_msg = str(error_msg)[:100]
                    print(f"  - Task {r['task_id']}: {error_msg}")

    def _save_report(self, results: List[dict]):
        """Save batch run report with detailed statistics"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save to batch_logs directory
        summary_file = self.batch_log_dir / "summary.json"
        detailed_file = self.batch_log_dir / "detailed_results.json"

        # Helper function to determine if a task is successful
        def is_task_successful(r: dict) -> bool:
            """Use LLM eval if available and overrides GT, otherwise use GT success"""
            if r.get("llm_overrides_gt", False):
                # LLM says success but GT says failure -> use LLM
                return r.get("llm_eval", {}).get("success", False)
            return r["success"]

        # Calculate statistics
        total = len(results)
        successful = sum(1 for r in results if is_task_successful(r))
        failed = total - successful
        total_time = sum(r.get("elapsed_time", 0) for r in results)
        avg_time = total_time / total if total > 0 else 0

        # Group by website
        by_website = {}
        for r in results:
            website = r.get("website", "unknown")
            if website not in by_website:
                by_website[website] = {"total": 0, "successful": 0, "failed": 0}
            by_website[website]["total"] += 1
            if is_task_successful(r):
                by_website[website]["successful"] += 1
            else:
                by_website[website]["failed"] += 1

        # Create summary report
        summary = {
            "batch_id": self.batch_log_dir.name,
            "timestamp": timestamp,
            "model": self.model,
            "max_steps": self.max_steps,
            "repeat": self.repeat,
            "statistics": {
                "total_runs": total,
                "successful_runs": successful,
                "failed_runs": failed,
                "success_rate": f"{successful/total*100:.2f}%" if total > 0 else "0%",
                "total_time_seconds": round(total_time, 2),
                "average_time_per_run_seconds": round(avg_time, 2),
            },
            "by_website": by_website,
            "successful_runs": [{"task_id": r["task_id"], "run_number": r.get("run_number", 1)}
                               for r in results if is_task_successful(r)],
            "failed_runs": [{"task_id": r["task_id"], "run_number": r.get("run_number", 1)}
                           for r in results if not is_task_successful(r)],
            "log_directory": str(self.batch_log_dir),
        }

        # Save summary
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        # Save detailed results
        detailed_report = {
            "batch_id": self.batch_log_dir.name,
            "timestamp": timestamp,
            "model": self.model,
            "max_steps": self.max_steps,
            "tasks": results,
        }

        with open(detailed_file, 'w') as f:
            json.dump(detailed_report, f, indent=2)

        # Also save a legacy report in results directory for backward compatibility
        legacy_report_file = self.results_dir / f"batch_report_{timestamp}.json"
        with open(legacy_report_file, 'w') as f:
            json.dump(detailed_report, f, indent=2)

        print(f"\n📊 Reports saved:")
        print(f"  - Summary: {summary_file}")
        print(f"  - Detailed: {detailed_file}")
        print(f"  - Legacy: {legacy_report_file}")
        print(f"  - Logs directory: {self.batch_log_dir}")


def main():
    parser = argparse.ArgumentParser(description="Batch runner for WebArena tasks")

    # Task selection (no longer required, can use --sites instead)
    task_group = parser.add_mutually_exclusive_group(required=False)
    task_group.add_argument("--tasks", type=str, help="Comma-separated task IDs (e.g., 700,701,702)")
    task_group.add_argument("--start", type=int, help="Start task ID (inclusive)")

    parser.add_argument("--end", type=int, help="End task ID (inclusive, required with --start)")
    parser.add_argument("--website", type=str,
                       choices=["shopping", "shopping_admin", "gitlab", "reddit", "map"],
                       help="Filter tasks by website")
    parser.add_argument("--sites", type=str,
                       help="Filter tasks by site (comma-separated, e.g., 'shopping_admin,gitlab')")
    parser.add_argument("--workers", type=int, default=1,
                       help="Number of parallel workers for group execution (default: 1)")

    # Execution options
    parser.add_argument("--model", type=str,
                       default="google/gemini-2.5-flash-preview-09-2025",
                       help="Model to use")
    parser.add_argument("--max_steps", type=int, default=20,
                       help="Maximum number of steps per task")
    parser.add_argument("--headless", action="store_true", default=True,
                       help="Run in headless mode")
    parser.add_argument("--no-headless", action="store_false", dest="headless",
                       help="Run with visible browser")

    # Resume option
    parser.add_argument("--resume", action="store_true",
                       help="Skip already completed tasks")

    # LLM evaluation options
    parser.add_argument("--no-llm-eval", action="store_false", dest="use_llm_eval",
                       default=True,
                       help="Disable LLM evaluation for string_match tasks")
    parser.add_argument("--llm-eval-model", type=str,
                       default="google/gemini-2.5-flash-preview-09-2025",
                       help="Model to use for LLM evaluation (default: gemini-2.5-flash)")

    # Pipeline options (like pipeline.py)
    parser.add_argument("--use-autoeval", action="store_true",
                       help="Run autoeval after each task (Step 2 of pipeline.py)")
    parser.add_argument("--update-workflow", action="store_true",
                       help="Update workflow after each task using induce_prompt.py (Step 3 of pipeline.py)")
    parser.add_argument("--induce-model", type=str,
                       default="google/gemini-2.5-flash-preview-09-2025",
                       help="Model to use for workflow induction (default: gemini-2.5-flash)")
    parser.add_argument("--num-samples", type=int,
                       default=1,
                       help="Number of samples per template for workflow induction")
    parser.add_argument("--repeat", type=int,
                       default=1,
                       help="Number of times to repeat each task (default: 1)")

    args = parser.parse_args()

    # Validate arguments
    if args.start is not None and args.end is None:
        parser.error("--end is required when using --start")

    # Ensure at least one task selection method is specified
    if not args.tasks and not args.sites and not args.website and args.start is None:
        parser.error("You must specify one of: --tasks, --sites, --website, or --start/--end")

    # Create runner
    runner = BatchRunner(
        model=args.model,
        max_steps=args.max_steps,
        headless=args.headless,
        use_llm_eval=args.use_llm_eval,
        llm_eval_model=args.llm_eval_model,
        use_autoeval=args.use_autoeval,
        update_workflow=args.update_workflow,
        induce_model=args.induce_model,
        num_samples=args.num_samples,
        repeat=args.repeat,
    )

    # Get task IDs
    if args.tasks:
        task_ids = [int(tid.strip()) for tid in args.tasks.split(",")]
    elif args.sites:
        # Filter by sites
        sites_list = [s.strip() for s in args.sites.split(",")]
        task_ids = runner.get_tasks_by_sites(sites_list)
    elif args.website:
        task_ids = runner.get_tasks_by_website(args.website, args.start or 0, args.end or 999999)
    else:
        task_ids = runner.get_tasks_by_range(args.start, args.end)

    if not task_ids:
        print("No tasks found matching the criteria")
        return

    # Run batch
    # Use parallel execution if workers > 1 or if sites is specified
    if args.workers > 1 or args.sites:
        runner.run_batch_parallel(task_ids, workers=args.workers, resume=args.resume)
    else:
        runner.run_batch(task_ids, resume=args.resume)


if __name__ == "__main__":
    main()
