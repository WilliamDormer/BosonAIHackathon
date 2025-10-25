"""
Universal logger for audio-based agentic pipeline.
Logs each step and result as JSON with unique run IDs based on date/time.
"""

import os
import json
from datetime import datetime
from typing import Any, Dict, Optional


class RunLogger:
    """Logger that saves intermediate steps and results to JSON files."""
    
    def __init__(self, run_name: Optional[str] = None, base_dir: str = "runs"):
        """
        Initialize logger with unique run identifier.
        
        Args:
            run_name: Optional custom run name. If None, generates from datetime
            base_dir: Base directory for saving logs (default: "runs")
        """
        # Generate unique run name from datetime if not provided
        if run_name is None:
            now = datetime.now()
            run_name = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        
        self.run_name = run_name
        self.base_dir = base_dir
        
        # Create run directory
        self.run_dir = os.path.join(base_dir, run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Initialize log structure
        self.logs: Dict[str, Any] = {
            "run_name": run_name,
            "start_time": datetime.now().isoformat(),
            "steps": [],
            "errors": [],
            "warnings": [],
            "results": {}
        }
        
        # Step counter for ordering
        self.step_counter = 0
        
        print(f"[Logger] Initialized run: {run_name}")
        print(f"[Logger] Log directory: {self.run_dir}")
    
    def log_step(self, step_name: str, data: Dict[str, Any]) -> None:
        """
        Log a step in the pipeline.
        
        Args:
            step_name: Name/identifier of the step
            data: Data associated with the step
        """
        self.step_counter += 1
        
        step_entry = {
            "step_number": self.step_counter,
            "step_name": step_name,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        self.logs["steps"].append(step_entry)
        
        # Save after each step
        self._save_logs()
        
        print(f"[Logger] Step {self.step_counter}: {step_name}")
    
    def log_error(self, error_name: str, error_message: str) -> None:
        """
        Log an error.
        
        Args:
            error_name: Name/identifier of the error
            error_message: Error message/description
        """
        error_entry = {
            "error_name": error_name,
            "timestamp": datetime.now().isoformat(),
            "message": error_message
        }
        
        self.logs["errors"].append(error_entry)
        self._save_logs()
        
        print(f"[Logger] Error: {error_name} - {error_message}")
    
    def log_warning(self, warning_name: str, warning_message: str) -> None:
        """
        Log a warning.
        
        Args:
            warning_name: Name/identifier of the warning
            warning_message: Warning message/description
        """
        warning_entry = {
            "warning_name": warning_name,
            "timestamp": datetime.now().isoformat(),
            "message": warning_message
        }
        
        self.logs["warnings"].append(warning_entry)
        self._save_logs()
        
        print(f"[Logger] Warning: {warning_name} - {warning_message}")
    
    def log_result(self, result_name: str, result_data: Any) -> None:
        """
        Log a final result.
        
        Args:
            result_name: Name/identifier of the result
            result_data: Result data (will be JSON serialized)
        """
        self.logs["results"][result_name] = {
            "timestamp": datetime.now().isoformat(),
            "data": result_data
        }
        
        self._save_logs()
        
        print(f"[Logger] Result: {result_name}")
    
    def _save_logs(self) -> None:
        """Save logs to JSON file."""
        log_file = os.path.join(self.run_dir, "log.json")
        
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(self.logs, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Logger] Error saving logs: {str(e)}")
    
    def finalize(self) -> None:
        """Finalize the run and save summary."""
        self.logs["end_time"] = datetime.now().isoformat()
        
        # Calculate duration
        try:
            start = datetime.fromisoformat(self.logs["start_time"])
            end = datetime.fromisoformat(self.logs["end_time"])
            duration = (end - start).total_seconds()
            self.logs["duration_seconds"] = duration
        except Exception:
            pass
        
        # Save final logs
        self._save_logs()
        
        # Create summary
        summary = {
            "run_name": self.run_name,
            "start_time": self.logs["start_time"],
            "end_time": self.logs["end_time"],
            "duration_seconds": self.logs.get("duration_seconds", 0),
            "total_steps": len(self.logs["steps"]),
            "total_errors": len(self.logs["errors"]),
            "total_warnings": len(self.logs["warnings"]),
            "results_count": len(self.logs["results"])
        }
        
        summary_file = os.path.join(self.run_dir, "summary.json")
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Logger] Error saving summary: {str(e)}")
        
        print(f"[Logger] Run finalized: {self.run_name}")
        print(f"[Logger] Total steps: {summary['total_steps']}")
        print(f"[Logger] Total errors: {summary['total_errors']}")
        print(f"[Logger] Total warnings: {summary['total_warnings']}")
    
    def get_log_path(self) -> str:
        """Get the path to the log file."""
        return os.path.join(self.run_dir, "log.json")
    
    def get_run_dir(self) -> str:
        """Get the run directory path."""
        return self.run_dir
    
    def save_file(self, filename: str, content: Any, mode: str = 'text') -> str:
        """
        Save additional file to the run directory.
        
        Args:
            filename: Name of the file
            content: Content to save
            mode: 'text' for text files, 'json' for JSON files, 'binary' for binary
        
        Returns:
            Path to saved file
        """
        file_path = os.path.join(self.run_dir, filename)
        
        try:
            if mode == 'json':
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(content, f, indent=2, ensure_ascii=False)
            elif mode == 'text':
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(str(content))
            elif mode == 'binary':
                with open(file_path, 'wb') as f:
                    f.write(content)
            else:
                raise ValueError(f"Unknown mode: {mode}")
            
            print(f"[Logger] Saved file: {filename}")
            return file_path
            
        except Exception as e:
            print(f"[Logger] Error saving file {filename}: {str(e)}")
            raise


def create_logger(run_name: Optional[str] = None, base_dir: str = "runs") -> RunLogger:
    """
    Factory function to create a new logger instance.
    
    Args:
        run_name: Optional custom run name
        base_dir: Base directory for logs
    
    Returns:
        RunLogger instance
    """
    return RunLogger(run_name=run_name, base_dir=base_dir)


if __name__ == "__main__":
    # Test the logger
    print("Testing RunLogger...")
    
    # Create logger
    logger = create_logger()
    
    # Log some steps
    logger.log_step("initialization", {"status": "started", "config": {"model": "test"}})
    logger.log_step("processing", {"input": "test data", "output": "processed"})
    
    # Log warning
    logger.log_warning("test_warning", "This is a test warning")
    
    # Log error
    logger.log_error("test_error", "This is a test error")
    
    # Log results
    logger.log_result("final_output", {"result": "success", "value": 42})
    
    # Finalize
    logger.finalize()
    
    print(f"\nTest complete! Check logs at: {logger.get_log_path()}")

