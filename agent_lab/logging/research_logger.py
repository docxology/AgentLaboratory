"""
Research logger for the Agent Laboratory.

This module provides logging for research activities, including agent interactions,
metrics, and experiment results.
"""

import os
import time
import json
import logging
import datetime
from typing import Dict, Any, Optional, List, Union, Tuple
import shutil
import contextlib

# Setup base logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class ResearchLogger:
    """Logger for research activities in the Agent Laboratory."""
    
    def __init__(
        self,
        output_dir: str,
        research_topic: str,
        debug: bool = False,
        log_to_console: bool = True,
        lab_index: int = 0
    ):
        """Initialize the research logger.
        
        Args:
            output_dir: Base output directory
            research_topic: Research topic
            debug: Enable debug logging
            log_to_console: Log to console in addition to files
            lab_index: Laboratory instance index
        """
        self.output_dir = output_dir
        self.research_topic = research_topic
        self.debug = debug
        self.log_to_console = log_to_console
        self.lab_index = lab_index
        
        # Create experiment directory with timestamp and lab index
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Clean up research topic for directory name
        topic_slug = self._get_topic_slug(research_topic)
        self.experiment_dir = os.path.join(
            output_dir, 
            f"{timestamp}_{topic_slug}_lab{lab_index}"
        )
        
        # Create experiment directories
        os.makedirs(self.experiment_dir, exist_ok=True)
        self.log_dir = os.path.join(self.experiment_dir, "logs")
        self.dialog_dir = os.path.join(self.experiment_dir, "dialog")
        self.artifacts_dir = os.path.join(self.experiment_dir, "artifacts")
        self.report_dir = os.path.join(self.experiment_dir, "report")
        self.code_dir = os.path.join(self.experiment_dir, "code")
        self.visualizations_dir = os.path.join(self.experiment_dir, "visualizations")
        self.state_dir = os.path.join(self.experiment_dir, "state")
        self.metadata_dir = os.path.join(self.experiment_dir, "metadata")
        
        # Create all directories
        for directory in [
            self.log_dir, self.dialog_dir, self.artifacts_dir, 
            self.report_dir, self.code_dir, self.visualizations_dir, 
            self.state_dir, self.metadata_dir
        ]:
            os.makedirs(directory, exist_ok=True)
        
        # Initialize Python logger
        self.logger = logging.getLogger(f"agent_lab.research_{lab_index}")
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)
        self.logger.handlers = []  # Clear existing handlers
        
        # Create file handler
        log_file = os.path.join(self.log_dir, "research.log")
        file_handler = logging.FileHandler(log_file)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
        
        # Add console handler if requested
        if log_to_console:
            console_handler = logging.StreamHandler()
            console_format = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_format)
            self.logger.addHandler(console_handler)
        
        # Initialize metrics storage
        self.metrics = {}
        self.phase_metrics = {}
        self.experiment_metadata = {
            "research_topic": research_topic,
            "started_at": timestamp,
            "lab_index": lab_index
        }
        
        # Log experiment start
        self.logger.info(f"Research experiment started for topic: {research_topic}")
        self.logger.info(f"Experiment directory: {self.experiment_dir}")
        
        # Save experiment metadata
        self.save_metadata()
    
    def _get_topic_slug(self, topic: str) -> str:
        """Convert a research topic to a slug for filenames.
        
        Args:
            topic: Research topic
            
        Returns:
            str: Slugified topic
        """
        # Replace non-alphanumeric characters with underscores
        slug = "".join(c if c.isalnum() else "_" for c in topic)
        # Limit length and remove trailing underscores
        slug = slug[:30].strip("_")
        return slug
    
    def log_info(self, message: str) -> None:
        """Log an informational message.
        
        Args:
            message: Message to log
        """
        self.logger.info(message)
    
    def log_warning(self, message: str) -> None:
        """Log a warning message.
        
        Args:
            message: Message to log
        """
        self.logger.warning(message)
    
    def log_error(self, message: str) -> None:
        """Log an error message.
        
        Args:
            message: Message to log
        """
        self.logger.error(message)
    
    def log_debug(self, message: str) -> None:
        """Log a debug message.
        
        Args:
            message: Message to log
        """
        self.logger.debug(message)
    
    def log_agent_message(
        self,
        agent_name: str,
        phase: str,
        message: str,
        message_type: str,
        step: Optional[int] = None
    ) -> None:
        """Log a message from an agent.
        
        Args:
            agent_name: Name of the agent
            phase: Research phase
            message: The message content
            message_type: Type of message (prompt, response, etc.)
            step: Optional step number
        """
        # Log summary to main log
        summary = f"{agent_name} {message_type} in {phase}" + (f" step {step}" if step else "")
        self.logger.info(summary)
        
        # If in debug mode, log the full message
        if self.debug:
            self.logger.debug(f"{summary}: {message[:100]}...")
        
        # Save detailed log to dialog directory
        step_str = f"_step{step}" if step is not None else ""
        dialog_file = os.path.join(
            self.dialog_dir,
            f"{phase}_{agent_name}_{message_type}{step_str}.txt"
        )
        
        with open(dialog_file, 'w', encoding='utf-8') as f:
            f.write(message)
    
    def log_phase_start(self, phase: str) -> None:
        """Log the start of a research phase.
        
        Args:
            phase: Research phase name
        """
        self.logger.info(f"Starting research phase: {phase}")
        
        # Initialize phase metrics
        self.phase_metrics[phase] = {
            "start_time": time.time(),
            "end_time": None,
            "duration": None,
            "steps": 0,
            "status": "running"
        }
    
    def log_phase_end(self, phase: str, status: str = "completed") -> None:
        """Log the end of a research phase.
        
        Args:
            phase: Research phase name
            status: Phase completion status
        """
        if phase in self.phase_metrics:
            end_time = time.time()
            start_time = self.phase_metrics[phase]["start_time"]
            duration = end_time - start_time
            
            self.phase_metrics[phase].update({
                "end_time": end_time,
                "duration": duration,
                "status": status
            })
            
            self.logger.info(
                f"Completed research phase: {phase} in {duration:.2f} seconds with status: {status}"
            )
        else:
            self.logger.warning(f"Ending phase {phase} that was not properly started")
    
    def log_artifact(self, name: str, content: str, artifact_type: str = "text") -> str:
        """Log a research artifact.
        
        Args:
            name: Artifact name
            content: Artifact content
            artifact_type: Type of artifact
            
        Returns:
            str: Path to the saved artifact
        """
        artifact_path = os.path.join(self.artifacts_dir, f"{name}.{artifact_type}")
        
        with open(artifact_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        self.logger.info(f"Saved {artifact_type} artifact: {name}")
        return artifact_path
    
    def log_report(self, report_content: str, filename: str = "research_report") -> str:
        """Log a research report.
        
        Args:
            report_content: Report content
            filename: Report filename
            
        Returns:
            str: Path to the saved report
        """
        report_path = os.path.join(self.report_dir, f"{filename}.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.logger.info(f"Saved research report: {filename}")
        return report_path
    
    def log_code(self, code: str, filename: str) -> str:
        """Log code generated during research.
        
        Args:
            code: Code content
            filename: Code filename
            
        Returns:
            str: Path to the saved code file
        """
        code_path = os.path.join(self.code_dir, filename)
        
        with open(code_path, 'w', encoding='utf-8') as f:
            f.write(code)
        
        self.logger.info(f"Saved code: {filename}")
        return code_path
    
    def log_metric(self, name: str, value: Any) -> None:
        """Log a metric for the research experiment.
        
        Args:
            name: Metric name
            value: Metric value
        """
        self.metrics[name] = value
        self.logger.info(f"Logged metric {name}: {value}")
        
        # Save metrics to file after each update
        self.save_metrics()
    
    def log_phase_metric(self, phase: str, name: str, value: Any) -> None:
        """Log a metric for a specific research phase.
        
        Args:
            phase: Research phase
            name: Metric name
            value: Metric value
        """
        if phase not in self.phase_metrics:
            self.phase_metrics[phase] = {}
            
        self.phase_metrics[phase][name] = value
        self.logger.info(f"Logged {phase} metric {name}: {value}")
        
        # Save metrics to file after each update
        self.save_metrics()
    
    def increment_phase_step(self, phase: str) -> int:
        """Increment the step counter for a phase.
        
        Args:
            phase: Research phase
            
        Returns:
            int: New step count
        """
        if phase not in self.phase_metrics:
            self.phase_metrics[phase] = {"steps": 0}
            
        self.phase_metrics[phase]["steps"] = self.phase_metrics[phase].get("steps", 0) + 1
        return self.phase_metrics[phase]["steps"]
    
    def save_metrics(self) -> str:
        """Save all metrics to a JSON file.
        
        Returns:
            str: Path to the saved metrics file
        """
        metrics_path = os.path.join(self.metadata_dir, "metrics.json")
        
        metrics_data = {
            "experiment": self.metrics,
            "phases": self.phase_metrics
        }
        
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2)
            
        return metrics_path
    
    def save_metadata(self) -> str:
        """Save experiment metadata to a JSON file.
        
        Returns:
            str: Path to the saved metadata file
        """
        metadata_path = os.path.join(self.metadata_dir, "experiment.json")
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.experiment_metadata, f, indent=2)
            
        return metadata_path
    
    def update_experiment_metadata(self, key: str, value: Any) -> None:
        """Update experiment metadata.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.experiment_metadata[key] = value
        self.save_metadata()
    
    def get_experiment_dir(self) -> str:
        """Get the experiment directory.
        
        Returns:
            str: Path to the experiment directory
        """
        return self.experiment_dir
    
    def get_state_dir(self) -> str:
        """Get the state directory.
        
        Returns:
            str: Path to the state directory
        """
        return self.state_dir
    
    def get_visualizations_dir(self) -> str:
        """Get the visualizations directory.
        
        Returns:
            str: Path to the visualizations directory
        """
        return self.visualizations_dir
    
    def set_phase(self, phase: str) -> None:
        """
        Set the current research phase.
        
        Args:
            phase: Name of the current research phase
        """
        if self.current_phase == phase:
            return
        
        if self.current_phase:
            self.phases_completed.append(self.current_phase)
            self.logger.info(f"Phase completed: {self.current_phase}")
        
        self.current_phase = phase
        self.logger.info(f"Starting phase: {phase}")
        
        # Create phase-specific directories
        phase_dialogs_dir = os.path.join(self.dialog_dir, phase)
        phase_artifacts_dir = os.path.join(self.artifacts_dir, phase)
        phase_vis_dir = os.path.join(self.visualizations_dir, phase)
        
        for directory in [phase_dialogs_dir, phase_artifacts_dir, phase_vis_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Create phase-specific logger
        phase_log_file = os.path.join(self.log_dir, f"{phase}.log")
        phase_logger = logging.getLogger(f"research.lab{self.lab_index}.{phase}")
        phase_logger.setLevel(logging.DEBUG if self.debug else logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        for handler in list(phase_logger.handlers):
            phase_logger.removeHandler(handler)
        
        # Add file handler for phase-specific logging
        phase_handler = logging.FileHandler(phase_log_file)
        phase_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        phase_logger.addHandler(phase_handler)
        
        self.phase_logger = phase_logger
    
    def log_human_feedback(
        self,
        message: str,
        agent_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log feedback from a human user.
        
        Args:
            message: Content of the feedback
            agent_name: Name of the agent receiving feedback (if applicable)
            metadata: Additional metadata for the feedback
        """
        if agent_name:
            self.log_agent_message(agent_name, self.current_phase, message, "human", None)
        
        # Also log to a central human feedback file
        feedback_file = os.path.join(self.log_dir, "human_feedback.jsonl")
        
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "phase": self.current_phase,
            "agent": agent_name,
            "content": message
        }
        
        if metadata:
            entry["metadata"] = metadata
        
        with open(feedback_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
        
        self.logger.info(f"Human feedback logged{f' for {agent_name}' if agent_name else ''}")
    
    def save_experiment_state(self, state_data: Dict[str, Any], label: Optional[str] = None) -> str:
        """
        Save the current state of the experiment.
        
        Args:
            state_data: Dictionary containing the experiment state
            label: Optional label for the state file
            
        Returns:
            Path to the saved state file
        """
        if label:
            filename = f"state_{label}_{self.timestamp}.json"
        else:
            phase_suffix = f"_{self.current_phase}" if self.current_phase else ""
            filename = f"state{phase_suffix}_{self.timestamp}.json"
        
        state_path = os.path.join(self.state_dir, filename)
        
        with open(state_path, "w") as f:
            json.dump(state_data, f, indent=2)
        
        self.logger.info(f"Experiment state saved: {state_path}")
        return state_path
    
    def load_experiment_state(self, state_path: str) -> Dict[str, Any]:
        """
        Load an experiment state from a file.
        
        Args:
            state_path: Path to the state file
            
        Returns:
            Dictionary containing the experiment state
            
        Raises:
            FileNotFoundError: If the state file doesn't exist
        """
        if not os.path.exists(state_path):
            raise FileNotFoundError(f"State file not found: {state_path}")
        
        with open(state_path, "r") as f:
            state_data = json.load(f)
        
        self.logger.info(f"Experiment state loaded from: {state_path}")
        return state_data
    
    def save_config(self, config: Dict[str, Any]) -> str:
        """
        Save the configuration to a file.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Path to the saved configuration file
        """
        config_path = os.path.join(self.metadata_dir, "config.yaml")
        
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        
        self.logger.info(f"Configuration saved: {config_path}")
        return config_path
    
    def save_html_report(self, html_content: str, filename: str = "report.html") -> str:
        """
        Save an HTML report.
        
        Args:
            html_content: HTML content of the report
            filename: Filename for the report
            
        Returns:
            Path to the saved report
        """
        report_path = os.path.join(self.report_dir, filename)
        
        with open(report_path, "w") as f:
            f.write(html_content)
        
        self.logger.info(f"HTML report saved: {report_path}")
        return report_path
    
    def save_source_code(self, source_files: List[Tuple[str, str]]) -> None:
        """
        Save source code files used in the experiment.
        
        Args:
            source_files: List of tuples (file_path, destination_path)
        """
        for src_path, dest_rel_path in source_files:
            if not os.path.exists(src_path):
                self.logger.warning(f"Source file not found: {src_path}")
                continue
            
            dest_path = os.path.join(self.code_dir, dest_rel_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            try:
                shutil.copy2(src_path, dest_path)
                self.logger.debug(f"Source file copied: {src_path} -> {dest_path}")
            except Exception as e:
                self.logger.error(f"Failed to copy source file {src_path}: {e}")
    
    @contextlib.contextmanager
    def phase_context(self, phase: str):
        """
        Context manager for a research phase.
        
        Args:
            phase: Name of the research phase
        """
        self.set_phase(phase)
        start_time = datetime.datetime.now()
        self.log_metric("start_time", start_time.isoformat())
        
        try:
            yield
        except Exception as e:
            self.logger.error(f"Error in phase {phase}: {e}", exc_info=True)
            self.log_metric("error", str(e))
            self.log_metric("completed", False)
            raise
        finally:
            end_time = datetime.datetime.now()
            self.log_metric("end_time", end_time.isoformat())
            duration = (end_time - start_time).total_seconds()
            self.log_metric("duration_seconds", duration)
            
            if self.current_phase == phase:
                self.log_metric("completed", True)
    
    def log_experiment_summary(self) -> None:
        """Log a summary of the experiment."""
        summary = {
            "research_topic": self.research_topic,
            "timestamp": self.timestamp,
            "lab_index": self.lab_index,
            "experiment_dir": self.experiment_dir,
            "phases_completed": self.phases_completed,
            "metrics": self.metrics
        }
        
        summary_path = os.path.join(self.metadata_dir, "experiment_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Experiment summary saved: {summary_path}")
        
        # Also print summary to console
        self.logger.info(f"Research experiment completed: {self.research_topic}")
        self.logger.info(f"Phases completed: {', '.join(self.phases_completed)}")
        self.logger.info(f"All outputs saved to: {self.experiment_dir}")
    
    def close(self) -> None:
        """Close the logger and finalize the experiment."""
        # Log experiment summary
        self.log_experiment_summary()
        
        # Close all file handlers
        for handler in list(self.logger.handlers):
            if isinstance(handler, logging.FileHandler):
                handler.close()
                self.logger.removeHandler(handler)
        
        self.logger.info("Research logger closed") 