"""Core workflow module for the Agent Laboratory.

This module defines the LaboratoryWorkflow class that serves as the main entry point
for running experiments.
"""

import os
import json
import yaml
import logging
import datetime
import time
import re
import shutil
import subprocess
import tempfile
from typing import Dict, List, Any, Optional, Callable
import asyncio

from agent_lab.core.llm_interface import LLMInterface
from agent_lab.agents.professor_agent import ProfessorAgent
from agent_lab.agents.multi_agent import MultiAgentTeam
from agent_lab.io import execute_code, compile_latex, compile_latex_file, ArxivSearcher, SemanticScholarSearcher
from agent_lab.core.base_agent import BaseAgent
from agent_lab.visualization.pdf_generator import generate_report_pdf, generate_latex_document

logger = logging.getLogger(__name__)

class LaboratoryWorkflow:
    """Main workflow for the Agent Laboratory.
    
    This class manages the overall flow of the experiment, including setting up agents,
    running steps, and saving results.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the laboratory workflow.
        
        Args:
            config: Configuration dictionary for the experiment
        """
        self.config = config
        self.step_count = 0
        self.current_phase_idx = 0
        self.agents = {}
        
        # Define the phases to run in sequence
        self.phases = [
            "literature-review",
            "plan-formulation",
            "data-preparation",
            "code-implementation",
            "running-experiments",
            "results-interpretation",
            "report-writing"
        ]
        
        # Set up the output directory
        self.output_dir = self.initialize_output_directory()
        self.artifacts_dir = os.path.join(self.output_dir, "artifacts")
        self.artifacts_temp_dir = os.path.join(self.output_dir, "artifacts_temp")
        os.makedirs(self.artifacts_dir, exist_ok=True)
        os.makedirs(self.artifacts_temp_dir, exist_ok=True)
        
        # Set up logging
        self.log_file = self.setup_logging()
        
        # Get API key from config or environment
        api_key = config.get("api_key") or config.get("llm_config", {}).get("api_key")
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                logger.info("Using API key from environment variable OPENAI_API_KEY")
            
        # Get deepseek API key from config or environment
        deepseek_api_key = config.get("deepseek_api_key")
        if not deepseek_api_key:
            deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
            if deepseek_api_key:
                logger.info("Using deepseek API key from environment variable DEEPSEEK_API_KEY")
        
        # Initialize the LLM interface
        self.llm = LLMInterface(
            model=config.get("llm_backend", "gpt-4o-mini"),
            api_key=api_key,
            deepseek_api_key=deepseek_api_key
        )
        
        # Initialize agents
        self.initialize_agents()
        
        # Initialize searchers
        self.arxiv_searcher = ArxivSearcher()
        self.semantic_scholar_searcher = SemanticScholarSearcher()
        
        # Initialize state
        self.state = {
            "current_phase": None,
            "phases_completed": [],
            "start_time": time.time(),
            "artifacts": {}
        }
        
        # Try to load existing state if possible
        try:
            self.load_state()
        except Exception as e:
            logger.warning(f"Could not load state: {e}")
    
    def initialize_agents(self):
        """Initialize the agents for the workflow."""
        # Currently only initializing the professor agent
        # In a full implementation, this would initialize all required agents
        # based on the configuration
        self.agents["professor"] = self.create_agent("professor")
        
        # Initialize the multi-agent team
        self.agent_team = MultiAgentTeam(self.config, self.output_dir)
        
        # Also keep a reference to the professor agent for backward compatibility
        self.professor_agent = self.agent_team.agents.get("professor")
    
    def create_agent(self, agent_type: str):
        """Create an agent of the specified type.
        
        Args:
            agent_type: Type of agent to create
            
        Returns:
            The created agent instance
            
        Raises:
            ValueError: If the agent type is unknown
        """
        # Get API key from config or environment
        api_key = self.config.get("api_key") or self.config.get("llm_config", {}).get("api_key")
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
            
        # Create agent based on type
        if agent_type.lower() == "professor":
            return ProfessorAgent(
                api_key=api_key,
                config=self.config
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    def run(self):
        """Run the entire workflow."""
        # Ensure artifacts directory exists
        os.makedirs(self.artifacts_dir, exist_ok=True)
        os.makedirs(self.artifacts_temp_dir, exist_ok=True)
        
        # Check if we should do a literature review first
        num_papers = self.config.get("num_papers_lit_review", 0) or self.config.get("agentrxiv_papers", 0)
        if num_papers > 0:
            self.perform_literature_review(num_papers)
        
        # Run all phases
        while self.current_phase_idx < len(self.phases):
            try:
                phase_result = self.run_step()
                
                # If this is the code-implementation or running-experiments phase and execute-code is enabled
                current_phase = self.phases[self.current_phase_idx - 1]  # We've just completed this phase
                if current_phase in ["code-implementation", "running-experiments"] and self.config.get("execute_code", False):
                    self.execute_implementation_code(phase_result, current_phase)
            except Exception as e:
                logger.error(f"Error in phase {self.phases[self.current_phase_idx]}: {e}")
                logger.exception(e)
                if self.config.get("except_if_fail", True):
                    break
        
        # Generate the final report
        try:
            self.generate_report()
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            logger.exception(e)
        
        # Save agent discourse summary to file
        try:
            discourse_summary = self.agent_team.generate_discourse_summary()
            discourse_file = os.path.join(self.artifacts_dir, "agent_discourse.md")
            with open(discourse_file, "w") as f:
                f.write(discourse_summary)
            logger.info(f"Saved agent discourse summary to {discourse_file}")
        except Exception as e:
            logger.error(f"Error generating discourse summary: {e}")
            logger.exception(e)
        
        # Compile all LaTeX files to PDF if enabled
        if self.config.get("compile_latex", False):
            try:
                # Import here to avoid circular imports
                from agent_lab.visualization.pdf_generator import compile_latex_file
                
                # Find all LaTeX files in the output directory
                latex_files = [f for f in os.listdir(self.output_dir) if f.endswith('.tex')]
                compiled_files = []
                
                for latex_file in latex_files:
                    try:
                        latex_path = os.path.join(self.output_dir, latex_file)
                        pdf_path = compile_latex_file(latex_path)
                        
                        if pdf_path and os.path.exists(pdf_path):
                            compiled_files.append(pdf_path)
                            logger.info(f"Compiled LaTeX file to PDF: {pdf_path}")
                    except Exception as e:
                        logger.error(f"Error compiling {latex_file} to PDF: {e}")
                
                if not compiled_files:
                    logger.warning("No LaTeX files were successfully compiled to PDF")
                else:
                    logger.info(f"Successfully compiled {len(compiled_files)} LaTeX files to PDF")
            except Exception as e:
                logger.error(f"Error compiling LaTeX files to PDF: {e}")
                logger.exception(e)
        
        logger.info("Workflow completed successfully")
    
    def execute_implementation_code(self, phase_result: str, phase_name: str):
        """Execute the implementation code produced during a phase.
        
        Args:
            phase_result: The text output from the phase
            phase_name: The name of the phase that produced the code
        """
        try:
            # Extract code blocks from the phase result
            code_blocks = re.findall(r'```python(.*?)```', phase_result, re.DOTALL)
            if not code_blocks:
                logger.warning(f"No Python code blocks found in {phase_name} phase output")
                return
            
            # Combine code blocks and save to a temporary file
            combined_code = "\n\n".join(code.strip() for code in code_blocks)
            
            # Add code to save visualizations to the artifacts directory
            visualization_code = """
import os
import matplotlib.pyplot as plt

# Ensure artifacts directory exists
artifacts_dir = '""" + self.artifacts_dir.replace("'", "\\'") + """' 
os.makedirs(artifacts_dir, exist_ok=True)

# Override the show method to save figures to the artifacts directory
original_show = plt.show
def save_and_show(*args, **kwargs):
    # Save the current figure to the artifacts directory
    figname = f"figure_{len(os.listdir(artifacts_dir)) + 1}.png"
    figpath = os.path.join(artifacts_dir, figname)
    plt.savefig(figpath, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {figpath}")
    return original_show(*args, **kwargs)

plt.show = save_and_show
"""
            
            # Add the visualization code to the beginning of the code
            combined_code = visualization_code + "\n\n" + combined_code
            
            # Write the code to a file
            code_file = os.path.join(self.artifacts_temp_dir, f"{phase_name}_code.py")
            with open(code_file, "w") as f:
                f.write(combined_code)
            
            logger.info(f"Executing code from {phase_name} phase")
            result, _, _ = execute_code(combined_code, timeout=120)
            
            # Save the results to a file
            result_file = os.path.join(self.artifacts_dir, f"{phase_name}_code_execution_result.txt")
            with open(result_file, "w") as f:
                f.write(result)
            
            logger.info(f"Code execution result saved to {result_file}")
            
            # Also save the code itself to the artifacts directory
            code_artifact_file = os.path.join(self.artifacts_dir, f"{phase_name}_code.py")
            with open(code_artifact_file, "w") as f:
                f.write(combined_code)
            
            logger.info(f"Saved code to {code_artifact_file}")
            
            # If the code generated any visualizations, they should be in the artifacts directory now
            # due to the override of plt.show()
        except Exception as e:
            logger.error(f"Error executing code from {phase_name} phase: {e}")
            logger.exception(e)
    
    def perform_literature_review(self, num_papers=5):
        """Perform a literature review on the research topic.
        
        Args:
            num_papers: Number of papers to review
        """
        logger.info(f"Performing literature review with {num_papers} papers")
        
        try:
            # Create a prompt for literature review
            research_topic = self.config.get("research_topic", "")
            prompt = f"""
            Conduct a comprehensive literature review on the topic: {research_topic}
            
            Focus on reviewing {num_papers} key papers or preprints in this field. For each paper, provide:
            
            1. Title, authors, and publication venue/date
            2. Key research questions or objectives
            3. Methodological approach
            4. Main findings and contributions
            5. Limitations or gaps identified
            
            After reviewing the individual papers, provide:
            
            1. Synthesis of common themes and findings
            2. Identification of research gaps and opportunities
            3. Methodological best practices in this field
            4. How this literature review informs our research goal
            
            Format the review as a well-structured markdown document with clear sections and citations.
            """
            
            # Use the professor agent for the literature review
            professor_agent = self.agent_team.agents.get("professor")
            if not professor_agent:
                logger.warning("Professor agent not found, cannot perform literature review")
                return
            
            logger.info("Requesting literature review from professor agent")
            review_content = professor_agent.get_completion(
                prompt,
                system_message="You are an expert research professor conducting a literature review. Provide a comprehensive, detailed, and scholarly literature review that critically analyzes and synthesizes the relevant research."
            )
            
            # Save to file
            os.makedirs(self.artifacts_dir, exist_ok=True)
            lit_review_file = os.path.join(self.artifacts_dir, "literature_review.md")
            with open(lit_review_file, "w") as f:
                f.write(review_content)
            
            logger.info(f"Saved literature review to {lit_review_file}")
            
            # Add to professor's dialogue history
            professor_agent.dialogue_history.append({
                "role": "user",
                "content": prompt
            })
            professor_agent.dialogue_history.append({
                "role": "assistant",
                "content": review_content
            })
            
        except Exception as e:
            logger.error(f"Error performing literature review: {e}")
            logger.exception(e)
    
    def run_step(self):
        """Run a single step of the workflow."""
        if self.current_phase_idx >= len(self.phases):
            logger.info("All phases completed")
            return
        
        current_phase = self.phases[self.current_phase_idx]
        logger.info(f"Running phase: {current_phase}")
        
        task_notes = self.config.get("task_notes", {}).get(current_phase, "")
        
        # If agent discourse is enabled, use multi-agent team instead of just professor agent
        if self.config.get("enable_agent_discourse", True):
            result = self.agent_team.complete_phase_with_discourse(current_phase, task_notes)
            result_text = result["output"]
        else:
            # Legacy approach: use only professor agent
            result_text = self.professor_agent.complete_phase(current_phase, task_notes)
        
        # Update workflow state
        self.current_phase_idx += 1
        self.save_state()
        
        logger.info(f"Completed phase: {current_phase}")
        
        return result_text
    
    def load_state(self, state_file: Optional[str] = None) -> None:
        """Load the state of the workflow from a file.
        
        Args:
            state_file: Path to the state file (or None to use default path)
        """
        if state_file is None:
            state_file = os.path.join(self.output_dir, "workflow_state.json")
        
        if not os.path.exists(state_file):
            logger.warning(f"No state file found at {state_file}, starting from scratch")
            return
        
        try:
            with open(state_file, "r") as f:
                state = json.load(f)
            
            self.current_phase_idx = state.get("current_phase_idx", 0)
            
            logger.info(f"Loaded workflow state from {state_file}")
            logger.info(f"Current phase index: {self.current_phase_idx}")
            
            # Also load agent states
            self.agent_team.load_states()
            
        except Exception as e:
            logger.error(f"Error loading state from {state_file}: {e}")
            logger.exception(e)
    
    def save_state(self) -> None:
        """Save the state of the workflow to a file."""
        state_file = os.path.join(self.output_dir, "workflow_state.json")
        
        try:
            state = {
                "current_phase_idx": self.current_phase_idx,
                "phases": self.phases,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"Saved workflow state to {state_file}")
            
            # Also save agent states
            self.agent_team.save_states()
            
        except Exception as e:
            logger.error(f"Error saving state to {state_file}: {e}")
            logger.exception(e)
    
    def initialize_output_directory(self):
        """Initialize the output directory.
        
        Returns:
            str: Path to the output directory
        """
        output_dir = self.config.get("output_dir", "outputs")
        
        # Create a subdirectory with a timestamp and config filename if not specified
        if output_dir == "outputs":
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            topic_name = self.config.get("research_topic", "research")[:20].lower().replace(" ", "_")
            
            # Extract config filename if available
            config_file = self.config.get("config_file", "")
            if config_file:
                # Extract just the filename without path or extension
                config_name = os.path.splitext(os.path.basename(config_file))[0]
                output_dir = os.path.join(output_dir, f"{timestamp}_{config_name}")
            else:
                output_dir = os.path.join(output_dir, f"{timestamp}_{topic_name}")
        
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    def setup_logging(self):
        """Set up logging for the workflow.
        
        Returns:
            str: Path to the log file
        """
        # Create a log file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.output_dir, f"workflow_{timestamp}.log")
        
        # Configure logging - reset handlers first to avoid duplicate handlers
        # Remove all existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            
        # Configure logging with new handlers
        logging.basicConfig(
            level=logging.INFO if not self.config.get("debug", False) else logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        logging.info(f"Logging to {log_file}")
        return log_file
    
    def generate_report(self):
        """Generate a summary report of the workflow."""
        logger.info("Generating report")
        
        try:
            # Prepare experiment data
            experiment_data = {
                "research_topic": self.config.get("research_topic", ""),
                "start_time": datetime.datetime.fromtimestamp(self.state["start_time"]).strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": datetime.datetime.fromtimestamp(self.state.get("end_time", time.time())).strftime("%Y-%m-%d %H:%M:%S"),
                "duration": self.state.get("duration", 0),
                "phases_completed": self.state["phases_completed"]
            }
            
            # Get dialog history from all agents
            agent_dialogs = []
            for agent_name, agent in self.agent_team.agents.items():
                agent_dialogs.append({
                    "agent": agent_name,
                    "dialog": agent.get_dialog()  # Use get_dialog() method instead of direct attribute access
                })
            
            # Save the implementation code
            implementation_code = self._extract_code_from_running_experiments()
            if implementation_code:
                code_file = os.path.join(self.artifacts_dir, "implementation.py")
                with open(code_file, "w") as f:
                    f.write(implementation_code)
                logger.info(f"Saved implementation code to {code_file}")
            
            # Create a comprehensive LaTeX report
            try:
                # First prepare artifact data
                artifact_data = self._prepare_artifact_data()
                
                # Generate the LaTeX report
                latex_content = self._generate_latex_report(experiment_data, agent_dialogs, artifact_data)
                
                # Save the LaTeX report
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                latex_file = os.path.join(self.output_dir, f"experiment_report_{timestamp}.tex")
                with open(latex_file, "w") as f:
                    f.write(latex_content)
                
                logger.info(f"Saved LaTeX report to {latex_file}")
                
                # Copy the LaTeX file to artifacts.tex for consistency
                artifacts_tex = os.path.join(self.output_dir, "artifacts.tex")
                shutil.copy(latex_file, artifacts_tex)
                logger.info(f"Copied LaTeX report to {artifacts_tex}")
                
                # Compile the LaTeX report to PDF if enabled
                if self.config.get("compile_latex", False):
                    try:
                        # Use the improved generate_report_pdf function from visualization.pdf_generator
                        from agent_lab.visualization.pdf_generator import generate_report_pdf
                        
                        # Get the output file path without extension
                        output_file = os.path.splitext(latex_file)[0]
                        
                        # Generate PDF from the experiment data and agent dialogs
                        pdf_path = generate_report_pdf(
                            experiment_data,
                            agent_dialogs,
                            output_file,
                            self.artifacts_dir
                        )
                        
                        if pdf_path and os.path.exists(pdf_path):
                            logger.info(f"Generated PDF report: {pdf_path}")
                        else:
                            # Fall back to direct compilation if PDF generation fails
                            from agent_lab.visualization.pdf_generator import compile_latex_file
                            pdf_path = compile_latex_file(latex_file)
                            logger.info(f"Compiled LaTeX report to PDF using fallback method: {pdf_path}")
                    except Exception as e:
                        logger.error(f"Error compiling LaTeX report to PDF: {e}")
                        logger.exception(e)
                
            except Exception as e:
                logger.error(f"Error generating LaTeX report: {e}")
                logger.exception(e)
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            logger.exception(e)
    
    def _extract_code_from_running_experiments(self):
        """Extract the code from the running-experiments phase.
        
        Returns:
            str: The extracted code
        """
        try:
            # Find the interaction file for the running-experiments phase
            interaction_file = os.path.join(self.output_dir, "running-experiments_interaction.json")
            if not os.path.exists(interaction_file):
                logger.warning(f"No interaction file found at {interaction_file}, cannot extract code")
                return ""
            
            with open(interaction_file, "r") as f:
                interaction = json.load(f)
            
            # Get the integrated output
            output = interaction.get("integrated_output", "")
            
            # Extract all Python code blocks
            code_blocks = re.findall(r'```python(.*?)```', output, re.DOTALL)
            if not code_blocks:
                logger.warning("No Python code blocks found in running-experiments phase output")
                return ""
            
            # Combine code blocks
            implementation_code = "\n\n".join(code.strip() for code in code_blocks)
            return implementation_code
            
        except Exception as e:
            logger.error(f"Error extracting code from running-experiments phase: {e}")
            logger.exception(e)
            return ""
    
    def _prepare_artifact_data(self):
        """Prepare artifact data for the report.
        
        Returns:
            Dict[str, Any]: Artifact data for the report
        """
        artifact_data = {
            "figures": [],
            "code_files": [],
            "results": []
        }
        
        try:
            # Get all figures from the artifacts directory
            for file in os.listdir(self.artifacts_dir):
                if file.endswith((".png", ".jpg", ".jpeg")):
                    figure_path = os.path.join(self.artifacts_dir, file)
                    artifact_data["figures"].append({
                        "path": figure_path,
                        "caption": f"Figure: {file}",
                        "filename": file
                    })
            
            # Get all code files from the artifacts directory
            for file in os.listdir(self.artifacts_dir):
                if file.endswith(".py"):
                    code_path = os.path.join(self.artifacts_dir, file)
                    with open(code_path, "r") as f:
                        code_content = f.read()
                    artifact_data["code_files"].append({
                        "path": code_path,
                        "content": code_content,
                        "filename": file
                    })
            
            # Get all result files from the artifacts directory
            for file in os.listdir(self.artifacts_dir):
                if file.endswith((".txt", ".json")):
                    result_path = os.path.join(self.artifacts_dir, file)
                    with open(result_path, "r") as f:
                        result_content = f.read()
                    artifact_data["results"].append({
                        "path": result_path,
                        "content": result_content,
                        "filename": file
                    })
        
        except Exception as e:
            logger.error(f"Error preparing artifact data: {e}")
            logger.exception(e)
        
        return artifact_data
    
    def _generate_latex_report(self, experiment_data, agent_dialogs, artifact_data):
        """Generate the LaTeX report.
        
        Args:
            experiment_data: Experiment data
            agent_dialogs: Agent dialog history
            artifact_data: Artifact data
            
        Returns:
            str: LaTeX report content
        """
        # Get agent discourse data
        discourse_data = []
        for phase in self.phases:
            interaction_file = os.path.join(self.output_dir, f"{phase}_interaction.json")
            if os.path.exists(interaction_file):
                with open(interaction_file, "r") as f:
                    interaction = json.load(f)
                discourse_data.append({
                    "phase": phase,
                    "discourse": interaction
                })
        
        # Get the research topic
        research_topic = self.config.get("research_topic", "")
        
        # Generate the LaTeX report using the PDF generator
        latex_content = generate_latex_document(
            research_topic=research_topic,
            discourse_data=discourse_data,
            artifact_data=artifact_data
        )
        
        return latex_content

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "LaboratoryWorkflow":
        """Create a workflow from a YAML configuration file.
        
        Args:
            yaml_path: Path to the YAML configuration file
            
        Returns:
            LaboratoryWorkflow: A new workflow instance
        """
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return cls(config)
    
    @classmethod
    def from_args(cls, args: Any) -> "LaboratoryWorkflow":
        """Create a workflow from command-line arguments.
        
        Args:
            args: Command-line arguments
            
        Returns:
            LaboratoryWorkflow: A new workflow instance
        """
        # Extract configuration from arguments
        config = {
            "research_topic": getattr(args, "research_topic", None),
            "llm_backend": getattr(args, "llm_backend", "gpt-4o-mini"),
            "api_key": getattr(args, "api_key", None),
            "deepseek_api_key": getattr(args, "deepseek_api_key", None),
            "copilot_mode": getattr(args, "copilot_mode", False),
            "output_dir": getattr(args, "output_dir", "outputs"),
            "max_steps": getattr(args, "max_steps", 100),
            "compile_latex": getattr(args, "compile_latex", True),
            "lab_index": getattr(args, "lab_index", 1)
        }
        
        # Load YAML configuration if provided
        yaml_path = getattr(args, "yaml_location", None)
        if yaml_path:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
                config.update(yaml_config)
        
        return cls(config)

    def run_workflow(self) -> Dict[str, Any]:
        """Run the complete workflow.
        
        Returns:
            Results of the workflow
        """
        logger.info("Starting workflow")
        
        # Run each phase
        results = {}
        
        # Literature review
        if "literature-review" not in self.state["phases_completed"]:
            results["literature-review"] = self.run_literature_review()
        
        # Plan formulation
        if "plan-formulation" not in self.state["phases_completed"]:
            results["plan-formulation"] = self.run_plan_formulation()
        
        # Data preparation
        if "data-preparation" not in self.state["phases_completed"]:
            results["data-preparation"] = self.run_data_preparation()
        
        # Code implementation
        if "code-implementation" not in self.state["phases_completed"]:
            results["code-implementation"] = self.run_code_implementation()
        
        # Running experiments
        if "running-experiments" not in self.state["phases_completed"]:
            results["running-experiments"] = self.run_experiments()
        
        # Results interpretation
        if "results-interpretation" not in self.state["phases_completed"]:
            results["results-interpretation"] = self.run_results_interpretation()
        
        # Report writing
        if "report-writing" not in self.state["phases_completed"]:
            results["report-writing"] = self.run_report_writing()
        
        # Update end time
        self.state["end_time"] = time.time()
        self.state["duration"] = self.state["end_time"] - self.state["start_time"]
        self.state["current_phase"] = None
        self._save_state()
        
        logger.info(f"Workflow completed in {self.state['duration']:.2f} seconds")
        return results

    def run_literature_review(self) -> Dict[str, Any]:
        """Run the literature review phase.
        
        Returns:
            Results of the literature review
        """
        logger.info("Starting literature review")
        self.state["current_phase"] = "literature-review"
        
        # Get the research topic from the config
        research_topic = self.config.get("research_topic", "")
        if not research_topic:
            logger.warning("No research topic specified, skipping literature review")
            return {}
        
        # Create specific search terms for POMDP and Active Inference
        search_terms = [
            research_topic,
            "POMDP Active Inference",
            "Partially Observable Markov Decision Process",
            "Active Inference thermal control",
            "Variational Free Energy POMDP",
            "Expected Free Energy action selection"
        ]
        
        all_arxiv_papers = []
        all_ss_papers = []
        
        # Search for papers on arXiv and Semantic Scholar using multiple search terms
        for term in search_terms:
            logger.info(f"Searching for papers on: {term}")
            
            # Search for papers on arXiv
            arxiv_papers = self.arxiv_searcher.search(term, max_results=3)
            all_arxiv_papers.extend(arxiv_papers)
            
            # Search for papers on Semantic Scholar
            ss_papers = self.semantic_scholar_searcher.search(term, max_results=3)
            all_ss_papers.extend(ss_papers)
        
        # Remove duplicate papers
        unique_arxiv_papers = []
        unique_arxiv_ids = set()
        for paper in all_arxiv_papers:
            paper_id = paper.get("id", "")
            if paper_id and paper_id not in unique_arxiv_ids:
                unique_arxiv_ids.add(paper_id)
                unique_arxiv_papers.append(paper)
        
        unique_ss_papers = []
        unique_ss_ids = set()
        for paper in all_ss_papers:
            paper_id = paper.get("paperId", "")
            if paper_id and paper_id not in unique_ss_ids:
                unique_ss_ids.add(paper_id)
                unique_ss_papers.append(paper)
        
        # Format papers for display
        arxiv_formatted = self.arxiv_searcher.format_papers(unique_arxiv_papers[:5])
        ss_formatted = self.semantic_scholar_searcher.format_papers(unique_ss_papers[:5])
        
        # Combine the results
        literature = {
            "arxiv_papers": unique_arxiv_papers[:5],
            "semantic_scholar_papers": unique_ss_papers[:5]
        }
        
        # Save the literature review
        literature_content = f"# Literature Review: {research_topic}\n\n"
        literature_content += "## arXiv Papers\n\n" + arxiv_formatted + "\n\n"
        literature_content += "## Semantic Scholar Papers\n\n" + ss_formatted
        
        self._save_artifact("literature_review", literature_content, "text")
        
        # Ask the professor agent to analyze the literature with specific focus on POMDP and Active Inference
        logger.info("Analyzing literature with professor agent")
        
        # Provide specific instructions for literature analysis
        analysis_instructions = f"""
        Analyze the provided papers with a specific focus on:
        1. Mathematical formulations of POMDPs and Active Inference
        2. Applications of Active Inference in control systems
        3. Integration of Variational Free Energy for state estimation
        4. Expected Free Energy methods for action selection
        5. Any thermal control examples or similar applications
        
        Highlight specific mathematical formulations, algorithms, and implementation approaches that 
        would be relevant for implementing a POMDP with 3 control states, 5 latent states, and 10 discrete 
        observation levels as specified in the research topic.
        """
        
        literature_analysis = self.professor_agent.analyze_literature(
            unique_arxiv_papers[:5], unique_ss_papers[:5], 
            research_topic + "\n\n" + analysis_instructions
        )
        
        # Save the literature analysis
        self._save_artifact("literature_analysis", literature_analysis, "text")
        
        # Update the state
        self.state["phases_completed"].append("literature-review")
        self._save_state()
        
        return literature

    def run_plan_formulation(self) -> Dict[str, Any]:
        """Run the plan formulation phase.
        
        Returns:
            Results of the plan formulation
        """
        logger.info("Starting plan formulation")
        self.state["current_phase"] = "plan-formulation"
        
        # Get the task notes from the config
        task_notes = self.config.get("task_notes", {}).get("plan_formulation", "")
        if not task_notes:
            logger.warning("No task notes for plan formulation, using default")
            task_notes = "Develop a comprehensive research plan for the experiment."
        
        # Get the research topic from the config
        research_topic = self.config.get("research_topic", "")
        
        # Ask the professor agent to formulate a plan
        logger.info("Formulating plan with professor agent")
        plan = self.professor_agent.complete_phase("plan-formulation", task_notes)
        
        # Save the plan
        self._save_artifact("research_plan", plan, "text")
        
        # Update the state
        self.state["phases_completed"].append("plan-formulation")
        self._save_state()
        
        return {"plan": plan}

    def run_data_preparation(self) -> Dict[str, Any]:
        """Run the data preparation phase.
        
        Returns:
            Results of the data preparation
        """
        logger.info("Starting data preparation")
        self.state["current_phase"] = "data-preparation"
        
        # Get the task notes from the config
        task_notes = self.config.get("task_notes", {}).get("data_preparation", "")
        if not task_notes:
            logger.warning("No task notes for data preparation, using default")
            task_notes = "Describe the data preparation process for the experiment."
        
        # Ask the professor agent to prepare the data
        logger.info("Preparing data with professor agent")
        data_preparation = self.professor_agent.complete_phase("data-preparation", task_notes)
        
        # Save the data preparation
        self._save_artifact("data_preparation", data_preparation, "text")
        
        # Update the state
        self.state["phases_completed"].append("data-preparation")
        self._save_state()
        
        return {"data_preparation": data_preparation}

    def run_code_implementation(self) -> Dict[str, Any]:
        """Run the code implementation phase.
        
        Returns:
            Results of the code implementation
        """
        logger.info("Starting code implementation")
        self.state["current_phase"] = "code-implementation"
        
        # Get the task notes from the config
        task_notes = self.config.get("task_notes", {}).get("code_implementation", "")
        if not task_notes:
            logger.warning("No task notes for code implementation, using default")
            task_notes = "Implement the code for the experiment."
        
        # Ask the professor agent to implement the code
        logger.info("Implementing code with professor agent")
        code_implementation = self.professor_agent.complete_phase("code-implementation", task_notes)
        
        # Save the code implementation
        code_file = self._save_artifact("experiment_code", code_implementation, "code")
        
        # Execute the code
        logger.info("Executing code")
        execution_result = execute_code(code_implementation)
        
        # Save the execution result
        self._save_artifact("code_execution_result", execution_result, "text")
        
        # Update the state
        self.state["phases_completed"].append("code-implementation")
        self._save_state()
        
        return {
            "code_implementation": code_implementation,
            "execution_result": execution_result
        }

    def run_experiments(self) -> Dict[str, Any]:
        """Run the experiments phase.
        
        Returns:
            Results of the experiments
        """
        logger.info("Starting experiments")
        self.state["current_phase"] = "running-experiments"
        
        # Get the task notes from the config
        task_notes = self.config.get("task_notes", {}).get("running_experiments", "")
        if not task_notes:
            logger.warning("No task notes for running experiments, using default")
            task_notes = "Run the experiments and collect results."
        
        # Ask the professor agent to run the experiments
        logger.info("Running experiments with professor agent")
        experiments = self.professor_agent.complete_phase("running-experiments", task_notes)
        
        # Save the experiments
        self._save_artifact("experiments", experiments, "text")
        
        # Update the state
        self.state["phases_completed"].append("running-experiments")
        self._save_state()
        
        return {"experiments": experiments}

    def run_results_interpretation(self) -> Dict[str, Any]:
        """Run the results interpretation phase.
        
        Returns:
            Results of the interpretation
        """
        logger.info("Starting results interpretation")
        self.state["current_phase"] = "results-interpretation"
        
        # Get the task notes from the config
        task_notes = self.config.get("task_notes", {}).get("results_interpretation", "")
        if not task_notes:
            logger.warning("No task notes for results interpretation, using default")
            task_notes = "Interpret the experimental results."
        
        # Ask the professor agent to interpret the results
        logger.info("Interpreting results with professor agent")
        interpretation = self.professor_agent.complete_phase("results-interpretation", task_notes)
        
        # Save the interpretation
        self._save_artifact("results_interpretation", interpretation, "text")
        
        # Update the state
        self.state["phases_completed"].append("results-interpretation")
        self._save_state()
        
        return {"interpretation": interpretation}

    def run_report_writing(self) -> Dict[str, Any]:
        """Run the report writing phase.
        
        Returns:
            Results of the report writing
        """
        logger.info("Starting report writing")
        self.state["current_phase"] = "report-writing"
        
        # Get the task notes from the config
        task_notes = self.config.get("task_notes", {}).get("report_writing", "")
        if not task_notes:
            logger.warning("No task notes for report writing, using default")
            task_notes = "Write a comprehensive report on the research findings."
        
        # Ask the professor agent to write the report
        logger.info("Writing report with professor agent")
        report = self.professor_agent.complete_phase("report-writing", task_notes)
        
        # Save the report content
        self._save_artifact("research_report", report, "text")
        
        # Create a temporary directory for LaTeX compilation
        import tempfile
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Generate the LaTeX document
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            latex_file = generate_latex_document(
                "POMDP with Active Inference for Thermal Homeostasis",
                datetime.datetime.now().strftime("%Y-%m-%d"),
                report,
                temp_dir
            )
            
            # Copy the LaTeX file to the output directory
            output_latex_file = os.path.join(self.output_dir, f"research_report_{timestamp}.tex")
            shutil.copy(latex_file, output_latex_file)
            
            # Compile the LaTeX report to PDF
            logger.info("Compiling report to PDF")
            pdf_success, pdf_message = compile_latex(
                report,
                self.output_dir,
                f"research_report_{timestamp}",
                timeout=120
            )
            
            if pdf_success:
                logger.info(f"Successfully compiled report to PDF: {pdf_message}")
            else:
                logger.error(f"Failed to compile report to PDF: {pdf_message}")
                
                # Try compiling with our explicit LaTeX file
                pdf_success, pdf_message = compile_latex_file(
                    output_latex_file,
                    timeout=120
                )
                
                if pdf_success:
                    logger.info(f"Successfully compiled report to PDF on second attempt: {pdf_message}")
                else:
                    logger.error(f"Failed to compile report to PDF on second attempt: {pdf_message}")
        
        except Exception as e:
            logger.error(f"Error generating or compiling LaTeX report: {e}")
            logger.exception(e)
            pdf_success = False
            pdf_message = str(e)
        finally:
            # Clean up the temporary directory
            shutil.rmtree(temp_dir)
        
        # Update the state
        self.state["phases_completed"].append("report-writing")
        self._save_state()
        
        return {
            "report": report,
            "pdf_success": pdf_success,
            "pdf_message": pdf_message
        }

    def _save_artifact(self, name: str, content: str, artifact_type: str = "text"):
        """Save an artifact to the output directory.
        
        Args:
            name: Name of the artifact
            content: Content of the artifact
            artifact_type: Type of the artifact (text, code, figure, etc.)
        """
        # Determine the appropriate directory
        if artifact_type == "code":
            directory = os.path.join(self.output_dir, "code")
            extension = ".py"
        elif artifact_type == "figure":
            directory = os.path.join(self.output_dir, "figures")
            extension = ".png"
        elif artifact_type == "latex":
            directory = os.path.join(self.output_dir, "latex")
            os.makedirs(directory, exist_ok=True)
            extension = ".tex"
        elif artifact_type == "paper":
            directory = os.path.join(self.output_dir, "papers")
            extension = ".json"
        else:
            directory = self.output_dir
            extension = ".txt"
        
        # Save the artifact
        filename = os.path.join(directory, f"{name}{extension}")
        with open(filename, 'w') as f:
            f.write(content)
        
        # Update the state
        self.state["artifacts"][name] = {
            "path": filename,
            "type": artifact_type,
            "timestamp": time.time()
        }
        
        # Save the state
        self._save_state()
        
        return filename

    def _save_state(self):
        """Save the current state of the workflow."""
        state_file = os.path.join(self.output_dir, "workflow_state.json")
        with open(state_file, 'w') as f:
            json.dump(self.state, f, indent=2) 