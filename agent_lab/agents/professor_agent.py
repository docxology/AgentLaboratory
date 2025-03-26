"""
Professor agent for the Agent Laboratory.

This module defines the ProfessorAgent class that implements a professor-like agent
for directing and evaluating research.
"""

from typing import Dict, Any, Optional, List
import os
import logging
import time

from agent_lab.core.base_agent import BaseAgent
from agent_lab.core.llm_interface import LLMInterface

logger = logging.getLogger(__name__)

class ProfessorAgent(BaseAgent):
    """Professor agent for directing research and synthesizing knowledge.
    
    This agent acts as the lead researcher, directing the research process,
    synthesizing information, and coordinating other agents.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the professor agent.
        
        Args:
            api_key: API key for the LLM service
            config: Configuration for the agent
        """
        # Extract parameters from config to match BaseAgent's expected parameters
        model = config.get("llm_backend", "gpt-4o-mini") if config else "gpt-4o-mini"
        notes = config.get("task_notes", {}) if config else {}
        max_steps = config.get("max_steps", 10) if config else 10
        
        # Initialize base agent with the extracted parameters
        super().__init__(
            model=model,
            notes=notes,
            max_steps=max_steps,
            api_key=api_key,
            logger=logger
        )
        
        # Store the full config for later use
        self.config = config or {}
        
        # Set agent-specific properties
        self.research_area = config.get("research_area", "artificial intelligence") if config else "artificial intelligence"
        self.expertise = config.get("expertise", ["machine learning", "computer science"]) if config else ["machine learning", "computer science"]
        
        # Initialize the agent's state
        self.state.update({
            "research_plan": None,
            "literature_review": None,
            "current_findings": [],
            "research_questions": []
        })
        
        logger.info(f"Initialized professor agent: {self.research_area}")
    
    def analyze_literature(self, arxiv_papers: List[Dict[str, str]], 
                           semantic_scholar_papers: List[Dict[str, Any]], 
                           research_topic: str) -> str:
        """Analyze literature for a research topic.
        
        Args:
            arxiv_papers: List of papers from arXiv
            semantic_scholar_papers: List of papers from Semantic Scholar
            research_topic: Research topic to analyze
            
        Returns:
            Analysis of the literature
        """
        logger.info(f"Professor agent analyzing literature for: {research_topic}")
        
        # Call the base method to get the analysis
        analysis = super().analyze_literature(arxiv_papers, semantic_scholar_papers, research_topic)
        
        # Update the agent's state
        self.state["literature_review"] = analysis
        
        # Extract research questions from the literature
        research_questions = self._extract_research_questions(analysis)
        self.state["research_questions"] = research_questions
        
        return analysis
    
    def _extract_research_questions(self, literature_analysis: str) -> List[str]:
        """Extract research questions from literature analysis.
        
        Args:
            literature_analysis: Literature analysis text
            
        Returns:
            List of research questions
        """
        if not self.llm:
            logger.error("LLM interface not initialized")
            return []
        
        prompt = f"""Based on the following literature analysis, identify and list the top 5 research questions that could be addressed in this field:

{literature_analysis}

FORMAT: Provide exactly 5 research questions, one per line, beginning with "Research Question:"
"""
        
        # Call the LLM
        output = self.llm.generate(prompt)
        
        # Parse the output
        questions = []
        for line in output.split("\n"):
            if line.startswith("Research Question:"):
                questions.append(line.replace("Research Question:", "").strip())
        
        return questions
    
    def complete_phase(self, phase_name: str, task_notes: str) -> str:
        """Complete a phase of the research workflow.
        
        Args:
            phase_name: Name of the phase to complete
            task_notes: Task notes for the phase
            
        Returns:
            Output from completing the phase
        """
        logger.info(f"Professor agent completing phase: {phase_name}")
        
        # Add additional context based on the phase
        context = ""
        if phase_name == "plan-formulation":
            if self.state.get("literature_review"):
                context = f"\nLiterature Review:\n{self.state['literature_review'][:1000]}...\n"
                context += f"\nResearch Questions:\n" + "\n".join([f"- {q}" for q in self.state.get("research_questions", [])])
        
        elif phase_name == "data-preparation":
            if self.state.get("research_plan"):
                context = f"\nResearch Plan:\n{self.state['research_plan'][:1000]}...\n"
            
            # Add specific guidance for POMDP implementation
            context += "\n\nFor the POMDP with Active Inference implementation, please specify:\n"
            context += "1. The formal mathematical model for a 5-state POMDP with 3 control actions and 10 observation levels\n"
            context += "2. How to calculate Variational Free Energy (VFE) for state estimation\n"
            context += "3. How to calculate Expected Free Energy (EFE) for action selection\n"
            context += "4. The thermal homeostasis dynamics (how temperature changes given states and actions)\n"
            context += "5. The parameters needed for the model (e.g., transition matrices, observation matrices)\n"
        
        elif phase_name == "code-implementation":
            if self.state.get("research_plan"):
                context = f"\nResearch Plan:\n{self.state['research_plan'][:1000]}...\n"
            if self.state.get("data_preparation"):
                context += f"\nData Preparation:\n{self.state['data_preparation'][:1000]}...\n"
            
            # Add detailed guidance for POMDP code implementation
            context += "\n\nIMPORTANT GUIDELINES FOR CODE IMPLEMENTATION:\n"
            context += "1. Implement a complete POMDP with Active Inference for thermal homeostasis\n"
            context += "2. The implementation must include:\n"
            context += "   - A generative model with 5 latent temperature states\n"
            context += "   - 3 control actions (cool, nothing, heat)\n"
            context += "   - 10 discrete observation levels (cold to hot)\n"
            context += "   - Functions to compute Variational Free Energy for state estimation\n"
            context += "   - Functions to compute Expected Free Energy for action selection\n"
            context += "3. Use numpy for matrix operations and matplotlib for visualizations\n"
            context += "4. Include clear docstrings and comments throughout the code\n"
            context += "5. Make the code modular with well-defined functions and classes\n"
            context += "6. Create a main function to demonstrate the model's behavior\n"
            context += "7. Use triple backticks to format code blocks: ```python\n# Your code here\n```\n"
        
        elif phase_name == "running-experiments":
            if self.state.get("code_implementation"):
                context = f"\nCode Implementation:\n{self.state['code_implementation'][:1000]}...\n"
            
            # Add specific guidance for POMDP experiments
            context += "\n\nIMPORTANT GUIDELINES FOR RUNNING EXPERIMENTS:\n"
            context += "1. Run simulations with the POMDP Active Inference model to demonstrate thermal homeostasis\n"
            context += "2. Experiments should include:\n"
            context += "   - Tracking belief states over time\n"
            context += "   - Showing how actions are selected based on Expected Free Energy\n"
            context += "   - Demonstrating the system's ability to maintain temperature around a target value\n"
            context += "   - Comparing performance with different parameters or conditions\n"
            context += "3. Generate clear visualizations of the results (state beliefs, actions, observations, etc.)\n"
            context += "4. Present quantitative metrics to evaluate performance (e.g., average distance from target state)\n"
            context += "5. Ensure all code can run efficiently with the provided time constraints\n"
            context += "6. Use triple backticks to format code blocks: ```python\n# Your code here\n```\n"
        
        elif phase_name == "results-interpretation":
            if self.state.get("experiments"):
                context = f"\nExperiment Results:\n{self.state['experiments'][:1000]}...\n"
            
            # Add guidance for interpreting POMDP results
            context += "\n\nGUIDELINES FOR INTERPRETING RESULTS:\n"
            context += "1. Analyze how well the system maintains temperature around the target state\n"
            context += "2. Discuss the effectiveness of using Active Inference (VFE and EFE) for control\n"
            context += "3. Compare with potential alternative approaches (e.g., PID control or MPC)\n"
            context += "4. Identify strengths and limitations of the implementation\n"
            context += "5. Discuss computational efficiency and scalability considerations\n"
            context += "6. Suggest potential improvements or extensions to the model\n"
        
        elif phase_name == "report-writing":
            if self.state.get("results_interpretation"):
                context = f"\nResults Interpretation:\n{self.state['results_interpretation'][:1000]}...\n"
            
            # Add guidance for comprehensive report writing
            context += "\n\nGUIDELINES FOR REPORT WRITING:\n"
            context += "1. Write a comprehensive paper with standard scientific structure\n"
            context += "2. Include clear mathematical formulations of Variational Free Energy and Expected Free Energy\n"
            context += "3. Present the POMDP model with precise notation for states, actions, and observations\n"
            context += "4. Include pseudocode and/or mathematical algorithms for key components\n"
            context += "5. Present experimental results with clear visualizations\n"
            context += "6. Discuss implications for Active Inference theory and practical applications\n"
            context += "7. Conclude with limitations and future research directions\n"
        
        # Call the base method with additional context
        detailed_task_notes = task_notes + context
        output = super().complete_phase(phase_name, detailed_task_notes)
        
        # Post-process the output for running-experiments phase to ensure proper code formatting
        if phase_name == "running-experiments":
            output = self._ensure_code_formatting(output)
        
        # Update the agent's state based on the phase
        if phase_name == "plan-formulation":
            self.state["research_plan"] = output
        elif phase_name == "data-preparation":
            self.state["data_preparation"] = output
        elif phase_name == "code-implementation":
            self.state["code_implementation"] = output
        elif phase_name == "running-experiments":
            self.state["experiments"] = output
        elif phase_name == "results-interpretation":
            self.state["results_interpretation"] = output
        elif phase_name == "report-writing":
            self.state["report"] = output
        
        # Save the output to disk for tracking
        self._save_phase_output(phase_name, output)
        
        return output
    
    def _ensure_code_formatting(self, text: str) -> str:
        """Ensure code blocks are properly formatted with triple backticks.
        
        Args:
            text: Text possibly containing code blocks
            
        Returns:
            Text with properly formatted code blocks
        """
        import re
        
        # Check if there are already properly formatted code blocks
        if "```python" in text:
            return text
            
        # Look for potential code blocks (indented code, or code without proper formatting)
        code_pattern = r'((?:^[ \t]*import\s+[\w\.]+|^[ \t]*from\s+[\w\.]+\s+import|\s*class\s+\w+|\s*def\s+\w+)(?:\s*\()?(?:\s*.*\)?)?:(?:\s*$)(?:[\s\S]*?)(?:(?:^[ \t]*$)|(?:^[ \t]*[^\s#])|\Z))'
        
        # Helper function to process matched code blocks
        def format_code_block(match):
            code = match.group(1).strip()
            if not code:
                return match.group(0)
            return f"\n```python\n{code}\n```\n"
            
        # Replace potential code blocks with properly formatted ones
        formatted_text = re.sub(code_pattern, format_code_block, text, flags=re.MULTILINE)
        
        return formatted_text
    
    def _save_phase_output(self, phase_name: str, content: str) -> None:
        """Save the phase output to disk for tracking.
        
        Args:
            phase_name: Name of the phase
            content: Content to save
        """
        # Create outputs directory if it doesn't exist
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = os.environ.get("AGENT_OUTPUT_DIR", "outputs/professor_outputs")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a filename with phase name and timestamp
        filename = f"{output_dir}/{phase_name}_{timestamp}.md"
        
        # Write the content to the file
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"# Phase: {phase_name}\n\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Content length: {len(content)} characters\n")
            f.write(f"Word count: {len(content.split())} words\n\n")
            f.write("---\n\n")
            f.write(content)
        
        logger.info(f"Saved {phase_name} output to {filename} [{len(content)} chars, {len(content.split())} words]")
    
    def role_description(self) -> str:
        """Get the role description for the agent.
        
        Returns:
            str: The role description
        """
        return (
            "knowledgeable professor and scientific advisor who is an expert in the "
            "field. You are guiding a research project, helping to organize the research, "
            "providing expert insights, and evaluating the quality of work."
        )
    
    def command_descriptions(self, phase: str) -> str:
        """Get the command descriptions for a specific phase.
        
        Args:
            phase: The current phase
            
        Returns:
            str: The command descriptions
        """
        if phase == "plan-formulation":
            return (
                "You are to help develop a comprehensive research plan. Your task is to:\n"
                "1. Analyze the research topic and break it down into manageable components\n"
                "2. Identify the key theoretical concepts that need to be incorporated\n"
                "3. Outline a step-by-step approach to implementation\n"
                "4. Suggest appropriate methodologies and evaluation metrics\n"
                "5. Identify potential challenges and propose solutions"
            )
        elif phase == "data-preparation":
            return (
                "You are to help with data preparation. Your task is to:\n"
                "1. Define appropriate data structures for the problem\n"
                "2. Design generators or loaders for synthetic or real data\n"
                "3. Suggest preprocessing steps and transformations\n"
                "4. Ensure data quality and address potential issues\n"
                "5. Provide guidance on appropriate testing data"
            )
        elif phase == "running-experiments":
            return (
                "You are to help with experiment execution. Your task is to:\n"
                "1. Design a series of experiments to test the implementation\n"
                "2. Provide guidance on parameter settings and configurations\n"
                "3. Suggest appropriate metrics for evaluation\n"
                "4. Help interpret intermediate results\n"
                "5. Recommend adjustments to improve performance"
            )
        elif phase == "results-interpretation":
            return (
                "You are to help with interpreting results. Your task is to:\n"
                "1. Analyze the experimental outcomes\n"
                "2. Compare results with theoretical expectations\n"
                "3. Identify strengths and limitations of the approach\n"
                "4. Draw meaningful conclusions\n"
                "5. Suggest future research directions"
            )
        elif phase == "report-writing":
            return (
                "You are to help with research documentation. Your task is to:\n"
                "1. Structure the research report\n"
                "2. Ensure clarity and precision in the presentation\n"
                "3. Suggest improvements to explanations and visualizations\n"
                "4. Maintain scientific rigor throughout the document\n"
                "5. Provide feedback on the overall quality and completeness"
            )
        else:
            return (
                "You are a scientific advisor providing expert guidance and feedback "
                "based on your extensive knowledge in the field."
            )
    
    def example_command(self, phase: str) -> str:
        """Get an example command for a specific phase.
        
        Args:
            phase: The current phase
            
        Returns:
            str: The example command
        """
        if phase == "plan-formulation":
            return (
                "Example output for plan formulation:\n\n"
                "# Research Plan: POMDP with Active Inference\n\n"
                "## 1. Theoretical Foundation\n"
                "- Define the mathematical framework for POMDP\n"
                "- Formulate Active Inference principles\n"
                "- Develop equations for Variational Free Energy and Expected Free Energy\n\n"
                "## 2. Implementation Approach\n"
                "- Design the thermal homeostasis system with specified parameters\n"
                "- Implement belief updating using VFE\n"
                "- Implement action selection using EFE\n\n"
                "## 3. Evaluation Strategy\n"
                "- Define metrics for assessing performance\n"
                "- Create test scenarios with different environmental conditions\n"
                "- Compare against baseline approaches\n\n"
                "## 4. Potential Challenges\n"
                "- Computational efficiency of belief updating\n"
                "- Numerical stability in probability calculations\n"
                "- Parameter tuning for optimal performance"
            )
        elif phase == "running-experiments":
            return (
                "Example output for running experiments:\n\n"
                "# Experiment Implementation\n\n"
                "```python\n"
                "import numpy as np\n"
                "import matplotlib.pyplot as plt\n\n"
                "class ThermalPOMDP:\n"
                "    def __init__(self, num_temp_states=5, num_obs_states=10):\n"
                "        # Initialize model parameters\n"
                "        self.num_temp_states = num_temp_states\n"
                "        self.num_obs_states = num_obs_states\n"
                "        self.num_actions = 3  # cool, nothing, heat\n\n"
                "        # Initialize transition and observation matrices\n"
                "        self.transition_matrix = self._create_transition_matrix()\n"
                "        self.observation_matrix = self._create_observation_matrix()\n\n"
                "    def _create_transition_matrix(self):\n"
                "        # Logic for creating the transition dynamics\n"
                "        # ...\n\n"
                "    def update_belief(self, prior_belief, observation, action):\n"
                "        # Implement belief updating using VFE\n"
                "        # ...\n\n"
                "    def select_action(self, current_belief):\n"
                "        # Implement action selection using EFE\n"
                "        # ...\n"
                "```"
            )
        else:
            return (
                "I'll provide expert guidance and feedback based on my knowledge "
                "in the field, tailored to the current research phase."
            )
    
    def context(self, phase: str) -> str:
        """Get context for a specific phase.
        
        Args:
            phase: The current phase
            
        Returns:
            str: The context
        """
        # Research topic from config
        research_topic = self.notes.get("research-topic", "Unspecified research topic")
        
        # Phase-specific notes
        phase_notes = self.notes.get(phase, [])
        phase_notes_str = "\n".join([f"- {note}" for note in phase_notes])
        
        context = f"Research Topic: {research_topic}\n\n"
        
        if phase_notes_str:
            context += f"Guidelines for {phase}:\n{phase_notes_str}\n\n"
        
        # Add specific context based on the phase
        if phase == "plan-formulation":
            context += (
                "We are at the beginning of the research project and need to develop "
                "a comprehensive plan. The focus should be on both theoretical formulation "
                "and practical implementation details."
            )
        elif phase == "data-preparation":
            context += (
                "We need to design appropriate data structures and generators for "
                "the thermal homeostasis system, including transition dynamics and "
                "observation models."
            )
        elif phase == "running-experiments":
            context += (
                "We need to implement the POMDP with Active Inference and run "
                "experiments to test its effectiveness in maintaining thermal homeostasis."
            )
        elif phase == "results-interpretation":
            context += (
                "We need to analyze the results of our experiments and interpret "
                "them in the context of Active Inference theory."
            )
        elif phase == "report-writing":
            context += (
                "We need to document our research findings in a clear and comprehensive "
                "manner, including both theoretical foundations and empirical results."
            )
        
        return context
    
    def phase_prompt(self, phase: str) -> str:
        """Get the prompt for a specific phase.
        
        Args:
            phase: The current phase
            
        Returns:
            str: The prompt
        """
        if phase == "plan-formulation":
            return (
                "Please develop a comprehensive research plan for implementing a POMDP "
                "with Active Inference for thermal homeostasis. The plan should include "
                "theoretical foundations, implementation approach, evaluation strategy, "
                "and potential challenges."
            )
        elif phase == "data-preparation":
            return (
                "Please design the data structures and generators needed for the thermal "
                "homeostasis POMDP. This should include transition dynamics, observation "
                "models, and prior beliefs."
            )
        elif phase == "running-experiments":
            return (
                "Please implement the POMDP with Active Inference for thermal homeostasis "
                "and design experiments to test its effectiveness. Include code for "
                "the implementation and experimental setup."
            )
        elif phase == "results-interpretation":
            return (
                "Please analyze the results of our experiments with the POMDP Active "
                "Inference model for thermal homeostasis. Interpret the findings in "
                "the context of theoretical expectations and identify strengths and "
                "limitations."
            )
        elif phase == "report-writing":
            return (
                "Please outline a comprehensive research report documenting our work "
                "on implementing a POMDP with Active Inference for thermal homeostasis. "
                "The report should include theoretical foundations, implementation details, "
                "experimental results, and conclusions."
            )
        else:
            return (
                "Please provide expert guidance on our research project based on your "
                "extensive knowledge in the field."
            )
    
    def evaluate_student_work(self, work: str, phase: str) -> str:
        """Evaluate student work for a specific phase.
        
        Args:
            work: The student's work
            phase: The current phase
            
        Returns:
            str: Evaluation of the student's work
        """
        system_message = (
            f"You are a professor evaluating student work on a research project. "
            f"The current phase is '{phase}'. Provide constructive feedback, highlighting "
            f"strengths and areas for improvement. Be thorough but encouraging."
        )
        
        prompt = f"Please evaluate the following student work for the {phase} phase:\n\n{work}"
        
        response = self.override_inference(prompt, temperature=0.3)
        
        return response
    
    def provide_feedback(self, work: str, evaluation: str) -> str:
        """Provide feedback on student work based on an evaluation.
        
        Args:
            work: The student's work
            evaluation: Evaluation of the student's work
            
        Returns:
            str: Feedback for the student
        """
        system_message = (
            "You are a professor providing feedback to a student on their research work. "
            "Based on the evaluation, provide constructive feedback that will help the "
            "student improve their work. Be specific, actionable, and encouraging."
        )
        
        prompt = (
            f"Based on the following evaluation of student work:\n\n{evaluation}\n\n"
            f"Provide feedback to help the student improve their work."
        )
        
        response = self.override_inference(prompt, temperature=0.3)
        
        return response 