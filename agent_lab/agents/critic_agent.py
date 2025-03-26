"""Critic Agent for critical review and quality assessment in Agent Laboratory.

This class implements an agent focused on critical review, quality assessment,
and improvement suggestions.
"""

import logging
from typing import Dict, Any, Optional

from agent_lab.core.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class CriticAgent(BaseAgent):
    """Agent that specializes in critical review and quality assessment."""
    
    def __init__(self, api_key: str, config: Dict[str, Any]):
        """Initialize the Critic Agent.
        
        Args:
            api_key: API key for LLM service
            config: Configuration dictionary
        """
        super().__init__(
            model=config.get("model", "gpt-4o-mini"),
            notes=config.get("notes", {}),
            max_steps=config.get("max_steps", 10),
            api_key=api_key,
            logger=logger
        )
        self.system_prompt = """You are an Expert Critic Agent specializing in scientific peer review, 
critical analysis, and quality assessment. Your expertise includes:

1. Identifying strengths and weaknesses in research
2. Providing constructive criticism and actionable feedback
3. Evaluating methodological rigor and scientific validity
4. Assessing clarity, coherence, and organization of scientific communication
5. Recommending specific improvements to enhance quality

Your critical reviews should be:
- Balanced: Acknowledge strengths while identifying weaknesses
- Specific: Provide concrete examples and detailed suggestions
- Constructive: Focus on how to improve, not just what is wrong
- Evidence-based: Ground your criticism in scientific principles and best practices
- Prioritized: Focus on the most important issues first

Your goal is to help improve the quality of the research through thoughtful, 
rigorous, and constructive feedback."""
        
        # Update with any expertise from config
        if "expertise" in config:
            expertise_str = ", ".join(config["expertise"])
            self.system_prompt += f"\n\nYour specific areas of expertise include: {expertise_str}."
        
        # Initialize dialogue history
        self.dialogue_history = []
        
        logger.info("Initialized Critic Agent")
    
    def role_description(self) -> str:
        """Get the role description for the agent.
        
        Returns:
            str: The role description
        """
        return "Scientific Critic and Quality Assessment Expert"
    
    def command_descriptions(self, phase: str) -> str:
        """Get the command descriptions for a specific phase.
        
        Args:
            phase: The current phase
            
        Returns:
            str: The command descriptions
        """
        general_desc = """
        You are expected to provide critical review and quality assessment.
        Your responses should be balanced, specific, constructive, evidence-based, and prioritized.
        """
        
        phase_specific = {
            "plan-formulation": "Evaluate research questions, methodology, and overall scientific approach.",
            "data-preparation": "Assess data collection methods, preprocessing steps, and quality control measures.",
            "running-experiments": "Analyze experimental design, controls, and scientific validity.",
            "results-interpretation": "Evaluate interpretation of findings, statistical analysis, and conclusion validity.",
            "report-writing": "Review clarity, organization, scientific rigor, and overall quality of the report."
        }
        
        phase_desc = phase_specific.get(phase, f"Provide critical evaluation for the {phase} phase.")
        
        return f"{general_desc}\n{phase_desc}"
    
    def example_command(self, phase: str) -> str:
        """Get an example command for a specific phase.
        
        Args:
            phase: The current phase
            
        Returns:
            str: The example command
        """
        examples = {
            "plan-formulation": """
            Example: "I've reviewed the research plan and identified several strengths and areas for improvement:
            
            STRENGTHS:
            1. Clear formulation of the primary research question
            2. Appropriate selection of methods for addressing the question
            
            AREAS FOR IMPROVEMENT:
            1. The hypotheses lack specificity and testability; consider reformulating as: [specific suggestions]
            2. The sampling strategy may introduce bias; recommend the following adjustments: [specific recommendations]
            3. Statistical power analysis is needed to determine appropriate sample size
            
            PRIORITY RECOMMENDATIONS:
            1. Refine hypotheses to be more specific and testable
            2. Add power analysis to justify sample size"
            """,
            
            "data-preparation": """
            Example: "My critical review of the data preparation approach:
            
            STRENGTHS:
            1. Comprehensive data collection from multiple sources
            2. Appropriate handling of missing values
            
            AREAS FOR IMPROVEMENT:
            1. Potential selection bias in how datasets were chosen; recommend: [specific recommendation]
            2. Insufficient validation of data quality; suggest adding: [specific suggestion]
            3. Data preprocessing steps may introduce artifacts; consider alternative approaches such as: [specific approaches]
            
            PRIORITY RECOMMENDATIONS:
            1. Implement robust data validation procedures
            2. Document potential biases and limitations of the dataset"
            """,
            
            "results-interpretation": """
            Example: "Critical evaluation of the results interpretation:
            
            STRENGTHS:
            1. Clear presentation of primary findings
            2. Appropriate statistical tests for the data type
            
            AREAS FOR IMPROVEMENT:
            1. Causal claims exceed what the correlational data support; recommend rephrasing as: [specific suggestion]
            2. Alternative explanations for findings are not adequately addressed; consider discussing: [specific alternatives]
            3. Limitations of the study are understated; suggest acknowledging: [specific limitations]
            
            PRIORITY RECOMMENDATIONS:
            1. Align claims more carefully with the strength of evidence
            2. Address alternative explanations for the findings"
            """
        }
        
        return examples.get(phase, "Example: Provide a balanced critique with specific, constructive recommendations.")
    
    def context(self, phase: str) -> str:
        """Get context for a specific phase.
        
        Args:
            phase: The current phase
            
        Returns:
            str: The context
        """
        contexts = {
            "plan-formulation": "Focus on research questions, methodology, and scientific approach.",
            "data-preparation": "Evaluate data sources, quality, preprocessing, and validation.",
            "running-experiments": "Assess experimental design, controls, and implementation.",
            "results-interpretation": "Examine statistical analysis, findings, and conclusions.",
            "report-writing": "Review organization, clarity, completeness, and scientific rigor."
        }
        
        return contexts.get(phase, "Provide critical evaluation relevant to this phase.")
    
    def phase_prompt(self, phase: str) -> str:
        """Get the prompt for a specific phase.
        
        Args:
            phase: The current phase
            
        Returns:
            str: The prompt
        """
        prompts = {
            "plan-formulation": """
            Critically evaluate the research plan. Focus on:
            - Clarity and specificity of research questions/hypotheses
            - Appropriateness of methodology
            - Scientific rigor and validity
            - Potential limitations or weaknesses
            - Alignment with scientific best practices
            
            Provide a balanced critique with specific recommendations for improvement.
            Organize your feedback into strengths, areas for improvement, and priority recommendations.
            """,
            
            "data-preparation": """
            Critically evaluate the data preparation approach. Focus on:
            - Quality and appropriateness of data sources
            - Methods for handling missing or problematic data
            - Validation and quality control procedures
            - Potential biases or limitations
            - Documentation and reproducibility
            
            Provide a balanced critique with specific recommendations for improvement.
            Organize your feedback into strengths, areas for improvement, and priority recommendations.
            """,
            
            "running-experiments": """
            Critically evaluate the experimental design and implementation. Focus on:
            - Scientific validity and rigor
            - Appropriate controls and comparisons
            - Methodological soundness
            - Potential confounds or limitations
            - Reproducibility and transparency
            
            Provide a balanced critique with specific recommendations for improvement.
            Organize your feedback into strengths, areas for improvement, and priority recommendations.
            """,
            
            "results-interpretation": """
            Critically evaluate the results interpretation. Focus on:
            - Alignment of conclusions with evidence
            - Statistical and methodological soundness
            - Alternative explanations for findings
            - Appropriate acknowledgment of limitations
            - Strength and validity of claims
            
            Provide a balanced critique with specific recommendations for improvement.
            Organize your feedback into strengths, areas for improvement, and priority recommendations.
            """,
            
            "report-writing": """
            Critically evaluate the scientific report. Focus on:
            - Organization and logical flow
            - Clarity and precision of language
            - Completeness of methods description
            - Appropriate presentation of results
            - Quality of discussion and conclusions
            
            Provide a balanced critique with specific recommendations for improvement.
            Organize your feedback into strengths, areas for improvement, and priority recommendations.
            """
        }
        
        return prompts.get(phase, f"Provide a critical evaluation for the {phase} phase.")
    
    def complete_phase(self, phase_name: str, task_notes: str) -> str:
        """Complete a specific phase of the research workflow.
        
        Args:
            phase_name: Name of the phase to complete
            task_notes: Notes or instructions for the phase
            
        Returns:
            str: The output from the agent for this phase
        """
        # Create a phase-specific prompt that emphasizes critical review
        phase_specific_prompts = {
            "plan-formulation": """
                Critically evaluate the research plan. Focus on:
                - Clarity and coherence of research questions/objectives
                - Appropriateness of methodology for the research questions
                - Logical structure and organization of the plan
                - Potential weaknesses, limitations, or gaps
                - Alignment with scientific best practices
                
                Provide a balanced critique with specific suggestions for improvement.
            """,
            
            "data-preparation": """
                Critically evaluate the data preparation approach. Consider:
                - Appropriateness of data sources and collection methods
                - Potential biases or limitations in the data
                - Thoroughness of preprocessing and cleaning procedures
                - Validity of assumptions about the data
                - Completeness of data documentation
                
                Provide specific recommendations to enhance data quality and preparation.
            """,
            
            "running-experiments": """
                Critically evaluate the experimental design and implementation. Focus on:
                - Methodological rigor and scientific validity
                - Appropriateness of experimental controls and comparisons
                - Potential confounds or threats to validity
                - Reproducibility and transparency of methods
                - Alignment with disciplinary standards
                
                Provide detailed feedback on improving experimental quality.
            """,
            
            "results-interpretation": """
                Critically evaluate the results interpretation. Consider:
                - Alignment between results and claims
                - Appropriate acknowledgment of limitations
                - Alternative interpretations of the data
                - Strength of evidence for conclusions
                - Logical consistency and thoroughness
                
                Provide balanced feedback on strengthening the interpretation.
            """,
            
            "report-writing": """
                Critically evaluate the scientific report. Focus on:
                - Clarity, coherence, and flow of the narrative
                - Appropriate contextualization within the literature
                - Completeness and accuracy of methodological descriptions
                - Effective presentation of results
                - Quality of discussion and conclusions
                
                Provide specific suggestions to enhance the report quality.
            """
        }
        
        # Get the phase-specific prompt or use a generic one
        phase_prompt = phase_specific_prompts.get(
            phase_name,
            f"Critically evaluate the content for the {phase_name} phase."
        )
        
        # Combine with task notes
        full_prompt = f"""
        {phase_prompt}
        
        Here is the content to review:
        
        {task_notes}
        
        Provide your critical evaluation with specific, constructive feedback for improvement.
        Organize your feedback into:
        
        STRENGTHS:
        [List the major strengths and positive aspects]
        
        AREAS FOR IMPROVEMENT:
        [List specific weaknesses or limitations with concrete suggestions for addressing each]
        
        OVERALL ASSESSMENT:
        [Provide a balanced summary assessment and prioritized recommendations]
        """
        
        # Log the phase start
        logger.info(f"Critic Agent starting work on phase: {phase_name}")
        
        # Get response from LLM
        response = self.get_completion(
            full_prompt,
            system_message=self.system_prompt
        )
        
        # Log completion
        logger.info(f"Critic Agent completed phase: {phase_name}")
        
        # Add to dialogue history
        self.dialogue_history.append({
            "role": "user",
            "content": full_prompt
        })
        self.dialogue_history.append({
            "role": "assistant",
            "content": response
        })
        
        return response

    def get_completion(self, prompt: str, system_message: Optional[str] = None) -> str:
        """Get a completion from the LLM.
        
        Args:
            prompt: The prompt to send to the model
            system_message: Optional system message
            
        Returns:
            str: The model's response
        """
        # Use the LLM interface to get a completion
        return self.llm.completion(
            prompt=prompt,
            system_message=system_message or self.system_prompt
        ) 