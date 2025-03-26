"""
Professor agent for the Agent Laboratory.

This module provides the ProfessorAgent class, responsible for research planning
and report writing in the Agent Laboratory.
"""

import os
from typing import Dict, List, Any, Optional, Union
import logging

from agent_lab.agents.base_agent import BaseAgent
from agent_lab.core.llm_interface import LLMInterface


class ProfessorAgent(BaseAgent):
    """
    Professor agent responsible for research planning and report writing.
    
    This agent acts as a senior researcher guiding the research process,
    formulating research plans, analyzing results, and writing reports.
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        deepseek_api_key: Optional[str] = None,
        notes: Optional[Dict[str, List[str]]] = None,
        max_steps: int = 10,
        temperature: float = 0.7,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the professor agent.
        
        Args:
            model: LLM model to use
            api_key: OpenAI API key
            deepseek_api_key: DeepSeek API key
            notes: Task notes for different phases
            max_steps: Maximum steps to execute
            temperature: Sampling temperature
            logger: Logger instance
        """
        super().__init__(
            name="professor",
            model=model,
            api_key=api_key,
            deepseek_api_key=deepseek_api_key,
            notes=notes,
            max_steps=max_steps,
            temperature=temperature,
            logger=logger
        )
        
        # Create LLM interface for model inference
        self.llm_interface = LLMInterface(
            model_name=model,
            api_key=api_key,
            deepseek_api_key=deepseek_api_key,
            temperature=temperature,
            logger=logger
        )
    
    def override_inference(
        self,
        query: str,
        temperature: Optional[float] = None
    ) -> str:
        """
        Run model inference with a custom query.
        
        Args:
            query: Query text
            temperature: Optional temperature override
            
        Returns:
            Model response
        """
        temp = temperature if temperature is not None else self.temperature
        return self.llm_interface.get_completion(query, temperature=temp)
    
    def generate_readme(
        self,
        research_topic: str,
        research_plan: str,
        results: str,
        output_dir: str
    ) -> str:
        """
        Generate a README file for the research project.
        
        Args:
            research_topic: Topic of the research
            research_plan: Research plan
            results: Research results
            output_dir: Output directory for the README
            
        Returns:
            Path to the generated README file
        """
        prompt = f"""Create a comprehensive README.md file for a research project with the following details:

Research Topic: {research_topic}

Research Plan:
{research_plan}

Results:
{results}

Include the following sections in the README:
1. Introduction
2. Research Objectives
3. Methodology
4. Key Findings
5. Conclusions
6. References

Format the README using Markdown syntax and ensure it's well-structured and professional.
"""
        
        readme_content = self.override_inference(prompt, temperature=0.3)
        readme_path = os.path.join(output_dir, "README.md")
        
        with open(readme_path, "w") as f:
            f.write(readme_content)
        
        return readme_path
    
    def role_description(self) -> str:
        """
        Get the role description for the professor agent.
        
        Returns:
            Role description string
        """
        return """You are an expert Professor Agent in the Agent Laboratory. Your role is to lead the research process with expertise in research methodologies, experimental design, and academic writing. You excel at formulating research plans, reviewing literature, designing experiments, interpreting results, and writing comprehensive reports. As a professor, you should use precise technical language appropriate to the research domain and follow academic standards in your approach to research problems."""
    
    def context(self, phase: str) -> str:
        """
        Get context information for a specific research phase.
        
        Args:
            phase: Name of the research phase
            
        Returns:
            Context string for the specified phase
        """
        if phase == "report-writing":
            return """For the report writing phase, focus on synthesizing all the research findings into a comprehensive academic report. Make sure to include:
1. An executive summary of the research
2. Clear documentation of the methodology
3. Detailed presentation of results with appropriate visualizations
4. Discussion of the implications and limitations
5. Conclusions and recommendations for future work
6. Complete references in an appropriate academic style

Your report should follow academic writing conventions and maintain scientific rigor throughout."""
        
        else:
            return f"""You are currently in the {phase} phase of the research process. Focus on providing detailed, well-structured, and academically rigorous output appropriate to this phase. Your responses should reflect deep expertise in the research topic and follow scientific methodological principles. When appropriate, suggest specific techniques, tools, or approaches that would be suitable for this research phase."""
    
    def example_command(self, phase: str) -> str:
        """
        Get an example command for a specific research phase.
        
        Args:
            phase: Name of the research phase
            
        Returns:
            Example command string for the specified phase
        """
        if phase == "plan-formulation":
            return "Develop a comprehensive research plan to investigate [research topic], including objectives, methodologies, and expected outcomes."
        
        elif phase == "report-writing":
            return "Generate a complete research report that synthesizes our findings on [research topic], including all standard academic sections."
        
        else:
            return f"Proceed with the {phase} phase of our research on the given topic."
    
    def command_descriptions(self, phase: str) -> Dict[str, str]:
        """
        Get the command descriptions for a specific research phase.
        
        Args:
            phase: Name of the research phase
            
        Returns:
            Dictionary mapping command names to descriptions
        """
        if phase == "plan-formulation":
            return {
                "develop_research_plan": "Develop a comprehensive research plan with clear objectives and methodology",
                "identify_research_questions": "Identify key research questions to be addressed in the study",
                "outline_methodology": "Outline the proposed methodology for investigating the research topic",
                "define_success_criteria": "Define criteria for evaluating the success of the research"
            }
        
        elif phase == "literature-review":
            return {
                "summarize_key_literature": "Summarize key literature relevant to the research topic",
                "identify_research_gaps": "Identify gaps in the existing research",
                "analyze_methodologies": "Analyze methodologies used in previous research",
                "synthesize_findings": "Synthesize findings from the literature review"
            }
        
        elif phase == "data-preparation":
            return {
                "identify_data_sources": "Identify appropriate data sources for the research",
                "design_data_collection": "Design the data collection process",
                "create_preprocessing_pipeline": "Create a data preprocessing pipeline",
                "develop_data_validation": "Develop methods for validating the data"
            }
        
        elif phase == "running-experiments":
            return {
                "design_experiments": "Design experiments to address the research questions",
                "implement_methodology": "Implement the proposed methodology",
                "execute_experiments": "Execute the designed experiments",
                "document_process": "Document the experimental process and parameters"
            }
        
        elif phase == "results-interpretation":
            return {
                "analyze_results": "Analyze the experimental results",
                "interpret_findings": "Interpret the findings in the context of the research questions",
                "identify_patterns": "Identify patterns and trends in the data",
                "assess_implications": "Assess the implications of the results"
            }
        
        elif phase == "report-writing":
            return {
                "write_introduction": "Write the introduction section of the research report",
                "describe_methodology": "Describe the methodology used in the research",
                "present_results": "Present the results of the research",
                "discuss_findings": "Discuss the findings and their implications",
                "draw_conclusions": "Draw conclusions based on the research",
                "complete_report": "Complete the full research report"
            }
        
        else:
            return {
                "proceed": f"Proceed with the {phase} phase of the research",
                "summarize": "Summarize the current state of the research",
                "suggest_next_steps": "Suggest next steps for the research"
            }
    
    def phase_prompt(self, phase: str) -> str:
        """
        Get the prompt for a specific research phase.
        
        Args:
            phase: Name of the research phase
            
        Returns:
            Prompt string for the specified phase
        """
        if phase == "plan-formulation":
            return """Please develop a comprehensive research plan for investigating the provided research topic. Your plan should include:

1. Clear research objectives and questions
2. Theoretical framework underpinning the research
3. Detailed methodology including:
   - Research design
   - Data collection methods
   - Analysis approaches
4. Expected outcomes and deliverables
5. Timeline for completing each phase of the research
6. Potential challenges and mitigation strategies

Format your plan in a clear, well-structured manner suitable for academic research."""
        
        elif phase == "literature-review":
            return """Please conduct a comprehensive literature review on the research topic. Your review should:

1. Identify key concepts and theories relevant to the topic
2. Summarize and synthesize existing research
3. Identify methodologies used in previous studies
4. Highlight gaps in current knowledge
5. Establish how this research will contribute to the field
6. Provide a critical analysis of the strength of existing evidence

Please organize the literature review by themes or chronologically as appropriate for the topic."""
        
        elif phase == "data-preparation":
            return """Please develop a detailed data preparation plan for this research. Your plan should include:

1. Identification of required data sources
2. Data collection methods and tools
3. Data cleaning and preprocessing steps
4. Feature selection or engineering approaches
5. Data validation and quality assurance methods
6. Data storage and management plan
7. Ethical considerations related to data handling

Please be specific about tools, techniques, and methodologies appropriate for this research topic."""
        
        elif phase == "running-experiments":
            return """Please design and outline the experimental protocol for this research. Your experimental design should include:

1. Clear definition of experimental variables
2. Experimental setup and configuration
3. Implementation details of the methodology
4. Tools and environments needed for the experiments
5. Procedures for recording and documenting results
6. Methods for ensuring reproducibility
7. Contingency plans for potential experimental issues

Please provide a step-by-step guide that would allow other researchers to replicate your experiments."""
        
        elif phase == "results-interpretation":
            return """Please provide a comprehensive interpretation of the research results. Your interpretation should include:

1. Summary of key findings
2. Analysis of patterns and trends in the data
3. Statistical significance of results (if applicable)
4. Comparison with findings from previous research
5. Discussion of how the results address the research questions
6. Identification of limitations in the results
7. Implications of the findings for theory and practice

Please support your interpretation with specific references to the data and use appropriate analytical frameworks."""
        
        elif phase == "report-writing":
            return """Please write a comprehensive research report synthesizing all aspects of this research project. Your report should include:

1. Abstract summarizing the research
2. Introduction with background and research questions
3. Literature review section
4. Detailed methodology
5. Results presentation with appropriate visualizations
6. Discussion of findings and their implications
7. Conclusion including recommendations for future work
8. Complete references in academic format
9. Appendices with supplementary material as needed

Format the report following academic conventions and ensure it communicates the research clearly to both expert and informed non-expert audiences."""
        
        else:
            return f"""Please proceed with the {phase} phase of the research process. Focus on producing high-quality, academically rigorous work that advances our understanding of the research topic. Provide detailed, well-structured output that demonstrates expertise in the subject matter.""" 