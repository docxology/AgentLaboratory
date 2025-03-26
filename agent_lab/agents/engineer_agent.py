"""Engineer Agent for technical implementation aspects in Agent Laboratory.

This class implements an agent focused on technical implementation,
code quality, and computational efficiency.
"""

import logging
from typing import Dict, Any, Optional

from agent_lab.core.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class EngineerAgent(BaseAgent):
    """Agent that specializes in technical implementation and engineering aspects."""
    
    def __init__(self, api_key: str, config: Dict[str, Any]):
        """Initialize the Engineer Agent.
        
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
        self.system_prompt = """You are an Expert Engineer Agent specializing in software development, 
algorithm implementation, and computational methods. Your expertise includes:

1. Writing high-quality, efficient code implementations
2. Designing robust software architectures
3. Optimizing algorithms for performance
4. Implementing computational models
5. Ensuring code correctness and testability

Your responses should focus on technical implementation details, code quality, 
computational efficiency, and practical engineering concerns. When providing code,
ensure it follows best practices, includes appropriate error handling, and is
well-documented.

Provide concrete technical solutions, pseudocode, or actual implementations when
appropriate. Be precise, detailed, and rigorous in your technical assessments."""
        
        # Update with any expertise from config
        if "expertise" in config:
            expertise_str = ", ".join(config["expertise"])
            self.system_prompt += f"\n\nYour specific areas of expertise include: {expertise_str}."
        
        # Initialize dialogue history
        self.dialogue_history = []
        
        logger.info("Initialized Engineer Agent")
    
    def role_description(self) -> str:
        """Get the role description for the agent.
        
        Returns:
            str: The role description
        """
        return "Software Engineer and Technical Implementation Expert"
    
    def command_descriptions(self, phase: str) -> str:
        """Get the command descriptions for a specific phase.
        
        Args:
            phase: The current phase
            
        Returns:
            str: The command descriptions
        """
        general_desc = """
        You are expected to provide technical expertise and implementation guidance.
        Your responses should be structured, clear, and focused on technical details.
        """
        
        phase_specific = {
            "plan-formulation": "Analyze technical feasibility and implementation approaches for the proposed plan.",
            "data-preparation": "Provide guidance on data structures, processing methods, and technical aspects of data preparation.",
            "running-experiments": "Suggest efficient implementation strategies, code organization, and technical optimizations.",
            "results-interpretation": "Analyze computational methods and provide technical insights on the results.",
            "report-writing": "Contribute clear technical explanations and ensure accuracy of technical content."
        }
        
        phase_desc = phase_specific.get(phase, f"Provide technical engineering guidance for the {phase} phase.")
        
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
            Example: "The proposed plan includes reinforcement learning for control optimization. 
            I recommend using the following technical approach:
            1. Implement a state-action-reward framework using PyTorch
            2. Define the environment model with clear state transitions
            3. Use a Deep Q-Network architecture with experience replay
            4. Consider computational efficiency through batch processing"
            """,
            
            "data-preparation": """
            Example: "For the time series data preparation, I suggest:
            1. Use pandas with the following preprocessing pipeline:
               ```python
               def preprocess_data(df):
                   # Handle missing values with forward fill
                   df = df.ffill()
                   # Normalize using robust scaler
                   from sklearn.preprocessing import RobustScaler
                   scaler = RobustScaler()
                   df[['feature1', 'feature2']] = scaler.fit_transform(df[['feature1', 'feature2']])
                   return df
               ```
            2. Implement efficient data loading with generators
            3. Ensure validation split is time-consistent"
            """,
            
            "running-experiments": """
            Example: "I've analyzed the experimental setup and recommend:
            1. Refactor the main algorithm to use vectorized operations:
               ```python
               # Instead of:
               for i in range(len(data)):
                   results[i] = compute_function(data[i])
                   
               # Use:
               results = np.vectorize(compute_function)(data)
               ```
            2. Implement parallel processing for the simulation runs
            3. Add proper error handling and logging to track experiment progress"
            """
        }
        
        return examples.get(phase, "Example: Provide detailed technical analysis with code examples where appropriate.")
    
    def context(self, phase: str) -> str:
        """Get context for a specific phase.
        
        Args:
            phase: The current phase
            
        Returns:
            str: The context
        """
        contexts = {
            "plan-formulation": "Consider technical feasibility, implementation complexity, and resource requirements.",
            "data-preparation": "Focus on data structures, processing efficiency, validation, and technical soundness.",
            "running-experiments": "Emphasize code quality, algorithm optimization, and robust implementation.",
            "results-interpretation": "Analyze computational methods and technical aspects of the results.",
            "report-writing": "Ensure technical accuracy and clarity in explanations."
        }
        
        return contexts.get(phase, "Provide technical expertise relevant to this phase.")
    
    def phase_prompt(self, phase: str) -> str:
        """Get the prompt for a specific phase.
        
        Args:
            phase: The current phase
            
        Returns:
            str: The prompt
        """
        prompts = {
            "plan-formulation": """
            Review the research plan from a technical implementation perspective. Analyze:
            - Technical feasibility of proposed methods
            - Implementation complexity and challenges
            - Required tools, libraries, and computational resources
            - Software architecture considerations
            
            Provide detailed technical recommendations to improve the plan.
            """,
            
            "data-preparation": """
            Analyze the data preparation approach from a technical perspective. Consider:
            - Data structures and processing pipelines
            - Efficiency and scalability of methods
            - Validation and quality control
            - Technical implementation details
            
            Suggest specific implementation improvements with code examples where appropriate.
            """,
            
            "running-experiments": """
            Review the experimental implementation. Focus on:
            - Code quality and organization
            - Algorithm efficiency and optimization
            - Technical soundness and robustness
            - Testing and validation approaches
            
            Provide concrete technical improvements with code snippets where beneficial.
            """,
            
            "results-interpretation": """
            Analyze the computational methods used for results analysis. Consider:
            - Technical validity of analysis methods
            - Implementation correctness
            - Alternative technical approaches
            - Computational efficiency
            
            Provide technical insights to strengthen the results interpretation.
            """,
            
            "report-writing": """
            Review the technical content of the report. Ensure:
            - Accuracy of technical explanations
            - Clarity of implementation details
            - Proper description of algorithms and methods
            - Completeness of technical documentation
            
            Suggest improvements to enhance the technical quality of the report.
            """
        }
        
        return prompts.get(phase, f"Provide technical engineering expertise for the {phase} phase.")
    
    def complete_phase(self, phase_name: str, task_notes: str) -> str:
        """Complete a specific phase of the research workflow.
        
        Args:
            phase_name: Name of the phase to complete
            task_notes: Notes or instructions for the phase
            
        Returns:
            str: The output from the agent for this phase
        """
        # Create a phase-specific prompt that emphasizes engineering aspects
        phase_specific_prompts = {
            "plan-formulation": """
                Review the research plan from an engineering perspective. Focus on:
                - Technical feasibility of the proposed approaches
                - Computational requirements and constraints
                - Software architecture considerations
                - Potential implementation challenges
                - Tools, libraries, and frameworks that would be appropriate
                
                Provide a technical assessment of the plan and suggest engineering improvements.
            """,
            
            "data-preparation": """
                Analyze the data preparation strategy. Consider:
                - Data structures and storage mechanisms
                - Processing efficiency and scalability
                - Validation and error handling
                - Preprocessing algorithms and techniques
                - Testing strategies for data pipeline
                
                Suggest optimal approaches for implementing the data preparation process.
            """,
            
            "running-experiments": """
                Review the experimental implementation. Focus on:
                - Code quality and organization
                - Algorithm optimization opportunities
                - Performance bottlenecks
                - Testing and validation procedures
                - Reproducibility considerations
                
                Provide detailed guidance on implementing the experiments efficiently and robustly.
            """,
            
            "results-interpretation": """
                Analyze the results from a technical perspective. Consider:
                - Statistical validity of the analysis
                - Computational methods used for analysis
                - Visualization techniques and tools
                - Alternative analysis approaches
                - Technical limitations and caveats
                
                Provide insights on improving the technical aspects of the results interpretation.
            """,
            
            "report-writing": """
                Review the technical sections of the report. Focus on:
                - Clarity of technical explanations
                - Accuracy of algorithm descriptions
                - Completeness of implementation details
                - Appropriateness of technical figures and tables
                - Reproducibility instructions
                
                Suggest improvements to the technical content of the report.
            """
        }
        
        # Get the phase-specific prompt or use a generic one
        phase_prompt = phase_specific_prompts.get(
            phase_name,
            f"Provide technical engineering input for the {phase_name} phase."
        )
        
        # Combine with task notes
        full_prompt = f"""
        {phase_prompt}
        
        Here are the specific task notes:
        
        {task_notes}
        
        Provide your detailed technical engineering analysis and recommendations.
        """
        
        # Log the phase start
        logger.info(f"Engineer Agent starting work on phase: {phase_name}")
        
        # Get response from LLM
        response = self.get_completion(
            full_prompt,
            system_message=self.system_prompt
        )
        
        # Log completion
        logger.info(f"Engineer Agent completed phase: {phase_name}")
        
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