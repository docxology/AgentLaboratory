"""Multi-agent integration module for Agent Laboratory.

This module integrates multiple agents to collaborate on research tasks,
providing more comprehensive and diverse perspectives.
"""

import logging
import os
import json
import datetime
import re
from typing import Dict, List, Any, Optional

from agent_lab.core.base_agent import BaseAgent
from agent_lab.agents.professor_agent import ProfessorAgent
from agent_lab.agents.engineer_agent import EngineerAgent
from agent_lab.agents.critic_agent import CriticAgent

logger = logging.getLogger(__name__)

class MultiAgentTeam:
    """A team of multiple agents that can collaborate on research tasks."""
    
    def __init__(self, config: Dict[str, Any], output_dir: str):
        """Initialize the multi-agent team.
        
        Args:
            config: Configuration dictionary
            output_dir: Output directory for agent artifacts
        """
        self.config = config
        self.output_dir = output_dir
        self.agents = {}
        self.interactions = []
        self.shared_knowledge = {}  # Repository of shared knowledge between agents
        
        # Initialize research plan with more details from config
        self._initialize_research_plan()
        
        # Get API key from config or environment
        api_key = config.get("api_key") or config.get("llm_config", {}).get("api_key")
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
        
        # Create agents
        self._create_agents(api_key)
        
        # Set up the artifacts directory
        self.artifacts_dir = os.path.join(self.output_dir, "artifacts")
        os.makedirs(self.artifacts_dir, exist_ok=True)
        
        logger.info(f"Initialized multi-agent team with {len(self.agents)} agents")
    
    def _initialize_research_plan(self):
        """Initialize the research plan with details from the configuration."""
        # Get basic research topic
        research_topic = self.config.get("research_topic", "")
        
        # Extract key parameters from the research topic
        control_states = self._extract_parameter(research_topic, r"(\d+)\s+control\s+states?", "3")
        latent_states = self._extract_parameter(research_topic, r"(\d+)\s+latent\s+states?", "5")
        observation_levels = self._extract_parameter(research_topic, r"(\d+)\s+discrete\s+levels?", "10")
        
        # Initialize the research parameters dictionary
        research_parameters = {
            "control_states": control_states,
            "latent_states": latent_states, 
            "observation_levels": observation_levels,
            "model_type": "POMDP with Active Inference",
            "application": "Thermal Homeostasis"
        }
        
        # Get task notes for initial phases to inform the plan
        task_notes = self.config.get("task_notes", {})
        initial_task_details = []
        
        # Extract important details from task notes
        for phase in ["literature-review", "plan-formulation", "data-preparation"]:
            if phase in task_notes:
                phase_notes = task_notes[phase]
                if isinstance(phase_notes, list):
                    initial_task_details.extend(phase_notes)
        
        # Create initial framework elements based on the task notes
        framework_elements = []
        for detail in initial_task_details:
            if "VFE" in detail or "Variational Free Energy" in detail:
                framework_elements.append("Variational Free Energy (VFE) for state estimation")
            if "EFE" in detail or "Expected Free Energy" in detail:
                framework_elements.append("Expected Free Energy (EFE) for action selection")
        
        # Initialize the research plan
        self.research_plan = {
            "topic": research_topic,
            "parameters": research_parameters,
            "framework_elements": framework_elements,
            "phases": [],
            "evolving_outline": {},
            "key_discoveries": [],
            "open_questions": [],
            "current_direction": "",
            "last_updated": str(datetime.datetime.now())
        }
        
        # Pre-populate evolving outline with our knowledge of the topic
        self.research_plan["evolving_outline"] = {
            "introduction": f"POMDP with Active Inference for {research_parameters['application']}",
            "background": "Variational and Expected Free Energy in Active Inference",
            "methodology": f"POMDP with {control_states} control states, {latent_states} latent states, and {observation_levels} observation levels",
            "implementation": "To be determined based on literature and methodology",
            "experiments": "To be designed",
            "results": "Pending experiments",
            "discussion": "To be developed",
            "conclusion": "To be determined"
        }
    
    def _extract_parameter(self, text: str, pattern: str, default: str) -> str:
        """Extract a parameter from text using regex.
        
        Args:
            text: Text to search in
            pattern: Regex pattern with a capture group
            default: Default value if not found
            
        Returns:
            str: Extracted parameter or default
        """
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
        return default
    
    def _create_agents(self, api_key: str):
        """Create the agent team.
        
        Args:
            api_key: API key for the LLM service
        """
        # Configure agent-specific settings
        professor_config = self.config.copy()
        professor_config["expertise"] = ["research methodology", "theoretical frameworks", 
                                        "experimental design", "mathematical modeling",
                                        "active inference", "POMDPs"]
        
        engineer_config = self.config.copy()
        engineer_config["expertise"] = ["programming", "computational methods", "algorithms",
                                       "software architecture", "numerical optimization",
                                       "data visualization", "testing frameworks"]
        
        critic_config = self.config.copy()
        critic_config["expertise"] = ["peer review", "scientific writing", "critical analysis",
                                     "result interpretation", "methodology assessment",
                                     "quality assurance", "reproducibility"]
        
        # Add memory limits if configured
        if self.config.get("agent_memory_limit"):
            memory_limit = self.config.get("agent_memory_limit", 10000)
            professor_config["memory_limit"] = memory_limit
            engineer_config["memory_limit"] = memory_limit
            critic_config["memory_limit"] = memory_limit
        
        # Create the agents
        self.agents["professor"] = ProfessorAgent(api_key=api_key, config=professor_config)
        self.agents["engineer"] = EngineerAgent(api_key=api_key, config=engineer_config)
        self.agents["critic"] = CriticAgent(api_key=api_key, config=critic_config)
    
    def save_states(self):
        """Save the state of all agents."""
        for agent_name, agent in self.agents.items():
            agent.save_state(self.output_dir)
            logger.info(f"Saved state for {agent_name} agent to {self.output_dir}")
        
        # Save shared knowledge
        knowledge_file = os.path.join(self.output_dir, "shared_knowledge.json")
        with open(knowledge_file, "w") as f:
            json.dump(self.shared_knowledge, f, indent=2)
        logger.info(f"Saved shared knowledge to {knowledge_file}")
        
        # Save research plan
        plan_file = os.path.join(self.output_dir, "research_plan.json")
        with open(plan_file, "w") as f:
            json.dump(self.research_plan, f, indent=2)
        logger.info(f"Saved research plan to {plan_file}")
    
    def load_states(self):
        """Load the state of all agents if state files exist."""
        for agent_name, agent in self.agents.items():
            state_file = os.path.join(self.output_dir, f"{agent.__class__.__name__}_state.json")
            if os.path.exists(state_file):
                agent.load_state(state_file)
                logger.info(f"Loaded state for {agent_name} agent from {state_file}")
        
        # Load shared knowledge
        knowledge_file = os.path.join(self.output_dir, "shared_knowledge.json")
        if os.path.exists(knowledge_file):
            with open(knowledge_file, "r") as f:
                self.shared_knowledge = json.load(f)
            logger.info(f"Loaded shared knowledge from {knowledge_file}")
            
        # Load research plan if available
        plan_file = os.path.join(self.output_dir, "research_plan.json")
        if os.path.exists(plan_file):
            with open(plan_file, "r") as f:
                self.research_plan = json.load(f)
            logger.info(f"Loaded research plan from {plan_file}")
    
    def _update_shared_knowledge(self, phase_name: str, content: str, source: str):
        """Update the shared knowledge repository.
        
        Args:
            phase_name: Name of the phase this knowledge comes from
            content: Content of the knowledge
            source: Source of the knowledge (agent name)
        """
        if phase_name not in self.shared_knowledge:
            self.shared_knowledge[phase_name] = []
        
        # Add new knowledge entry
        self.shared_knowledge[phase_name].append({
            "source": source,
            "content": content,
            "timestamp": str(datetime.datetime.now())
        })
        
        # Save the shared knowledge
        knowledge_file = os.path.join(self.output_dir, "shared_knowledge.json")
        with open(knowledge_file, "w") as f:
            json.dump(self.shared_knowledge, f, indent=2)
    
    def _extract_key_insights(self, content: str, phase_name: str) -> List[str]:
        """Extract key insights from content based on the phase.
        
        Args:
            content: The content to extract insights from
            phase_name: The current phase name
            
        Returns:
            List[str]: List of key insights
        """
        # Split content into paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        # Define patterns to look for based on phase
        key_patterns = {
            "literature-review": ["gap", "approach", "finding", "insight", "framework", "contribution", "Variational Free Energy", "Expected Free Energy", "Active Inference", "POMDP"],
            "plan-formulation": ["approach", "step", "method", "strategy", "outline", "procedure", "Variational Free Energy", "Expected Free Energy", "transition", "observation", "formulation"],
            "data-preparation": ["dataset", "model", "parameter", "variable", "structure", "representation", "transition", "observation", "belief", "action", "update"],
            "code-implementation": ["function", "class", "algorithm", "code", "implementation", "module", "POMDP", "Active Inference", "VFE", "EFE"],
            "running-experiments": ["result", "experiment", "performance", "metric", "comparison", "test", "simulation", "benchmark", "thermal"],
            "results-interpretation": ["finding", "insight", "implication", "conclusion", "analysis", "performance", "precision", "accuracy", "comparison"],
            "report-writing": ["section", "structure", "argument", "evidence", "conclusion", "formulation", "derivation"]
        }
        
        # Look specifically for equations and code blocks as high priority
        equations = re.findall(r'\$\$(.*?)\$\$|\$(.*?)\$', content, re.DOTALL)
        code_blocks = re.findall(r'```(?:python)?\s*(.*?)```', content, re.DOTALL)
        
        # Use default patterns if phase not in key_patterns
        patterns = key_patterns.get(phase_name, ["finding", "approach", "result", "method"])
        
        # Extract paragraphs containing key patterns
        insights = []
        
        # First add any equations or code blocks (high priority)
        if equations:
            for eq in equations[:2]:  # Take up to 2 equations
                eq_content = eq[0] if eq[0] else eq[1]
                if eq_content:
                    insights.append(f"Mathematical formulation: ${eq_content}$")
        
        if code_blocks:
            for code in code_blocks[:1]:  # Take up to 1 code block
                if len(code) > 300:
                    insights.append(f"Code implementation: ```python\n{code[:300]}...\n```")
                else:
                    insights.append(f"Code implementation: ```python\n{code}\n```")
        
        # Then add paragraphs with key patterns
        for paragraph in paragraphs:
            if any(pattern.lower() in paragraph.lower() for pattern in patterns):
                # Trim to reasonable length if needed
                if len(paragraph) > 300:
                    paragraph = paragraph[:300] + "..."
                insights.append(paragraph)
        
        # If no insights found with patterns, take first paragraph and another mid-document
        if not insights and paragraphs:
            insights.append(paragraphs[0])
            if len(paragraphs) > 2:
                mid_idx = len(paragraphs) // 2
                insights.append(paragraphs[mid_idx])
        
        return insights[:3]  # Return at most 3 insights
    
    def _update_research_plan(self, phase_name: str, content: str):
        """Update the research plan based on new insights.
        
        Args:
            phase_name: The current phase name
            content: The content to extract insights from
        """
        # Extract key insights
        insights = self._extract_key_insights(content, phase_name)
        
        # Update the research plan phases if this is a new phase
        if phase_name not in [p.get("name") for p in self.research_plan["phases"]]:
            self.research_plan["phases"].append({
                "name": phase_name,
                "status": "completed",
                "insights": insights,
                "timestamp": str(datetime.datetime.now())
            })
        else:
            # Update existing phase with new insights
            for phase in self.research_plan["phases"]:
                if phase["name"] == phase_name:
                    phase["insights"] = insights
                    phase["status"] = "completed"
                    phase["timestamp"] = str(datetime.datetime.now())
        
        # Update evolving outline based on phase
        if phase_name == "literature-review":
            self.research_plan["evolving_outline"]["background"] = "Updated based on literature review"
            self.research_plan["key_discoveries"].extend(insights)
            
            # Look for Active Inference or POMDP related frameworks to update
            for insight in insights:
                if any(term in insight for term in ["VFE", "EFE", "Variational Free Energy", "Expected Free Energy"]):
                    self.research_plan["framework_elements"].append(insight)
        
        elif phase_name == "plan-formulation":
            self.research_plan["evolving_outline"]["methodology"] = "Updated based on planning phase"
            self.research_plan["current_direction"] = insights[0] if insights else ""
            
            # Extract detailed mathematical framework if present
            math_expressions = re.findall(r'\$(.*?)\$', content, re.DOTALL)
            if math_expressions:
                self.research_plan["framework_elements"].extend([f"${expr}$" for expr in math_expressions[:3]])
        
        elif phase_name == "data-preparation":
            self.research_plan["evolving_outline"]["data_approach"] = "Updated based on data preparation"
            
            # Look for transition and observation models
            for insight in insights:
                if "transition" in insight.lower():
                    self.research_plan["evolving_outline"]["transition_model"] = insight[:100]
                if "observation" in insight.lower():
                    self.research_plan["evolving_outline"]["observation_model"] = insight[:100]
        
        elif phase_name == "code-implementation":
            self.research_plan["evolving_outline"]["implementation"] = "Updated based on code implementation"
            
            # Extract code structure information
            code_blocks = re.findall(r'```(?:python)?\s*(.*?)```', content, re.DOTALL)
            if code_blocks:
                # Extract class and function names from the code
                class_pattern = r'class\s+(\w+)'
                function_pattern = r'def\s+(\w+)'
                
                classes = []
                functions = []
                
                for block in code_blocks:
                    classes.extend(re.findall(class_pattern, block))
                    functions.extend(re.findall(function_pattern, block))
                
                if classes:
                    self.research_plan["evolving_outline"]["code_classes"] = classes[:5]
                if functions:
                    self.research_plan["evolving_outline"]["code_functions"] = functions[:5]
        
        elif phase_name == "running-experiments":
            self.research_plan["evolving_outline"]["experiments"] = "Updated based on experiment results"
            
            # Look for experimental conditions and results
            for insight in insights:
                if any(term in insight.lower() for term in ["experiment", "simulation", "result", "performance"]):
                    self.research_plan["key_discoveries"].append(insight)
        
        elif phase_name == "results-interpretation":
            self.research_plan["evolving_outline"]["results"] = "Updated based on results interpretation"
            self.research_plan["key_discoveries"].extend(insights)
        
        elif phase_name == "report-writing":
            self.research_plan["evolving_outline"]["conclusion"] = "Updated based on report writing"
        
        # Update timestamp
        self.research_plan["last_updated"] = str(datetime.datetime.now())
        
        # Save the updated plan
        plan_file = os.path.join(self.output_dir, "research_plan.json")
        with open(plan_file, "w") as f:
            json.dump(self.research_plan, f, indent=2)
        
        logger.info(f"Updated research plan with insights from {phase_name} phase")
    
    def _get_shared_knowledge_for_phase(self, phase_name: str) -> str:
        """Get shared knowledge for a specific phase.
        
        Args:
            phase_name: Name of the phase to get knowledge for
            
        Returns:
            str: Formatted shared knowledge for the phase
        """
        # Collect relevant knowledge from previous phases
        relevant_knowledge = []
        
        # Map of which previous phases are most relevant to the current phase
        relevance_map = {
            "plan-formulation": ["literature-review"],
            "data-preparation": ["literature-review", "plan-formulation"],
            "code-implementation": ["plan-formulation", "data-preparation"],
            "running-experiments": ["code-implementation", "data-preparation"],
            "results-interpretation": ["running-experiments", "code-implementation"],
            "report-writing": ["results-interpretation", "running-experiments", "literature-review"]
        }
        
        # Add research plan summary
        research_plan_summary = self._get_research_plan_summary(phase_name)
        
        # Get relevant phases for the current phase
        relevant_phases = relevance_map.get(phase_name, [])
        
        # Extract knowledge from relevant phases
        for prev_phase in relevant_phases:
            if prev_phase in self.shared_knowledge:
                # For each phase, prioritize different types of insights
                phase_insights = []
                
                for entry in self.shared_knowledge[prev_phase]:
                    # Extract insights based on the source and phase
                    insights = self._extract_key_insights(entry["content"], prev_phase)
                    source = entry["source"]
                    
                    if insights:
                        phase_insights.append(f"FROM {prev_phase.upper()} (by {source}):\n\n" + "\n\n".join(insights))
                
                if phase_insights:
                    relevant_knowledge.extend(phase_insights[:2])  # Limit to 2 insights per phase
        
        # Format the knowledge as a string
        combined_knowledge = ""
        
        if research_plan_summary:
            combined_knowledge += "CURRENT RESEARCH PLAN:\n\n" + research_plan_summary + "\n\n"
            
        if relevant_knowledge:
            combined_knowledge += "RELEVANT KNOWLEDGE FROM PREVIOUS PHASES:\n\n" + "\n\n".join(relevant_knowledge)
            
        return combined_knowledge
    
    def _get_research_plan_summary(self, current_phase: str) -> str:
        """Generate a summary of the current research plan focused on needs for the current phase.
        
        Args:
            current_phase: The current phase name
            
        Returns:
            str: Summary of the research plan relevant to the current phase
        """
        if not self.research_plan["phases"]:
            # Initial research plan with more details about the POMDP parameters
            parameters = self.research_plan.get("parameters", {})
            summary_parts = [
                f"RESEARCH TOPIC: {self.research_plan['topic']}",
                "MODEL PARAMETERS:",
                f"- Control states: {parameters.get('control_states', '3')} (cool, nothing, heat)",
                f"- Latent states: {parameters.get('latent_states', '5')} (room temperature states)",
                f"- Observation levels: {parameters.get('observation_levels', '10')} (cold to hot)"
            ]
            
            # Add framework elements if available
            if self.research_plan.get("framework_elements"):
                summary_parts.append("KEY COMPONENTS:")
                for element in self.research_plan["framework_elements"]:
                    summary_parts.append(f"- {element}")
            
            return "\n".join(summary_parts)
        
        # Start with topic and parameters
        parameters = self.research_plan.get("parameters", {})
        summary_parts = [
            f"RESEARCH TOPIC: {self.research_plan['topic']}",
            "MODEL PARAMETERS:",
            f"- Control states: {parameters.get('control_states', '3')} (cool, nothing, heat)",
            f"- Latent states: {parameters.get('latent_states', '5')} (room temperature states)",
            f"- Observation levels: {parameters.get('observation_levels', '10')} (cold to hot)"
        ]
        
        # Add completed phases
        completed_phases = [p["name"] for p in self.research_plan["phases"] if p["status"] == "completed"]
        if completed_phases:
            summary_parts.append(f"COMPLETED PHASES: {', '.join(completed_phases)}")
        
        # Add key discoveries if available
        if self.research_plan["key_discoveries"]:
            discoveries = self.research_plan["key_discoveries"][-3:]  # Get last 3 discoveries
            summary_parts.append("KEY DISCOVERIES:")
            for discovery in discoveries:
                summary_parts.append(f"- {discovery}")
        
        # Add framework elements if available
        if self.research_plan.get("framework_elements"):
            summary_parts.append("MATHEMATICAL FRAMEWORK:")
            for element in self.research_plan["framework_elements"][-3:]:  # Most recent 3 elements
                summary_parts.append(f"- {element}")
        
        # Add current direction if available
        if self.research_plan["current_direction"]:
            summary_parts.append(f"CURRENT DIRECTION: {self.research_plan['current_direction']}")
        
        # Add phase-specific focus based on the research parameters
        if current_phase == "literature-review":
            summary_parts.append("FOCUS ON: Identifying key papers on Active Inference and POMDPs, particularly those involving thermal systems or similar control problems")
        elif current_phase == "plan-formulation":
            summary_parts.append("FOCUS ON: Formulating the mathematical framework for the POMDP with Active Inference, including Variational Free Energy and Expected Free Energy")
        elif current_phase == "data-preparation":
            summary_parts.append(f"FOCUS ON: Defining transition dynamics and observation model for the thermal system with {parameters.get('control_states', '3')} control states and {parameters.get('latent_states', '5')} latent states")
        elif current_phase == "code-implementation":
            summary_parts.append("FOCUS ON: Implementing the POMDP model with Active Inference components, including belief updating and policy selection mechanisms")
        elif current_phase == "running-experiments":
            summary_parts.append("FOCUS ON: Running simulations of the thermal control system under different conditions to evaluate performance")
        elif current_phase == "results-interpretation":
            summary_parts.append("FOCUS ON: Analyzing how well Active Inference performs for thermal homeostasis compared to alternative approaches")
        elif current_phase == "report-writing":
            summary_parts.append("FOCUS ON: Synthesizing all findings into a comprehensive paper with clear mathematical formulations and experimental results")
        
        return "\n".join(summary_parts)
    
    def complete_phase_with_discourse(self, phase_name: str, task_notes: str) -> Dict[str, Any]:
        """Complete a phase with discourse between agents.
        
        Args:
            phase_name: Name of the phase to complete
            task_notes: Task notes for the phase
            
        Returns:
            Dict[str, Any]: Results including phase output and agent contributions
        """
        logger.info(f"Starting multi-agent discourse for phase: {phase_name}")
        
        # Get relevant knowledge from previous phases
        shared_knowledge = self._get_shared_knowledge_for_phase(phase_name)
        
        # Enhanced task notes to explicitly reference research parameters
        parameters = self.research_plan.get("parameters", {})
        explicit_parameters = f"""
This research focuses on a POMDP for thermal homeostasis with:
- {parameters.get('control_states', '3')} control states (cool, nothing, heat)
- {parameters.get('latent_states', '5')} latent states for room temperature 
- {parameters.get('observation_levels', '10')} discrete observation levels (cold to hot)

The implementation should use Variational Free Energy for state estimation and Expected Free Energy for action selection.
"""
        
        # Enhanced task notes with explicit parameters and shared knowledge
        enhanced_task_notes = f"{task_notes}\n\n{explicit_parameters}"
        if shared_knowledge:
            enhanced_task_notes += f"\n\n{shared_knowledge}"
        
        # Step 1: Initial work by the professor agent
        logger.info(f"Professor agent starting initial work on {phase_name}")
        professor_output = self.agents["professor"].complete_phase(phase_name, enhanced_task_notes)
        
        # Save professor's output to a timestamped file for better logging
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = os.path.join(self.output_dir, "professor_outputs", f"{phase_name}_{timestamp}.md")
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        with open(output_filename, "w") as f:
            f.write(professor_output)
        logger.info(f"Saved professor's initial output to {output_filename}")
        
        # Update shared knowledge
        self._update_shared_knowledge(phase_name, professor_output, "professor")
        
        # Step 2: Engineer agent contributes technical perspective
        logger.info(f"Engineer agent contributing to {phase_name}")
        engineer_prompt = f"""
        The professor has provided the following output for the {phase_name} phase:
        
        {professor_output[:2000]}...
        
        Please review this from a technical implementation perspective and provide:
        1. Technical improvements or clarifications
        2. Any implementation considerations
        3. Suggestions for computational efficiency or technical robustness
        4. Specific code improvements or implementations if applicable
        
        {explicit_parameters}
        
        {shared_knowledge}
        """
        engineer_contribution = self.agents["engineer"].complete_phase(
            phase_name, 
            engineer_prompt
        )
        
        # Update shared knowledge
        self._update_shared_knowledge(phase_name, engineer_contribution, "engineer")
        
        # Step 3: Critic agent reviews and provides feedback
        logger.info(f"Critic agent reviewing {phase_name} outputs")
        critic_prompt = f"""
        For the {phase_name} phase, two experts have contributed:
        
        PROFESSOR:
        {professor_output[:1500]}...
        
        ENGINEER:
        {engineer_contribution[:1500]}...
        
        Please provide a critical review focusing on:
        1. Strengths and weaknesses of both contributions
        2. Areas that need more development or clarification
        3. Suggestions for improvement
        4. Overall assessment of quality and completeness
        5. Specific recommendations for addressing any identified issues
        
        {explicit_parameters}
        
        {shared_knowledge}
        """
        critic_feedback = self.agents["critic"].complete_phase(
            phase_name, 
            critic_prompt
        )
        
        # Update shared knowledge
        self._update_shared_knowledge(phase_name, critic_feedback, "critic")
        
        # Step 4: Professor agent integrates feedback and contributions
        logger.info(f"Professor agent integrating contributions for {phase_name}")
        integration_prompt = f"""
        You have received feedback and contributions for your {phase_name} work:
        
        ENGINEER CONTRIBUTION:
        {engineer_contribution[:1500]}...
        
        CRITIC FEEDBACK:
        {critic_feedback[:1500]}...
        
        Please review this feedback and create an improved, integrated final output for the {phase_name} phase.
        Incorporate the valuable insights from both the engineer and critic.
        
        Your integrated output should be comprehensive, technically sound, and address all the critical points raised.
        If code was provided or suggested, ensure it is properly implemented and documented in your integrated response.
        
        {explicit_parameters}
        
        {shared_knowledge}
        """
        final_output = self.agents["professor"].complete_phase(
            f"{phase_name}-integration", 
            integration_prompt
        )
        
        # Save professor's integration to a timestamped file for better logging
        integration_filename = os.path.join(self.output_dir, "professor_outputs", f"{phase_name}-integration_{timestamp}.md")
        with open(integration_filename, "w") as f:
            f.write(final_output)
        logger.info(f"Saved professor's integration to {integration_filename}")
        
        # Update shared knowledge with the final integrated output
        self._update_shared_knowledge(f"{phase_name}-integration", final_output, "professor-integration")
        
        # Update the research plan with the integrated output
        self._update_research_plan(phase_name, final_output)
        
        # Record the interaction
        interaction = {
            "phase": phase_name,
            "task_notes": task_notes,
            "shared_knowledge": shared_knowledge,
            "professor_initial": professor_output,
            "engineer_contribution": engineer_contribution,
            "critic_feedback": critic_feedback,
            "integrated_output": final_output
        }
        self.interactions.append(interaction)
        
        # Save the interaction to file
        interaction_file = os.path.join(self.output_dir, f"{phase_name}_interaction.json")
        with open(interaction_file, 'w') as f:
            json.dump(interaction, f, indent=2)
        
        # Save agent states
        self.save_states()
        
        logger.info(f"Completed multi-agent discourse for phase: {phase_name}")
        
        return {
            "output": final_output,
            "discourse": interaction
        }
    
    def generate_discourse_summary(self) -> str:
        """Generate a summary of all agent interactions.
        
        Returns:
            str: Summary of agent interactions
        """
        if not self.interactions:
            return "No agent interactions recorded yet."
        
        summary = "# Multi-Agent Discourse Summary\n\n"
        
        # Add research plan summary
        summary += "## Current Research Plan\n\n"
        summary += f"**Research Topic:** {self.research_plan['topic']}\n\n"
        
        # Add model parameters
        parameters = self.research_plan.get("parameters", {})
        summary += "**Model Parameters:**\n\n"
        summary += f"- Control states: {parameters.get('control_states', '3')} (cool, nothing, heat)\n"
        summary += f"- Latent states: {parameters.get('latent_states', '5')} (room temperature states)\n"
        summary += f"- Observation levels: {parameters.get('observation_levels', '10')} (cold to hot)\n\n"
        
        # Add phases and their status
        summary += "**Completed Phases:**\n\n"
        for phase in self.research_plan["phases"]:
            summary += f"- {phase['name'].replace('-', ' ').title()}: {phase['status']}\n"
            if phase.get("insights"):
                summary += "  - Key insights:\n"
                for insight in phase.get("insights", []):
                    summary += f"    - {insight[:100]}...\n"
        
        # Add mathematical framework elements
        if self.research_plan.get("framework_elements"):
            summary += "\n**Mathematical Framework:**\n\n"
            for element in self.research_plan["framework_elements"]:
                summary += f"- {element}\n"
        
        summary += "\n**Current Research Direction:**\n\n"
        summary += self.research_plan["current_direction"] or "Direction pending further research."
        
        summary += "\n\n## Agent Interactions by Phase\n\n"
        
        for i, interaction in enumerate(self.interactions):
            phase = interaction["phase"]
            summary += f"## Phase: {phase}\n\n"
            
            # Add professor initial contribution
            summary += "### Initial Research Direction (Professor)\n\n"
            summary += interaction["professor_initial"][:2000] + "...\n\n"
            
            # Add engineer contribution
            summary += "### Technical Perspective (Engineer)\n\n"
            summary += interaction["engineer_contribution"][:2000] + "...\n\n"
            
            # Add critic feedback
            summary += "### Critical Assessment (Critic)\n\n"
            summary += interaction["critic_feedback"][:2000] + "...\n\n"
            
            # Add integrated output
            summary += "### Integrated Solution (Professor)\n\n"
            summary += interaction["integrated_output"][:2000] + "...\n\n"
            
            # Add separator between phases
            if i < len(self.interactions) - 1:
                summary += "---\n\n"
        
        return summary 