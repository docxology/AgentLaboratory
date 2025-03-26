"""
Agents package for the Agent Laboratory.

This package provides various agent implementations for the Agent Laboratory.
"""

from agent_lab.agents.base_agent import BaseAgent
from agent_lab.agents.professor_agent import ProfessorAgent

# Add other agents as they are implemented
# from agent_lab.agents.postdoc_agent import PostdocAgent
# from agent_lab.agents.mlengineer_agent import MLEngineerAgent
# from agent_lab.agents.swengineer_agent import SWEngineerAgent
# from agent_lab.agents.phdstudent_agent import PhDStudentAgent
# from agent_lab.agents.reviewer_agent import ReviewerAgent

__all__ = [
    "BaseAgent",
    "ProfessorAgent",
    # Add other agents as they are implemented
    # "PostdocAgent",
    # "MLEngineerAgent",
    # "SWEngineerAgent",
    # "PhDStudentAgent",
    # "ReviewerAgent",
]
