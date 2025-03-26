"""HTML rendering module for visualizing agent interactions."""

import os
import json
import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import jinja2


def render_agent_interactions(
    output_dir: str,
    experiment_title: str,
    agents_data: Dict[str, List[Dict[str, Any]]],
    experiment_metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Render agent interactions as an HTML page.
    
    Args:
        output_dir: Directory to save the HTML file.
        experiment_title: Title of the experiment.
        agents_data: Dictionary mapping agent names to their message lists.
        experiment_metadata: Additional metadata about the experiment.
    
    Returns:
        Path to the generated HTML file.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up Jinja2 environment
    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir))
    template = env.get_template("agent_interaction.html")
    
    # Process agent data
    agents = []
    for agent_name, messages in agents_data.items():
        agents.append({
            "name": agent_name,
            "messages": messages
        })
    
    # Get current date and time
    experiment_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Default metadata if not provided
    if experiment_metadata is None:
        experiment_metadata = {}
    
    # Generate JavaScript for visualization
    visualization_js = _generate_visualization_js(agents)
    
    # Render the template
    html_content = template.render(
        experiment_title=experiment_title,
        experiment_date=experiment_date,
        agents=agents,
        total_steps=experiment_metadata.get("total_steps", len(max(agents_data.values(), key=len)) if agents_data else 0),
        runtime=experiment_metadata.get("runtime", "N/A"),
        agent_count=len(agents),
        status=experiment_metadata.get("status", "Completed"),
        visualization_js=visualization_js
    )
    
    # Save the HTML file
    output_file = os.path.join(output_dir, "agent_interactions.html")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    return output_file


def _generate_visualization_js(agents: List[Dict[str, Any]]) -> str:
    """
    Generate JavaScript code for visualizing agent interactions.
    
    Args:
        agents: List of agent data dictionaries.
    
    Returns:
        JavaScript code as a string.
    """
    # Example: Generate a simple visualization using D3.js
    # In a real implementation, this would create more sophisticated visualizations
    js_code = """
    // This is a placeholder for the visualization JavaScript
    // In a real implementation, this would be replaced with D3.js or another visualization library
    document.addEventListener('DOMContentLoaded', function() {
        const diagram = document.querySelector('.interaction-diagram');
        
        // Add a placeholder message
        const placeholder = document.createElement('div');
        placeholder.style.padding = '20px';
        placeholder.style.textAlign = 'center';
        placeholder.textContent = 'Agent interaction visualization would be displayed here';
        diagram.appendChild(placeholder);
        
        // In a full implementation, we would:
        // 1. Create a force-directed graph, sequence diagram, or other visualization
        // 2. Show message flows between agents
        // 3. Allow interaction such as filtering or zooming
    });
    """
    
    return js_code 