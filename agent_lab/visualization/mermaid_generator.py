"""Module for generating Mermaid diagrams from agent interactions."""

import os
from typing import Dict, List, Any, Optional


def generate_sequence_diagram(
    output_dir: str,
    agent_interactions: Dict[str, List[Dict[str, Any]]],
    output_filename: Optional[str] = None
) -> str:
    """
    Generate a Mermaid sequence diagram from agent interactions.
    
    Args:
        output_dir: Directory to save the Mermaid file.
        agent_interactions: Dictionary mapping agent names to their message lists.
        output_filename: Name of the output file (default: "sequence_diagram.md").
    
    Returns:
        Path to the generated Mermaid file.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the output filename
    if output_filename is None:
        output_filename = "sequence_diagram.md"
    output_path = os.path.join(output_dir, output_filename)
    
    # Start the Mermaid sequence diagram
    mermaid_code = ["```mermaid", "sequenceDiagram"]
    
    # Define the participants (agents)
    for agent_name in agent_interactions.keys():
        mermaid_code.append(f"    participant {agent_name}")
    
    # Add a line break
    mermaid_code.append("")
    
    # Add the interactions
    message_counter = 1
    for step in range(100):  # Limit to 100 steps maximum
        step_complete = True
        
        for sender, messages in agent_interactions.items():
            # Find messages that correspond to this step
            # In a real implementation, this would need proper ordering logic
            if step < len(messages):
                step_complete = False
                message = messages[step]
                
                # Determine the recipient (in a real implementation, this would be more sophisticated)
                # For now, we'll use a simple heuristic: the next agent in the list
                agent_names = list(agent_interactions.keys())
                sender_idx = agent_names.index(sender)
                recipient = agent_names[(sender_idx + 1) % len(agent_names)]
                
                # Create a label for the message (truncated if too long)
                content = message.get('content', '')
                label = content[:50] + "..." if len(content) > 50 else content
                label = label.replace("\n", " ").replace("\"", "'")
                
                # Add the message to the diagram
                mermaid_code.append(f"    {sender}->>+{recipient}: {label}")
                mermaid_code.append(f"    {recipient}-->>-{sender}: Response {message_counter}")
                message_counter += 1
        
        # If no messages were found for this step, break out of the loop
        if step_complete:
            break
    
    # End the Mermaid diagram
    mermaid_code.append("```")
    
    # Write the Mermaid code to a file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(mermaid_code))
    
    return output_path


def generate_class_diagram(
    output_dir: str,
    class_definitions: Dict[str, Dict[str, Any]],
    output_filename: Optional[str] = None
) -> str:
    """
    Generate a Mermaid class diagram from class definitions.
    
    Args:
        output_dir: Directory to save the Mermaid file.
        class_definitions: Dictionary mapping class names to their attributes and methods.
        output_filename: Name of the output file (default: "class_diagram.md").
    
    Returns:
        Path to the generated Mermaid file.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the output filename
    if output_filename is None:
        output_filename = "class_diagram.md"
    output_path = os.path.join(output_dir, output_filename)
    
    # Start the Mermaid class diagram
    mermaid_code = ["```mermaid", "classDiagram"]
    
    # Add the classes
    for class_name, definition in class_definitions.items():
        # Add the class
        mermaid_code.append(f"    class {class_name} {{")
        
        # Add attributes
        for attr in definition.get('attributes', []):
            mermaid_code.append(f"        +{attr}")
        
        # Add methods
        for method in definition.get('methods', []):
            mermaid_code.append(f"        +{method}()")
        
        # Close the class
        mermaid_code.append("    }")
    
    # Add the relationships
    for class_name, definition in class_definitions.items():
        # Add inheritance relationships
        for parent in definition.get('inherits_from', []):
            mermaid_code.append(f"    {parent} <|-- {class_name}")
        
        # Add composition relationships
        for composed in definition.get('composed_of', []):
            mermaid_code.append(f"    {class_name} *-- {composed}")
        
        # Add aggregation relationships
        for aggregated in definition.get('aggregates', []):
            mermaid_code.append(f"    {class_name} o-- {aggregated}")
    
    # End the Mermaid diagram
    mermaid_code.append("```")
    
    # Write the Mermaid code to a file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(mermaid_code))
    
    return output_path


def generate_flowchart(
    output_dir: str,
    workflow_steps: List[Dict[str, Any]],
    output_filename: Optional[str] = None
) -> str:
    """
    Generate a Mermaid flowchart from workflow steps.
    
    Args:
        output_dir: Directory to save the Mermaid file.
        workflow_steps: List of workflow steps with their connections.
        output_filename: Name of the output file (default: "flowchart.md").
    
    Returns:
        Path to the generated Mermaid file.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the output filename
    if output_filename is None:
        output_filename = "flowchart.md"
    output_path = os.path.join(output_dir, output_filename)
    
    # Start the Mermaid flowchart
    mermaid_code = ["```mermaid", "flowchart TD"]
    
    # Add the nodes
    for i, step in enumerate(workflow_steps):
        node_id = f"N{i}"
        label = step.get('label', f"Step {i+1}")
        mermaid_code.append(f"    {node_id}[{label}]")
    
    # Add the connections
    for i, step in enumerate(workflow_steps):
        node_id = f"N{i}"
        for connection in step.get('connections', []):
            target_id = f"N{connection}"
            mermaid_code.append(f"    {node_id} --> {target_id}")
    
    # End the Mermaid diagram
    mermaid_code.append("```")
    
    # Write the Mermaid code to a file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(mermaid_code))
    
    return output_path 