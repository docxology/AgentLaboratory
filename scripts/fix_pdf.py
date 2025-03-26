#!/usr/bin/env python

import os
import sys
import json
import logging
import re
import glob
import datetime
from pathlib import Path

# Add project root to path to ensure imports work
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir))
sys.path.insert(0, project_root)

from agent_lab.io.executor.latex_executor import compile_latex
from agent_lab.visualization.pdf_generator import generate_report_pdf

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('fix_pdf')

def find_most_recent_workflow():
    """Find the most recent workflow directory."""
    output_dirs = glob.glob('outputs/*_research')
    if not output_dirs:
        logger.error("No workflow directories found")
        return None
    
    # Sort by modification time
    output_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    logger.info(f"Found workflow directory: {output_dirs[0]}")
    return output_dirs[0]

def validate_phase_content(state, phase_name):
    """Validate that a phase's output exists and has reasonable content."""
    if not state or "phases" not in state:
        logger.warning(f"No phases found in state")
        return False
    
    phases = state.get("phases", {})
    if phase_name not in phases:
        logger.warning(f"Phase {phase_name} not found in state")
        return False
    
    phase_output = phases.get(phase_name, {}).get("output", "")
    if not phase_output or len(phase_output) < 100:
        logger.warning(f"Phase {phase_name} has no or minimal output: {len(phase_output) if phase_output else 0} chars")
        return False
    
    logger.info(f"Phase {phase_name} validated with {len(phase_output)} chars")
    return True

def load_discourse_from_file(artifacts_dir):
    """Load discourse content from the agent_discourse.md file."""
    discourse_file = os.path.join(artifacts_dir, "agent_discourse.md")
    if not os.path.exists(discourse_file):
        logger.warning(f"Discourse file not found: {discourse_file}")
        return None
    
    try:
        with open(discourse_file, 'r', encoding='utf-8') as f:
            content = f.read()
            logger.info(f"Loaded discourse content: {len(content)} chars")
            return content
    except Exception as e:
        logger.error(f"Error loading discourse file: {e}")
        return None

def validate_and_fix_latex(latex_file):
    """Validate the LaTeX file and fix common issues."""
    if not os.path.exists(latex_file):
        logger.error(f"LaTeX file not found: {latex_file}")
        return False
    
    try:
        with open(latex_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fix nested itemize environments - this is a common issue
        content = re.sub(r'\\begin{itemize}\s*\\begin{itemize}', r'\\begin{itemize}', content)
        content = re.sub(r'\\end{itemize}\s*\\end{itemize}', r'\\end{itemize}', content)
        
        # Fix unclosed environments by checking balanced begin/end pairs
        for env in ['itemize', 'enumerate', 'figure', 'table']:
            begin_count = content.count(f'\\begin{{{env}}}')
            end_count = content.count(f'\\end{{{env}}}')
            
            if begin_count > end_count:
                logger.warning(f"Found {begin_count} begin tags but only {end_count} end tags for {env}")
                # Add missing end tags at the end of the content
                content += '\n' + '\\end{' + env + '}' * (begin_count - end_count)
            elif end_count > begin_count:
                logger.warning(f"Found {end_count} end tags but only {begin_count} begin tags for {env}")
                # Remove excess end tags (this is a simplistic approach)
                for _ in range(end_count - begin_count):
                    last_end = content.rfind(f'\\end{{{env}}}')
                    if last_end >= 0:
                        content = content[:last_end] + content[last_end + len(f'\\end{{{env}}}'):]
        
        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"LaTeX file validated and fixed: {latex_file}")
        return True
    except Exception as e:
        logger.error(f"Error validating LaTeX file: {e}")
        return False

def regenerate_pdf(workflow_dir):
    """Regenerate the PDF report for a workflow."""
    # Load workflow state
    state_file = os.path.join(workflow_dir, "state.json")
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            logger.info(f"Loaded workflow state: {len(state)} keys")
        except Exception as e:
            logger.error(f"Error loading state file: {e}")
            state = {}
    else:
        logger.warning(f"State file not found: {state_file}")
        state = {}
    
    # Validate key phases
    validate_phase_content(state, "research-problem-analysis")
    validate_phase_content(state, "literature-review")
    validate_phase_content(state, "implementation")
    
    # Load discourse content
    artifacts_dir = os.path.join(workflow_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)  # Ensure artifacts directory exists
    discourse_content = load_discourse_from_file(artifacts_dir)
    
    # Create experiment data structure
    experiment_data = {
        "research_topic": "POMDP with Active Inference for Thermal Homeostasis",
        "steps": [
            {"phase": "research-problem-analysis"},
            {"phase": "literature-review"},
            {"phase": "implementation"},
            {"phase": "experiments"},
            {"phase": "report-writing"}
        ]
    }
    
    # Create agent_dialogs structure
    agent_dialogs = [
        {
            "phase": "report-writing",
            "dialog": [
                {"role": "system", "content": "System message"},
                {"role": "user", "content": "User message"},
                {"role": "assistant", "content": discourse_content or "No discourse content available"}
            ]
        }
    ]
    
    # Generate the report
    output_file = os.path.join(workflow_dir, "artifacts", f"report_{os.path.basename(workflow_dir)}")
    try:
        latex_file = generate_report_pdf(
            experiment_data=experiment_data,
            agent_dialogs=agent_dialogs,
            output_file=output_file,
            artifacts_dir=artifacts_dir
        )
        
        # Validate and fix the LaTeX file before compilation
        if latex_file.endswith('.pdf'):
            latex_file = latex_file.replace('.pdf', '.tex')
        
        if os.path.exists(latex_file):
            logger.info(f"LaTeX file found: {latex_file}")
            validate_and_fix_latex(latex_file)
            
            # Compile the fixed LaTeX file
            pdf_file = latex_file.replace('.tex', '.pdf')
            compile_latex(latex_file, pdf_file)
            
            if os.path.exists(pdf_file):
                logger.info(f"PDF report generated: {pdf_file}")
                logger.info(f"PDF file size: {os.path.getsize(pdf_file)} bytes")
            else:
                logger.warning(f"PDF compilation completed but file not found: {pdf_file}")
        else:
            logger.warning(f"LaTeX file not found: {latex_file}")
        
        return True
    except Exception as e:
        logger.error(f"Error generating PDF: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main function."""
    # Find the most recent workflow
    workflow_dir = find_most_recent_workflow()
    if not workflow_dir:
        return
    
    # Regenerate the PDF
    regenerate_pdf(workflow_dir)

if __name__ == '__main__':
    main() 