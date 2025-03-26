"""
Experiment visualizer for the Agent Laboratory.

This module provides visualization and reporting capabilities for research experiments,
including agent interaction graphs, metrics dashboards, and HTML reports.
"""

import os
import json
import datetime
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import shutil
import base64
from jinja2 import Environment, FileSystemLoader, select_autoescape


class ExperimentVisualizer:
    """Visualizer for research experiments."""
    
    def __init__(
        self,
        output_dir: str,
        experiment_dir: str,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the experiment visualizer.
        
        Args:
            output_dir: Base directory for outputs
            experiment_dir: Directory for the current experiment
            logger: Logger instance
        """
        self.output_dir = output_dir
        self.experiment_dir = experiment_dir
        self.logger = logger or logging.getLogger(__name__)
        
        # Create visualization directory
        self.vis_dir = os.path.join(experiment_dir, "visualizations")
        os.makedirs(self.vis_dir, exist_ok=True)
        
        # Create reports directory
        self.reports_dir = os.path.join(experiment_dir, "reports")
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Setup Jinja2 environment for HTML templates
        self.template_dir = os.path.join(os.path.dirname(__file__), "templates")
        
        # If templates directory doesn't exist, create it with default templates
        if not os.path.exists(self.template_dir):
            os.makedirs(self.template_dir, exist_ok=True)
            self._create_default_templates()
        
        self.jinja_env = Environment(
            loader=FileSystemLoader(self.template_dir),
            autoescape=select_autoescape(['html', 'xml'])
        )
    
    def _create_default_templates(self) -> None:
        """Create default templates if they don't exist."""
        # Create base template
        base_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Agent Laboratory Report{% endblock %}</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f9f9f9;
        }
        h1, h2, h3, h4 {
            color: #2c3e50;
            margin-top: 1.5em;
        }
        h1 {
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        .container {
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .metric-card {
            background-color: #f8f9fa;
            border-radius: 4px;
            padding: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #3498db;
        }
        .metric-name {
            color: #7f8c8d;
            font-size: 14px;
        }
        .artifact {
            margin: 10px 0;
            padding: 10px;
            background-color: #f8f9fa;
            border-left: 4px solid #3498db;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        th, td {
            padding: 12px 15px;
            border-bottom: 1px solid #ddd;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        .phase-section {
            margin-top: 30px;
        }
        .dialog {
            margin: 10px 0;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 5px;
        }
        .dialog-system {
            background-color: #ecf0f1;
            border-left: 4px solid #95a5a6;
        }
        .dialog-user {
            background-color: #e8f4fd;
            border-left: 4px solid #3498db;
        }
        .dialog-agent {
            background-color: #f9f2e7;
            border-left: 4px solid #e67e22;
        }
        .dialog-human {
            background-color: #eafaf1;
            border-left: 4px solid #2ecc71;
        }
        .dialog-header {
            font-weight: bold;
            margin-bottom: 5px;
        }
        .code {
            font-family: monospace;
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
        }
        .summary-box {
            background-color: #e8f4fd;
            padding: 15px;
            border-radius: 4px;
            margin: 20px 0;
        }
        footer {
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            color: #7f8c8d;
            font-size: 14px;
            text-align: center;
        }
    </style>
</head>
<body>
    <header>
        <h1>{% block header %}Agent Laboratory Report{% endblock %}</h1>
    </header>
    
    <main>
        {% block content %}{% endblock %}
    </main>
    
    <footer>
        <p>Generated by Agent Laboratory on {{ generation_time }}</p>
    </footer>
</body>
</html>
"""
        
        # Create report template
        report_template = """{% extends "base.html" %}

{% block title %}Research Report: {{ research_topic }}{% endblock %}

{% block header %}Research Report: {{ research_topic }}{% endblock %}

{% block content %}
<div class="container">
    <h2>Experiment Summary</h2>
    <div class="summary-box">
        <p><strong>Research Topic:</strong> {{ research_topic }}</p>
        <p><strong>Experiment ID:</strong> {{ experiment_id }}</p>
        <p><strong>Start Time:</strong> {{ start_time }}</p>
        <p><strong>Duration:</strong> {{ duration }} minutes</p>
        <p><strong>Phases Completed:</strong> {{ phases_completed|join(', ') }}</p>
    </div>
    
    <h3>Key Metrics</h3>
    <div class="metrics-grid">
        {% for metric in key_metrics %}
        <div class="metric-card">
            <div class="metric-value">{{ metric.value }}</div>
            <div class="metric-name">{{ metric.name }}</div>
        </div>
        {% endfor %}
    </div>
</div>

{% for phase in phases %}
<div class="container phase-section">
    <h2>Phase: {{ phase.name }}</h2>
    
    <div class="summary-box">
        <p><strong>Duration:</strong> {{ phase.duration }} minutes</p>
        <p><strong>Status:</strong> {{ phase.status }}</p>
        {% if phase.description %}
        <p>{{ phase.description }}</p>
        {% endif %}
    </div>
    
    {% if phase.metrics %}
    <h3>Metrics</h3>
    <table>
        <tr>
            <th>Metric</th>
            <th>Value</th>
        </tr>
        {% for name, value in phase.metrics.items() %}
        <tr>
            <td>{{ name }}</td>
            <td>{{ value }}</td>
        </tr>
        {% endfor %}
    </table>
    {% endif %}
    
    {% if phase.artifacts %}
    <h3>Artifacts</h3>
    {% for artifact in phase.artifacts %}
    <div class="artifact">
        <h4>{{ artifact.name }}</h4>
        <p>{{ artifact.description }}</p>
        {% if artifact.type == 'code' %}
        <div class="code">{{ artifact.content }}</div>
        {% elif artifact.type == 'text' %}
        <p>{{ artifact.content }}</p>
        {% elif artifact.type == 'link' %}
        <p><a href="{{ artifact.path }}" target="_blank">View {{ artifact.name }}</a></p>
        {% endif %}
    </div>
    {% endfor %}
    {% endif %}
    
    {% if phase.dialogs %}
    <h3>Agent Dialogs</h3>
    {% for dialog in phase.dialogs %}
    <div class="dialog dialog-{{ dialog.role }}">
        <div class="dialog-header">
            {{ dialog.agent }} ({{ dialog.role }})
        </div>
        <div class="dialog-content">
            {{ dialog.content }}
        </div>
    </div>
    {% endfor %}
    {% endif %}
</div>
{% endfor %}

{% if conclusion %}
<div class="container">
    <h2>Conclusion</h2>
    <div class="summary-box">
        {{ conclusion }}
    </div>
</div>
{% endif %}
{% endblock %}
"""
        
        # Write templates to files
        with open(os.path.join(self.template_dir, "base.html"), "w") as f:
            f.write(base_template)
        
        with open(os.path.join(self.template_dir, "report.html"), "w") as f:
            f.write(report_template)
    
    def plot_agent_interaction_graph(
        self,
        interactions: List[Dict[str, Any]],
        phase: Optional[str] = None,
        filename: str = "agent_interactions.png"
    ) -> str:
        """
        Plot a graph of agent interactions.
        
        Args:
            interactions: List of interaction dictionaries
            phase: Optional phase name
            filename: Filename for the output image
            
        Returns:
            Path to the saved image
        """
        # Determine output path
        if phase:
            output_dir = os.path.join(self.vis_dir, phase)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, filename)
        else:
            output_path = os.path.join(self.vis_dir, filename)
        
        # Create a simple HTML visualization instead of a graph
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Agent Interactions{f" - {phase}" if phase else ""}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .interaction {{ border: 1px solid #ddd; margin: 10px 0; padding: 10px; border-radius: 5px; }}
        .source {{ font-weight: bold; color: #0066cc; }}
        .target {{ font-weight: bold; color: #cc6600; }}
        .timestamp {{ color: #666; font-size: 0.8em; }}
        .message {{ margin-top: 5px; white-space: pre-wrap; }}
    </style>
</head>
<body>
    <h1>Agent Interactions{f" - {phase}" if phase else ""}</h1>
    <p>Total interactions: {len(interactions)}</p>
    
    <div class="interactions">
"""
        
        # Add interactions
        for i, interaction in enumerate(interactions):
            source = interaction.get("source", "Unknown")
            target = interaction.get("target", "Unknown")
            message = interaction.get("message", "")
            timestamp = interaction.get("timestamp", "")
            phase_name = interaction.get("phase", "")
            
            html_content += f"""
        <div class="interaction">
            <div class="header">
                <span class="source">{source}</span> → 
                <span class="target">{target}</span>
                <span class="timestamp"> ({timestamp})</span>
                {f' <span class="phase">[{phase_name}]</span>' if phase_name else ''}
            </div>
            <div class="message">{message[:200] + "..." if len(message) > 200 else message}</div>
        </div>
"""
        
        html_content += """
    </div>
</body>
</html>
"""
        
        # Write HTML file
        with open(output_path, "w") as f:
            f.write(html_content)
        
        self.logger.info(f"Agent interaction visualization saved to {output_path}")
        return output_path
    
    def create_agent_dialog_summary(
        self,
        dialogs_dir: str,
        phase: Optional[str] = None,
        max_entries: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Create a summary of agent dialogs.
        
        Args:
            dialogs_dir: Directory containing the dialog files
            phase: Optional phase name to filter
            max_entries: Maximum number of dialog entries to include
            
        Returns:
            List of dialog entry dictionaries
        """
        dialog_entries = []
        
        # Determine the base directory
        if phase:
            base_dir = os.path.join(dialogs_dir, phase)
        else:
            base_dir = dialogs_dir
        
        if not os.path.exists(base_dir):
            self.logger.warning(f"Dialog directory not found: {base_dir}")
            return dialog_entries
        
        # Find all dialog files
        for agent_dir in os.listdir(base_dir):
            agent_path = os.path.join(base_dir, agent_dir)
            
            if not os.path.isdir(agent_path):
                continue
            
            dialog_file = os.path.join(agent_path, "dialog.jsonl")
            
            if not os.path.exists(dialog_file):
                continue
            
            # Read dialog entries
            with open(dialog_file, "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        entry["agent"] = agent_dir
                        dialog_entries.append(entry)
                    except json.JSONDecodeError:
                        self.logger.warning(f"Invalid JSON in dialog file: {dialog_file}")
        
        # Sort by timestamp
        dialog_entries.sort(key=lambda x: x.get("timestamp", ""))
        
        # Limit the number of entries
        if len(dialog_entries) > max_entries:
            dialog_entries = dialog_entries[-max_entries:]
        
        return dialog_entries
    
    def create_experiment_report(
        self,
        metrics: Dict[str, Any],
        config: Dict[str, Any],
        dialogs_dir: str,
        artifacts_dir: str,
        phases_completed: List[str],
        start_time: datetime.datetime,
        end_time: Optional[datetime.datetime] = None,
        template_name: str = "report.html",
        filename: str = "experiment_report.html"
    ) -> str:
        """
        Create an HTML report for the experiment.
        
        Args:
            metrics: Dictionary of metrics
            config: Configuration dictionary
            dialogs_dir: Directory containing dialog files
            artifacts_dir: Directory containing artifacts
            phases_completed: List of completed phases
            start_time: Start time of the experiment
            end_time: End time of the experiment (defaults to now)
            template_name: Name of the template to use
            filename: Filename for the output HTML file
            
        Returns:
            Path to the saved HTML file
        """
        # Default end time to now if not provided
        if end_time is None:
            end_time = datetime.datetime.now()
        
        # Calculate duration in minutes
        duration_seconds = (end_time - start_time).total_seconds()
        duration_minutes = duration_seconds / 60
        
        # Prepare template data
        template_data = {
            "research_topic": config.get("research-topic", "Unknown Research Topic"),
            "experiment_id": os.path.basename(self.experiment_dir),
            "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration": f"{duration_minutes:.1f}",
            "phases_completed": phases_completed,
            "generation_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "conclusion": None,
            "key_metrics": [],
            "phases": []
        }
        
        # Add key metrics
        token_usage = sum(
            metrics.get(phase, {}).get("total_tokens", 0)
            for phase in metrics
            if isinstance(metrics[phase], dict)
        )
        
        template_data["key_metrics"] = [
            {"name": "Total Phases", "value": len(phases_completed)},
            {"name": "Token Usage", "value": token_usage},
            {"name": "Duration (min)", "value": f"{duration_minutes:.1f}"}
        ]
        
        # Process each phase
        for phase in phases_completed:
            phase_data = {
                "name": phase,
                "metrics": {},
                "artifacts": [],
                "dialogs": [],
                "duration": 0,
                "status": "Completed",
                "description": ""
            }
            
            # Extract phase metrics
            if phase in metrics:
                phase_metrics = metrics[phase]
                
                # Calculate phase duration
                if isinstance(phase_metrics, dict):
                    if "start_time" in phase_metrics and "end_time" in phase_metrics:
                        try:
                            start = datetime.datetime.fromisoformat(phase_metrics["start_time"])
                            end = datetime.datetime.fromisoformat(phase_metrics["end_time"])
                            duration_sec = (end - start).total_seconds()
                            phase_data["duration"] = f"{duration_sec / 60:.1f}"
                        except (ValueError, TypeError):
                            pass
                    
                    # Extract other metrics
                    for key, value in phase_metrics.items():
                        if key not in ["start_time", "end_time"] and not isinstance(value, dict):
                            phase_data["metrics"][key] = value
            
            # Find artifacts for this phase
            phase_artifacts_dir = os.path.join(artifacts_dir, phase)
            if os.path.exists(phase_artifacts_dir):
                for root, _, files in os.walk(phase_artifacts_dir):
                    for filename in files:
                        if filename.endswith(".meta.json"):
                            continue
                        
                        artifact_path = os.path.join(root, filename)
                        rel_path = os.path.relpath(artifact_path, self.experiment_dir)
                        
                        # Try to determine artifact type and content
                        artifact_type = "link"
                        artifact_content = ""
                        
                        if filename.endswith((".py", ".json", ".md", ".txt")):
                            try:
                                with open(artifact_path, "r") as f:
                                    content = f.read()
                                    if len(content) < 5000:  # Only include small files
                                        artifact_type = "code" if filename.endswith(".py") else "text"
                                        artifact_content = content
                            except:
                                pass
                        
                        phase_data["artifacts"].append({
                            "name": filename,
                            "path": rel_path,
                            "type": artifact_type,
                            "content": artifact_content,
                            "description": f"Artifact from {phase} phase"
                        })
            
            # Get dialog entries for this phase
            dialog_entries = self.create_agent_dialog_summary(dialogs_dir, phase, max_entries=20)
            for entry in dialog_entries:
                phase_data["dialogs"].append({
                    "agent": entry.get("agent", "Unknown"),
                    "role": entry.get("role", "agent"),
                    "content": entry.get("content", ""),
                    "timestamp": entry.get("timestamp", "")
                })
            
            template_data["phases"].append(phase_data)
        
        # Render the template
        try:
            template = self.jinja_env.get_template(template_name)
            html_content = template.render(**template_data)
            
            # Save the HTML report
            report_path = os.path.join(self.reports_dir, filename)
            with open(report_path, "w") as f:
                f.write(html_content)
            
            self.logger.info(f"Experiment report saved to {report_path}")
            return report_path
            
        except Exception as e:
            self.logger.error(f"Error creating experiment report: {e}")
            raise
    
    def generate_experiment_dashboard(
        self,
        metrics: Dict[str, Any],
        config: Dict[str, Any],
        filename: str = "dashboard.html"
    ) -> str:
        """
        Generate an interactive dashboard for the experiment.
        
        Args:
            metrics: Dictionary of metrics
            config: Configuration dictionary
            filename: Filename for the output HTML file
            
        Returns:
            Path to the saved HTML file
        """
        # This is a simplified version that just creates a basic HTML file
        
        dashboard_html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Experiment Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .card {{ background: #f9f9f9; border-radius: 5px; padding: 15px; margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric {{ font-size: 24px; font-weight: bold; color: #0066cc; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Experiment Dashboard: {config.get("research-topic", "Unknown Research")}</h1>
        
        <div class="card">
            <h2>Configuration</h2>
            <pre>{json.dumps(config, indent=2)}</pre>
        </div>
        
        <div class="card">
            <h2>Metrics</h2>
            <pre>{json.dumps(metrics, indent=2)}</pre>
        </div>
    </div>
</body>
</html>
"""
        
        # Save the dashboard
        dashboard_path = os.path.join(self.reports_dir, filename)
        with open(dashboard_path, "w") as f:
            f.write(dashboard_html)
        
        self.logger.info(f"Experiment dashboard saved to {dashboard_path}")
        return dashboard_path 