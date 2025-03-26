"""PDF generation module for Agent Laboratory.

This module provides tools for generating PDF reports from experiments.
"""

import os
import re
import logging
import tempfile
import subprocess
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

def generate_report_pdf(
    experiment_data: Dict[str, Any],
    agent_dialogs: List[Dict[str, Any]],
    output_file: str,
    artifacts_dir: Optional[str] = None
):
    """Generate a PDF report of the experiment.
    
    Args:
        experiment_data: Data about the experiment
        agent_dialogs: List of agent dialog histories
        output_file: Path to output file (without extension)
        artifacts_dir: Directory containing artifacts
    """
    # Validate input parameters
    if not isinstance(experiment_data, dict):
        logger.warning("experiment_data should be a dictionary, using empty dict")
        experiment_data = {}
    
    if not isinstance(agent_dialogs, list):
        logger.warning("agent_dialogs should be a list, using empty list")
        agent_dialogs = []
        
    # Extract artifact data if artifacts_dir is provided
    artifact_data = None
    if artifacts_dir and os.path.isdir(artifacts_dir):
        artifact_data = {
            "figures": [], 
            "code_files": []
        }
        # Add code to extract figures and code files if needed
    
    # Get the research topic
    research_topic = experiment_data.get("research_topic", "Research Project")
    
    # Generate LaTeX document
    latex_content = generate_latex_document(
        research_topic=research_topic,
        discourse_data=agent_dialogs,
        artifact_data=artifact_data
    )
    
    # Write LaTeX to file
    latex_file = f"{output_file}.tex"
    os.makedirs(os.path.dirname(os.path.abspath(latex_file)), exist_ok=True)
    with open(latex_file, "w") as f:
        f.write(latex_content)
    
    # Compile the LaTeX document to PDF
    try:
        # Try using compile_latex_file which runs directly on the file
        logger.info(f"Compiling LaTeX file: {latex_file}")
        pdf_path = compile_latex_file(latex_file)
        logger.info(f"Generated PDF report: {pdf_path}")
        return pdf_path
    except Exception as e:
        logger.error(f"Error compiling LaTeX file: {e}")
        # Fall back to compile_latex if file compilation fails
        try:
            logger.info("Attempting to compile with fallback method")
            pdf_path = compile_latex(latex_content, output_file)
            logger.info(f"Generated PDF report using fallback method: {pdf_path}")
            return pdf_path
        except Exception as e:
            logger.error(f"Error compiling LaTeX with fallback method: {e}")
            return None

def compile_latex(content: str, output_file: str) -> str:
    """Compile LaTeX content to PDF.
    
    Args:
        content: LaTeX content
        output_file: Path to output file (without extension)
        
    Returns:
        str: Path to the compiled PDF
    """
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Write content to a temporary file
        temp_file = os.path.join(temp_dir, "document.tex")
        with open(temp_file, "w") as f:
            f.write(content)
        
        # Run pdflatex twice for proper cross-references and TOC
        for i in range(2):
            process = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", temp_file],
                cwd=temp_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Check for common LaTeX errors
            if process.returncode != 0:
                output = process.stdout.decode()
                error_msg = "Error compiling LaTeX"
                
                # Try to extract specific error messages
                if "Emergency stop" in output:
                    error_line = next((line for line in output.split('\n') if "Emergency stop" in line), "")
                    error_context = output.split(error_line)[0].split('\n')[-5:]
                    error_msg = f"LaTeX emergency stop: {' '.join(error_context)}"
                elif "Fatal error" in output:
                    error_line = next((line for line in output.split('\n') if "Fatal error" in line), "")
                    error_msg = f"LaTeX fatal error: {error_line}"
                
                logger.error(f"{error_msg}: {process.stderr.decode()}")
                # Write the LaTeX error log to a file for debugging
                with open(f"{output_file}.log", "w") as f:
                    f.write(output)
                
                # Only raise exception on the second pass
                if i == 1:
                    raise Exception(error_msg)
        
        # Copy the compiled PDF to the output file
        pdf_path = f"{output_file}.pdf"
        os.makedirs(os.path.dirname(os.path.abspath(pdf_path)), exist_ok=True)
        
        temp_pdf = os.path.join(temp_dir, "document.pdf")
        if os.path.exists(temp_pdf):
            import shutil
            shutil.copy(temp_pdf, pdf_path)
            return pdf_path
        else:
            raise FileNotFoundError("Compiled PDF not found")

def compile_latex_file(latex_file: str) -> str:
    """Compile a LaTeX file to PDF.
    
    Args:
        latex_file: Path to the LaTeX file
        
    Returns:
        str: Path to the compiled PDF
    """
    # Get directory and filename
    dir_path = os.path.dirname(os.path.abspath(latex_file))
    file_name = os.path.basename(latex_file)
    
    # Compile the document using pdflatex
    process = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", file_name],
        cwd=dir_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    if process.returncode != 0:
        logger.error(f"Error compiling LaTeX file {latex_file}: {process.stderr.decode()}")
        raise Exception(f"Error compiling LaTeX file {latex_file}")
    
    # Run pdflatex twice for cross-references
    process = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", file_name],
        cwd=dir_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Return path to the compiled PDF
    pdf_file = os.path.splitext(latex_file)[0] + ".pdf"
    if os.path.exists(pdf_file):
        return pdf_file
    else:
        raise FileNotFoundError(f"Compiled PDF not found for {latex_file}")

def generate_latex_document(
    research_topic: str,
    discourse_data: Optional[List[Dict[str, Any]]] = None,
    artifact_data: Optional[Dict[str, Any]] = None
) -> str:
    """Generate a LaTeX document from the research data.
    
    Args:
        research_topic: The title of the research
        discourse_data: Data about agent discourse
        artifact_data: Data about artifacts
        
    Returns:
        str: LaTeX document content
    """
    # Ensure discourse_data is a list
    if discourse_data is None:
        discourse_data = []
        
    # Escape the research topic for LaTeX
    safe_research_topic = escape_latex(research_topic)
    
    # LaTeX preamble
    latex_content = r"""
\documentclass[11pt,a4paper]{article}

% Basic packages
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{parskip}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{listings}
\usepackage{float}
\usepackage{soul}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{mathtools}
% Try to only include algorithm packages if available
\IfFileExists{algorithm.sty}{
    \usepackage{algorithm}
    \usepackage{algpseudocode}
}{
    % Algorithm package not available, skip it
}
\usepackage{tikz}
\usetikzlibrary{shapes,arrows,positioning}

% Page setup
\usepackage[margin=1in]{geometry}

% Header and footer
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0.4pt}
\fancyhead[L]{Agent Laboratory}
\fancyhead[R]{\today}
\fancyfoot[C]{\thepage}

% Custom colors
\definecolor{codegreen}{rgb}{0,0.6,0}
\definecolor{codegray}{rgb}{0.5,0.5,0.5}
\definecolor{codepurple}{rgb}{0.58,0,0.82}
\definecolor{backcolour}{rgb}{0.95,0.95,0.95}

% Code listing style
\lstdefinestyle{mystyle}{
    backgroundcolor=\color{backcolour},   
    commentstyle=\color{codegreen},
    keywordstyle=\color{magenta},
    numberstyle=\tiny\color{codegray},
    stringstyle=\color{codepurple},
    basicstyle=\ttfamily\footnotesize,
    breakatwhitespace=false,         
    breaklines=true,                 
    captionpos=b,                    
    keepspaces=true,                 
    numbers=left,                    
    numbersep=5pt,                  
    showspaces=false,                
    showstringspaces=false,
    showtabs=false,                  
    tabsize=2
}
\lstset{style=mystyle}

% Title
\title{\textbf{\Large{Research Report:}} \\ \huge{\textsf{""" + safe_research_topic + r"""}}}
\author{Agent Laboratory Team}
\date{\today}

\begin{document}

\maketitle

\begin{abstract}
This report documents the methodology, experiments, and findings of research conducted using the Agent Laboratory framework. The focus of this research is on \textbf{""" + safe_research_topic + r"""}.

The research was conducted through a systematic process involving multiple agent collaborations, including professors, engineers, and critics, to ensure comprehensive and robust results.
\end{abstract}

\tableofcontents
\newpage

\section{Introduction}
This research addresses """ + safe_research_topic + r""". The work was conducted using a systematic process implemented within the Agent Laboratory framework, involving multiple phases of research, development, and analysis.

The research followed these key phases:
\begin{itemize}
"""
    
    # Add phases based on discourse data
    if discourse_data:
        for phase_entry in discourse_data:
            phase_name = phase_entry.get("phase", "")
            if phase_name:
                latex_content += f"\n\\item \\textbf{{{escape_latex(phase_name.replace('-', ' ').title())}}} - "
                
                # Add a description for each phase based on common descriptions
                phase_descriptions = {
                    "literature-review": "Review of existing research and methodologies",
                    "plan-formulation": "Developing the research plan and approach",
                    "data-preparation": "Preparing data and defining the experimental setup",
                    "code-implementation": "Implementing the algorithms and models",
                    "running-experiments": "Executing experiments and collecting results",
                    "results-interpretation": "Analyzing and interpreting the experimental findings",
                    "report-writing": "Compiling findings into a comprehensive report"
                }
                
                latex_content += escape_latex(phase_descriptions.get(phase_name, "Conducted research tasks"))
    else:
        # Add default phases when no discourse data is available
        latex_content += """
\item \\textbf{Literature Review} - Review of existing research and methodologies
\item \\textbf{Implementation} - Development and implementation of algorithms and models
\item \\textbf{Evaluation} - Testing and evaluating the implemented solution
\item \\textbf{Analysis} - Analysis of results and drawing conclusions
"""
    
    latex_content += """
\end{itemize}

Each phase was approached collaboratively by multiple expert agents, including research professors, engineers, and critics, to ensure comprehensive, rigorous, and technically sound results.

"""
    
    # Add literature review section if available
    if discourse_data and any(entry.get("phase") == "literature-review" for entry in discourse_data):
        lit_review_entry = next((entry for entry in discourse_data if entry.get("phase") == "literature-review"), None)
        if lit_review_entry:
            latex_content += """
\section{Literature Review}

The literature review phase identified relevant research, methodologies, and findings in the field. The following summarizes the key insights from this review:

"""
            # Extract content from professor's initial contribution or the integrated output
            discourse = lit_review_entry.get("discourse", {})
            lit_review_content = discourse.get("integrated_output", discourse.get("professor_initial", ""))
            
            # Convert markdown headers to LaTeX subsections
            lit_review_content = re.sub(r'^## (.*)', r'\\subsection{\1}', lit_review_content, flags=re.MULTILINE)
            lit_review_content = re.sub(r'^### (.*)', r'\\subsubsection{\1}', lit_review_content, flags=re.MULTILINE)
            
            # Convert markdown lists
            lit_review_content = re.sub(r'^\s*\-\s*(.*)', r'\\begin{itemize}\n\\item \1', lit_review_content, flags=re.MULTILINE)
            lit_review_content = re.sub(r'^\s*\d+\.\s*(.*)', r'\\begin{enumerate}\n\\item \1', lit_review_content, flags=re.MULTILINE)
            
            # Clean up converted content
            lit_review_content = escape_latex(lit_review_content)
            
            latex_content += lit_review_content
    
    # Add methodology section combining plan-formulation and data-preparation
    methodology_content = ""
    plan_entry = next((entry for entry in discourse_data if entry.get("phase") == "plan-formulation"), None)
    data_entry = next((entry for entry in discourse_data if entry.get("phase") == "data-preparation"), None)
    
    if plan_entry or data_entry:
        latex_content += """
\section{Methodology}

This section describes the research methodology, including the approach, experimental setup, and data preparation.

"""
        if plan_entry:
            discourse = plan_entry.get("discourse", {})
            plan_content = discourse.get("integrated_output", discourse.get("professor_initial", ""))
            
            latex_content += """
\subsection{Research Plan and Approach}

"""
            # Clean up and add plan content
            plan_content = escape_latex(plan_content)
            latex_content += plan_content
        
        if data_entry:
            discourse = data_entry.get("discourse", {})
            data_content = discourse.get("integrated_output", discourse.get("professor_initial", ""))
            
            latex_content += """
\subsection{Data Preparation and Experimental Setup}

"""
            # Clean up and add data preparation content
            data_content = escape_latex(data_content)
            latex_content += data_content
    
    # Add implementation section
    code_entry = next((entry for entry in discourse_data if entry.get("phase") == "code-implementation"), None)
    if code_entry:
        latex_content += """
\section{Implementation}

This section presents the implementation code for the research project, focusing on the key algorithms and methods developed.

"""
        discourse = code_entry.get("discourse", {})
        code_content = discourse.get("integrated_output", discourse.get("professor_initial", ""))
        
        # Extract code blocks
        code_blocks = re.findall(r'```python(.*?)```', code_content, flags=re.DOTALL)
        
        for i, code_block in enumerate(code_blocks):
            clean_code = code_block.strip()
            latex_content += f"""
\\subsection{{Code Block {i+1}}}

\\begin{{lstlisting}}[language=Python, caption={{Implementation code for the POMDP with Active Inference}}]
{clean_code}
\\end{{lstlisting}}

"""
    
    # Add experiments section
    experiments_entry = next((entry for entry in discourse_data if entry.get("phase") == "running-experiments"), None)
    if experiments_entry:
        latex_content += """
\section{Experiments}

This section describes the experiments conducted to evaluate the implementation.

"""
        discourse = experiments_entry.get("discourse", {})
        experiments_content = discourse.get("integrated_output", discourse.get("professor_initial", ""))
        
        # Clean up and add experiments content
        experiments_content = escape_latex(experiments_content)
        latex_content += experiments_content
    
    # Add results section
    results_entry = next((entry for entry in discourse_data if entry.get("phase") == "results-interpretation"), None)
    if results_entry:
        latex_content += """
\section{Results and Discussion}

This section presents and discusses the results of the experiments.

"""
        discourse = results_entry.get("discourse", {})
        results_content = discourse.get("integrated_output", discourse.get("professor_initial", ""))
        
        # Clean up and add results content
        results_content = escape_latex(results_content)
        latex_content += results_content
    
    # Add figures if available
    if artifact_data and artifact_data.get("figures"):
        latex_content += """
\section{Visualizations}

This section presents visualizations generated during the experiments.

"""
        for i, figure in enumerate(artifact_data.get("figures", [])):
            figure_path = figure.get("path", "")
            caption = figure.get("caption", f"Figure {i+1}")
            
            if os.path.exists(figure_path):
                latex_content += f"""
\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.8\\textwidth]{{{figure_path}}}
\\caption{{{escape_latex(caption)}}}
\\label{{fig:{i+1}}}
\\end{{figure}}

"""
    
    # Add code files as appendix
    if artifact_data and artifact_data.get("code_files"):
        latex_content += """
\section{Appendix: Implementation Code}

This appendix contains the full implementation code used in the experiments.

"""
        for i, code_file in enumerate(artifact_data.get("code_files", [])):
            filename = code_file.get("filename", f"Code File {i+1}")
            content = code_file.get("content", "")
            
            latex_content += f"""
\\subsection{{{escape_latex(filename)}}}

\\begin{{lstlisting}}[language=Python, caption={{Full implementation code}}]
{content}
\\end{{lstlisting}}

"""
    
    # Add agent discourse section
    latex_content += """
\section{Agent Discourse and Collaboration}

The research process involved collaboration between multiple expert agents, each contributing their specific expertise to enhance the quality and rigor of the work.

"""
    
    if discourse_data:
        for phase_entry in discourse_data:
            phase_name = phase_entry.get("phase", "")
            discourse = phase_entry.get("discourse", {})
            
            if phase_name and discourse:
                latex_content += f"""
\\subsection{{{escape_latex(phase_name.replace('-', ' ').title())} Phase}}

"""
                
                # Add brief summary of agent contributions
                latex_content += """
This phase involved contributions from:

\\begin{itemize}
\\item \\textbf{Professor Agent}: Led the initial research direction and synthesized the final approach.
\\item \\textbf{Engineer Agent}: Provided technical expertise and implementation recommendations.
\\item \\textbf{Critic Agent}: Offered critical assessment and identified areas for improvement.
\\end{itemize}

\\subsubsection{Key Insights and Contributions}

"""
                
                # Add brief excerpts from each agent
                prof_content = discourse.get("professor_initial", "")
                eng_content = discourse.get("engineer_contribution", "")
                critic_content = discourse.get("critic_feedback", "")
                
                # Extract first paragraph from each agent
                prof_excerpt = extract_first_paragraph(prof_content)
                eng_excerpt = extract_first_paragraph(eng_content)
                critic_excerpt = extract_first_paragraph(critic_content)
                
                if prof_excerpt:
                    latex_content += f"""
\\textbf{{Professor's Initial Direction:}} {escape_latex(prof_excerpt)}

"""
                
                if eng_excerpt:
                    latex_content += f"""
\\textbf{{Engineer's Technical Perspective:}} {escape_latex(eng_excerpt)}

"""
                
                if critic_excerpt:
                    latex_content += f"""
\\textbf{{Critic's Assessment:}} {escape_latex(critic_excerpt)}

"""
    else:
        latex_content += "\\textit{No agent discourse available.}\n\n"
    
    # Add conclusion
    latex_content += """
\section{Conclusion}

This report has documented the comprehensive research process for %s. Through systematic collaboration between expert agents, including professors, engineers, and critics, the research progressed through multiple phases from initial planning to final implementation and analysis.

The key contributions include:
\\begin{itemize}
    \item A systematic methodology for approaching %s
    \item Technical implementation demonstrating the principles in action
    \item Critical analysis of results and implications
    \item Insights for future research directions
\\end{itemize}

The Agent Laboratory framework has facilitated this multi-agent, multi-phase research process, enabling structured collaboration and comprehensive documentation throughout.

\end{document}
""" % (
        escape_latex(research_topic),
        escape_latex(research_topic)
    )
    
    return latex_content

def escape_latex(text):
    """Escape LaTeX special characters in a string.
    
    Args:
        text: String to escape
        
    Returns:
        str: Escaped string
    """
    if text is None:
        return ""
        
    # Define LaTeX special characters and their escaped versions
    latex_special_chars = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
        '\\': r'\textbackslash{}',
        '<': r'\textless{}',
        '>': r'\textgreater{}'
    }
    
    # Convert to string if necessary
    if not isinstance(text, str):
        text = str(text)
    
    # Handle empty string
    if not text:
        return ""
    
    # Handle markdown headers properly by converting them to LaTeX sections BEFORE escaping
    # Convert markdown headers to LaTeX sections (important to do this before escaping other characters)
    lines = text.split('\n')
    for i, line in enumerate(lines):
        # Handle markdown headers (###, ##, #)
        if line.startswith('### '):
            lines[i] = '\\subsubsection{' + line[4:] + '}'
        elif line.startswith('## '):
            lines[i] = '\\subsection{' + line[3:] + '}'
        elif line.startswith('# '):
            lines[i] = '\\section{' + line[2:] + '}'
        # Handle other variations of markdown headers
        elif line.startswith('####'):
            lines[i] = '\\paragraph{' + line.lstrip('#').lstrip() + '}'
    
    # Rejoin text
    text = '\n'.join(lines)
    
    # Now escape special characters in the remaining text
    for char, escaped in latex_special_chars.items():
        text = text.replace(char, escaped)
    
    # Convert escaped markdown-style headers that might remain
    text = re.sub(r'\\#\\#\\#\\#\\s+(.+)', r'\\paragraph{\1}', text)
    text = re.sub(r'\\#\\#\\#\\s+(.+)', r'\\subsubsection{\1}', text)
    text = re.sub(r'\\#\\#\\s+(.+)', r'\\subsection{\1}', text)
    text = re.sub(r'\\#\\s+(.+)', r'\\section{\1}', text)
    
    return text

def extract_first_paragraph(text):
    """Extract the first paragraph from a text.
    
    Args:
        text: Input text
        
    Returns:
        str: First paragraph
    """
    # Split by double newline (paragraph break) and take the first non-empty paragraph
    paragraphs = [p.strip() for p in text.split('\n\n')]
    paragraphs = [p for p in paragraphs if p]
    
    if not paragraphs:
        return ""
    
    # Return the first paragraph, truncated if too long
    first_para = paragraphs[0]
    if len(first_para) > 300:
        return first_para[:300] + "..."
    return first_para 