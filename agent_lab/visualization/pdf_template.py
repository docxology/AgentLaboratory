"""
PDF Template for the Agent Laboratory reports.
"""

REPORT_TEMPLATE = r"""
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
\title{\textbf{\Large{Research Report:} \\ \huge{\textsf{%RESEARCH_TOPIC%}}}}
\author{Agent Laboratory Team}
\date{\today}

\begin{document}

\maketitle

\begin{abstract}
This report documents the methodology, experiments, and findings of research conducted using the Agent Laboratory framework. The focus of this research is on \textbf{%RESEARCH_TOPIC%}.

The research was conducted through a systematic process involving multiple agent collaborations, including professors, engineers, and critics, to ensure comprehensive and robust results.
\end{abstract}

\tableofcontents
\newpage

\section{Introduction}
This research addresses %RESEARCH_TOPIC%. The work was conducted using a systematic process implemented within the Agent Laboratory framework, involving multiple phases of research, development, and analysis.

The research followed these key phases:
%PHASES_LIST%

Each phase was approached collaboratively by multiple expert agents, including research professors, engineers, and critics, to ensure comprehensive, rigorous, and technically sound results.

\section{Implementation}

This section presents the implementation code for the research project, focusing on the key algorithms and methods developed.

\subsection{Core Implementation}
%IMPLEMENTATION_CODE%

\section{Agent Discourse and Collaboration}

The research process involved collaboration between multiple expert agents, each contributing their specific expertise to enhance the quality and rigor of the work.

%AGENT_DISCOURSE%

\section{Conclusion}

This report has documented the comprehensive research process for %RESEARCH_TOPIC%. Through systematic collaboration between expert agents, including professors, engineers, and critics, the research progressed through multiple phases from initial planning to final implementation and analysis.

The key contributions include:
\begin{itemize}
    \item A systematic methodology for approaching %RESEARCH_TOPIC%
    \item Technical implementation demonstrating the principles in action
    \item Critical analysis of results and implications
    \item Insights for future research directions
\end{itemize}

The Agent Laboratory framework has facilitated this multi-agent, multi-phase research process, enabling structured collaboration and comprehensive documentation throughout.

\end{document}
""" 