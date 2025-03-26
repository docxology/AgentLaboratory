"""LaTeX executor for PDF generation in Agent Laboratory.

This module handles the compilation of LaTeX documents to PDF.
"""

import os
import logging
import subprocess
import shutil
import tempfile
import re
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

def ensure_latex_packages() -> bool:
    """Ensure required LaTeX packages are installed.
    
    Returns:
        bool: True if all required packages are installed, False otherwise
    """
    # Check if pdflatex is installed
    try:
        result = subprocess.run(
            ["pdflatex", "--version"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            timeout=5
        )
        if result.returncode != 0:
            logger.warning("pdflatex is not installed or not working properly")
            return False
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.warning("pdflatex is not installed or not in PATH")
        return False
    
    logger.info("LaTeX is properly installed")
    return True

def log_command_output(output: bytes, error: bytes, level: str = "info") -> None:
    """Log command output and error.
    
    Args:
        output: Command output
        error: Command error
        level: Logging level (info, warning, error)
    """
    log_func = getattr(logger, level)
    
    if output:
        output_str = output.decode('utf-8', errors='replace')
        for line in output_str.splitlines():
            if line.strip():
                log_func(f"LaTeX output: {line}")
    
    if error:
        error_str = error.decode('utf-8', errors='replace')
        for line in error_str.splitlines():
            if line.strip():
                log_func(f"LaTeX error: {line}")

def compile_latex(input_file: str, output_file: str) -> bool:
    """Compile a LaTeX document to PDF.
    
    Args:
        input_file: Path to input LaTeX file
        output_file: Path to output PDF file
        
    Returns:
        bool: True if compilation was successful, False otherwise
    """
    logger.info(f"Compiling LaTeX document {input_file} to {output_file}")
    
    # Ensure required LaTeX packages are installed
    if not ensure_latex_packages():
        logger.error("Required LaTeX packages are not installed")
        return False
    
    # Create a temporary directory for compilation
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Created temporary directory for LaTeX compilation: {temp_dir}")
    
    try:
        # Copy the input file to the temporary directory
        input_filename = os.path.basename(input_file)
        temp_input_file = os.path.join(temp_dir, input_filename)
        shutil.copy(input_file, temp_input_file)
        
        # Copy supporting files if they exist
        input_dir = os.path.dirname(input_file)
        for file in os.listdir(input_dir):
            if file != input_filename and not file.endswith('.pdf'):
                src_file = os.path.join(input_dir, file)
                if os.path.isfile(src_file):
                    shutil.copy(src_file, os.path.join(temp_dir, file))
        
        # Run pdflatex with appropriate arguments
        logger.info(f"Running pdflatex on {temp_input_file}")
        
        # Run pdflatex twice to resolve references
        for i in range(2):
            try:
                # Use a longer timeout (120 seconds) for pdflatex
                process = subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", input_filename],
                    cwd=temp_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=120
                )
                
                # Log output and error
                log_command_output(process.stdout, process.stderr)
                
                if process.returncode != 0:
                    logger.error(f"pdflatex returned non-zero exit status: {process.returncode}")
                    
                    # Try to extract error message from log
                    log_file = os.path.join(temp_dir, os.path.splitext(input_filename)[0] + ".log")
                    if os.path.exists(log_file):
                        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                            log_content = f.read()
                            errors = re.findall(r"! (.*?)\.\\s", log_content, re.DOTALL)
                            for error in errors:
                                logger.error(f"LaTeX error: {error.strip()}")
                    
                    # Only return False if the final run fails
                    if i == 1:
                        return False
                
            except subprocess.TimeoutExpired:
                logger.error("pdflatex timed out after 120 seconds")
                return False
            
            except Exception as e:
                logger.error(f"Error running pdflatex: {e}")
                return False
        
        # Check if PDF was generated
        pdf_filename = os.path.splitext(input_filename)[0] + ".pdf"
        temp_output_file = os.path.join(temp_dir, pdf_filename)
        
        if not os.path.exists(temp_output_file):
            logger.error(f"PDF file not generated: {temp_output_file}")
            return False
        
        # Copy the PDF to the output location
        shutil.copy(temp_output_file, output_file)
        logger.info(f"Successfully compiled LaTeX document to {output_file}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error compiling LaTeX document: {e}")
        logger.exception(e)
        return False
    
    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Error cleaning up temporary directory: {e}")

def check_latex_document(latex_content: str) -> Tuple[bool, str]:
    """Check if LaTeX content is properly structured.
    
    Args:
        latex_content: LaTeX content to check
        
    Returns:
        Tuple[bool, str]: (is_valid, fixed_content)
    """
    # Check for document class
    has_document_class = re.search(r'\\documentclass', latex_content) is not None
    
    # Check for document environment
    has_begin_document = re.search(r'\\begin\{document\}', latex_content) is not None
    has_end_document = re.search(r'\\end\{document\}', latex_content) is not None
    
    if not has_document_class or not has_begin_document or not has_end_document:
        logger.warning("LaTeX document is not properly structured")
        
        # Fix document
        fixed_content = latex_content
        
        # Add document class if missing
        if not has_document_class:
            fixed_content = r'\documentclass[11pt,a4paper]{article}' + '\n' + fixed_content
        
        # Add document environment if missing
        if not has_begin_document:
            fixed_content = fixed_content + '\n\\begin{document}\n'
        
        if not has_end_document:
            fixed_content = fixed_content + '\n\\end{document}\n'
        
        return False, fixed_content
    
    return True, latex_content

def compile_latex_file(latex_file_path, timeout=60):
    """Compile a LaTeX file directly to PDF.
    
    Args:
        latex_file_path: Path to the LaTeX file to compile
        timeout: Maximum compilation time in seconds
        
    Returns:
        tuple: (success, message)
    """
    import subprocess
    import os
    
    # Get the directory of the LaTeX file
    latex_dir = os.path.dirname(latex_file_path)
    
    # Get the base name without extension
    latex_basename = os.path.basename(latex_file_path)
    latex_basename = os.path.splitext(latex_basename)[0]
    
    try:
        # Run pdflatex command
        process = subprocess.Popen(
            ["pdflatex", "-interaction=nonstopmode", latex_basename],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=latex_dir,
            universal_newlines=True
        )
        
        # Wait for the process to complete with timeout
        stdout, stderr = process.communicate(timeout=timeout)
        
        if process.returncode != 0:
            return False, f"LaTeX compilation failed with return code {process.returncode}"
        
        # Run pdflatex again to resolve references
        process = subprocess.Popen(
            ["pdflatex", "-interaction=nonstopmode", latex_basename],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=latex_dir,
            universal_newlines=True
        )
        
        stdout, stderr = process.communicate(timeout=timeout)
        
        if process.returncode != 0:
            return False, f"LaTeX compilation failed with return code {process.returncode}"
        
        # Check if the PDF file was created
        pdf_file = os.path.join(latex_dir, f"{latex_basename}.pdf")
        if not os.path.exists(pdf_file):
            return False, "PDF file was not created"
        
        return True, f"Successfully compiled PDF: {pdf_file}"
    
    except subprocess.TimeoutExpired:
        return False, f"LaTeX compilation exceeded the timeout limit of {timeout} seconds"
    except Exception as e:
        return False, f"Error compiling LaTeX: {str(e)}" 