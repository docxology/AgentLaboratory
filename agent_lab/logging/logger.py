"""Basic logger for the Agent Laboratory.

This module provides a simple logger that can be used throughout the codebase.
"""

import os
import logging
from typing import Optional

class Logger:
    """Simple logger for the Agent Laboratory."""
    
    def __init__(
        self,
        name: str = "agent_lab",
        log_level: int = logging.INFO,
        log_file: Optional[str] = None,
        log_to_console: bool = True
    ):
        """Initialize the logger.
        
        Args:
            name: Logger name
            log_level: Log level (default: INFO)
            log_file: Optional log file path
            log_to_console: Whether to log to console
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        
        # Clear existing handlers
        self.logger.handlers = []
        
        # Add file handler if provided
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_format = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_format)
            self.logger.addHandler(file_handler)
        
        # Add console handler if requested
        if log_to_console:
            console_handler = logging.StreamHandler()
            console_format = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_format)
            self.logger.addHandler(console_handler)
    
    def debug(self, message: str) -> None:
        """Log a debug message.
        
        Args:
            message: Message to log
        """
        self.logger.debug(message)
    
    def info(self, message: str) -> None:
        """Log an informational message.
        
        Args:
            message: Message to log
        """
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log a warning message.
        
        Args:
            message: Message to log
        """
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log an error message.
        
        Args:
            message: Message to log
        """
        self.logger.error(message)
    
    def critical(self, message: str) -> None:
        """Log a critical message.
        
        Args:
            message: Message to log
        """
        self.logger.critical(message)
    
    def get_logger(self) -> logging.Logger:
        """Get the underlying logger.
        
        Returns:
            logging.Logger: The underlying logger
        """
        return self.logger 