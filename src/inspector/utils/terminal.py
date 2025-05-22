"""
Terminal utility functions for DB Inspector.
"""

import subprocess
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def run_terminal_cmd(
    command: str,
    is_background: bool = False,
    require_user_approval: bool = True,
    explanation: Optional[str] = None
) -> None:
    """
    Run a terminal command with proper logging and error handling.
    
    Args:
        command: The command to run
        is_background: Whether to run the command in the background
        require_user_approval: Whether to require user approval before running
        explanation: Optional explanation of what the command does
    """
    if explanation:
        logger.info(f"Command explanation: {explanation}")
    
    logger.info(f"Running command: {command}")
    
    try:
        # Run the command
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if not is_background:
            # Wait for the command to complete and get output
            stdout, stderr = process.communicate()
            
            # Log the output
            if stdout:
                logger.info(f"Command stdout:\n{stdout}")
            if stderr:
                logger.error(f"Command stderr:\n{stderr}")
            
            # Check return code
            if process.returncode != 0:
                logger.error(f"Command failed with return code {process.returncode}")
                raise subprocess.CalledProcessError(process.returncode, command)
            
            logger.info("Command completed successfully")
    except Exception as e:
        logger.error(f"Error running command: {e}")
        raise 