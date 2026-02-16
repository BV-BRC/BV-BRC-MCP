"""
Workflow Engine HTTP Client

This module provides a client for interacting with the workflow engine REST API.
"""

import aiohttp
import asyncio
from typing import Dict, Any, Optional
import sys


class WorkflowEngineClient:
    """HTTP client for workflow engine REST API."""
    
    def __init__(self, base_url: str, timeout: int = 30):
        """
        Initialize workflow engine client.
        
        Args:
            base_url: Base URL for workflow engine API (e.g., "http://localhost:8000/api/v1")
            timeout: Request timeout in seconds (default: 30)
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        
    async def submit_workflow(self, workflow_json: Dict[str, Any], auth_token: str) -> Dict[str, Any]:
        """
        Submit a workflow to the workflow engine for execution.
        
        Args:
            workflow_json: Complete workflow manifest dictionary
            auth_token: BV-BRC authentication token
            
        Returns:
            Dictionary with:
            - workflow_id: The assigned workflow ID
            - status: Initial status (typically "pending")
            - message: Confirmation message
            
        Raises:
            WorkflowEngineError: If submission fails
        """
        url = f"{self.base_url}/workflows/submit"
        headers = {
            "Content-Type": "application/json",
            "Authorization": auth_token
        }
        
        try:
            print(f"Submitting workflow to workflow engine: {url}", file=sys.stderr)
            
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(url, json=workflow_json, headers=headers) as response:
                    response_text = await response.text()
                    
                    if response.status == 201:
                        result = await response.json()
                        print(f"Workflow submitted successfully: {result.get('workflow_id')}", file=sys.stderr)
                        return result
                    elif response.status == 400:
                        # Validation error
                        try:
                            error_data = await response.json()
                            error_msg = error_data.get('detail', response_text)
                        except:
                            error_msg = response_text
                        raise WorkflowEngineError(
                            f"Workflow validation failed: {error_msg}",
                            error_type="VALIDATION_FAILED",
                            status_code=400
                        )
                    elif response.status == 500:
                        # Server error
                        try:
                            error_data = await response.json()
                            error_msg = error_data.get('detail', response_text)
                        except:
                            error_msg = response_text
                        raise WorkflowEngineError(
                            f"Workflow engine internal error: {error_msg}",
                            error_type="ENGINE_ERROR",
                            status_code=500
                        )
                    else:
                        raise WorkflowEngineError(
                            f"Unexpected response from workflow engine: {response.status} - {response_text}",
                            error_type="UNKNOWN_ERROR",
                            status_code=response.status
                        )
                        
        except WorkflowEngineError:
            # Re-raise our custom errors (must be first to avoid being caught by other handlers)
            raise
        except aiohttp.ClientConnectorError as e:
            print(f"Failed to connect to workflow engine at {url}: {e}", file=sys.stderr)
            raise WorkflowEngineError(
                f"Cannot connect to workflow engine at {self.base_url}. Is it running?",
                error_type="CONNECTION_FAILED"
            ) from e
        except asyncio.TimeoutError as e:
            print(f"Workflow engine request timed out: {e}", file=sys.stderr)
            raise WorkflowEngineError(
                f"Workflow engine request timed out after {self.timeout.total}s",
                error_type="TIMEOUT"
            ) from e
        except Exception as e:
            print(f"Unexpected error submitting workflow: {e}", file=sys.stderr)
            raise WorkflowEngineError(
                f"Unexpected error: {str(e)}",
                error_type="UNKNOWN_ERROR"
            ) from e

    async def validate_workflow(self, workflow_json: Dict[str, Any], auth_token: str) -> Dict[str, Any]:
        """
        Validate and normalize a workflow in the workflow engine without submitting it.

        Args:
            workflow_json: Complete workflow manifest dictionary
            auth_token: BV-BRC authentication token

        Returns:
            Dictionary with:
            - valid: True if validation succeeded
            - workflow_json: Normalized/validated workflow manifest
            - warnings: Optional validation warnings
            - auto_fixes: Optional list of auto-applied fixes
            - message: Validation status message

        Raises:
            WorkflowEngineError: If validation fails
        """
        url = f"{self.base_url}/workflows/validate"
        headers = {
            "Content-Type": "application/json",
            "Authorization": auth_token
        }

        try:
            print(f"Validating workflow in workflow engine: {url}", file=sys.stderr)

            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(url, json=workflow_json, headers=headers) as response:
                    response_text = await response.text()

                    if response.status == 200:
                        result = await response.json()
                        print("Workflow validated successfully", file=sys.stderr)
                        return result
                    elif response.status == 400:
                        try:
                            error_data = await response.json()
                            error_msg = error_data.get('detail', response_text)
                        except Exception:
                            error_msg = response_text
                        raise WorkflowEngineError(
                            f"Workflow validation failed: {error_msg}",
                            error_type="VALIDATION_FAILED",
                            status_code=400
                        )
                    elif response.status == 404:
                        raise WorkflowEngineError(
                            "Workflow engine validate endpoint not found",
                            error_type="ENDPOINT_NOT_FOUND",
                            status_code=404
                        )
                    elif response.status == 500:
                        try:
                            error_data = await response.json()
                            error_msg = error_data.get('detail', response_text)
                        except Exception:
                            error_msg = response_text
                        raise WorkflowEngineError(
                            f"Workflow engine internal error during validation: {error_msg}",
                            error_type="ENGINE_ERROR",
                            status_code=500
                        )
                    else:
                        raise WorkflowEngineError(
                            f"Unexpected response from workflow engine validation: {response.status} - {response_text}",
                            error_type="UNKNOWN_ERROR",
                            status_code=response.status
                        )

        except WorkflowEngineError:
            raise
        except aiohttp.ClientConnectorError as e:
            print(f"Failed to connect to workflow engine at {url}: {e}", file=sys.stderr)
            raise WorkflowEngineError(
                f"Cannot connect to workflow engine at {self.base_url}. Is it running?",
                error_type="CONNECTION_FAILED"
            ) from e
        except asyncio.TimeoutError as e:
            print(f"Workflow engine validation request timed out: {e}", file=sys.stderr)
            raise WorkflowEngineError(
                f"Workflow engine validation request timed out after {self.timeout.total}s",
                error_type="TIMEOUT"
            ) from e
        except Exception as e:
            print(f"Unexpected error validating workflow: {e}", file=sys.stderr)
            raise WorkflowEngineError(
                f"Unexpected error: {str(e)}",
                error_type="UNKNOWN_ERROR"
            ) from e
    
    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """
        Get the status of a workflow.
        
        Args:
            workflow_id: The workflow ID to query
            
        Returns:
            Dictionary with workflow status information including:
            - workflow_id
            - workflow_name
            - status
            - created_at
            - updated_at
            - steps (array of step status objects)
            
        Raises:
            WorkflowEngineError: If query fails
        """
        url = f"{self.base_url}/workflows/{workflow_id}/status"
        
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 404:
                        raise WorkflowEngineError(
                            f"Workflow {workflow_id} not found",
                            error_type="NOT_FOUND",
                            status_code=404
                        )
                    else:
                        response_text = await response.text()
                        raise WorkflowEngineError(
                            f"Failed to get workflow status: {response.status} - {response_text}",
                            error_type="QUERY_FAILED",
                            status_code=response.status
                        )
                        
        except WorkflowEngineError:
            # Re-raise our custom errors (must be first)
            raise
        except aiohttp.ClientConnectorError as e:
            raise WorkflowEngineError(
                f"Cannot connect to workflow engine at {self.base_url}",
                error_type="CONNECTION_FAILED"
            ) from e
        except Exception as e:
            raise WorkflowEngineError(
                f"Unexpected error querying workflow status: {str(e)}",
                error_type="UNKNOWN_ERROR"
            ) from e
    
    async def health_check(self) -> bool:
        """
        Check if workflow engine is available and healthy.
        
        Returns:
            True if workflow engine is healthy, False otherwise
        """
        url = f"{self.base_url}/health"
        
        try:
            # Use a shorter timeout for health checks
            quick_timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=quick_timeout) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        # Check if MongoDB is connected
                        return data.get('mongodb') == 'connected'
                    return False
        except Exception as e:
            print(f"Workflow engine health check failed: {e}", file=sys.stderr)
            return False


class WorkflowEngineError(Exception):
    """Custom exception for workflow engine errors."""
    
    def __init__(self, message: str, error_type: str = "UNKNOWN_ERROR", status_code: Optional[int] = None):
        """
        Initialize workflow engine error.
        
        Args:
            message: Error message
            error_type: Type of error (e.g., "CONNECTION_FAILED", "VALIDATION_FAILED")
            status_code: HTTP status code if applicable
        """
        super().__init__(message)
        self.error_type = error_type
        self.status_code = status_code

