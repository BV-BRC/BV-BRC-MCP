import requests
import json
from typing import Any, Dict, Optional
import sys


class JsonRpcCaller:
    """A minimal, generic JSON-RPC caller class."""
    
    def __init__(self, service_url: str):
        """
        Initialize the JSON-RPC caller with service URL and authentication token.
        
        Args:
            service_url: The base URL for the service API
            token: Authentication token for API calls
        """
        self.service_url = service_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/jsonrpc+json'
        })
    
    def call(self, method: str, params: Optional[Any] = None, request_id: int = 1, token: str = None) -> Any:
        """
        Make a JSON-RPC call to the service API.
        
        Args:
            method: The RPC method name to call
            params: Optional parameters for the method (can be dict or list)
            request_id: Request ID for the JSON-RPC call
            token: Authentication token for API calls
        Returns:
            The response from the API call
            
        Raises:
            requests.RequestException: If the HTTP request fails
            ValueError: If the response contains an error
        """
        # Handle case where params is a list (for AppService.start_app2)
        # AppService.start_app2 expects params as a list: [app_name, params_dict, {}]
        try:
            if params is None:
                params_dict = {}
                params_dict['base_url'] = 'https://www.patricbrc.org'
                params = params_dict
            elif type(params).__name__ == 'list':
                # If params is a list, use it as-is (AppService.start_app2 format)
                # Don't modify it, just pass through
                pass
            elif type(params).__name__ == 'dict':
                # If params is a dict, add base_url
                params_dict = dict(params)  # Make a copy to avoid modifying the original
                params_dict['base_url'] = 'https://www.patricbrc.org'
                params = params_dict
            else:
                # Convert other types to dict
                params = {'data': params, 'base_url': 'https://www.patricbrc.org'}
        except (TypeError, AttributeError) as e:
            # Fallback: if we can't determine the type, assume it's a list and pass through
            print(f"Warning: Could not determine params type, passing through as-is: {e}", file=sys.stderr)
            pass

        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "id": request_id,
            "params": params,
        }

        if token:
            self.session.headers.update({
                'Authorization': f'{token}'
            })

        try:
            response = self.session.post(
                self.service_url,
                data=json.dumps(payload),
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Check for JSON-RPC errors (result should be a dict in JSON-RPC format)
            if isinstance(result, dict) and "error" in result:
                raise ValueError(f"JSON-RPC error: {result['error']}")
            
            # Return the result field, which could be a dict, list, or other type
            if isinstance(result, dict):
                return result.get("result", {})
            # If result is not a dict (unexpected), return it as-is
            return result
        
        except AttributeError as e:
            # Handle case where e doesn't have response attribute
            print(f"error: {str(e)}", file=sys.stderr)
            raise
        except Exception as e:
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                print(f"error: {e.response.text}", file=sys.stderr)
            else:
                print(f"error: {str(e)}", file=sys.stderr)
            raise
        except requests.RequestException as e:
            raise requests.RequestException(f"HTTP request failed: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {e}")

    
    def close(self):
        """Close the HTTP session."""
        self.session.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

