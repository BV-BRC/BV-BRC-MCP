"""
Workflow manifest generation functions
"""
import os
import json
import sys
import time
from typing import Dict, List, Any, Optional
from common.llm_client import LLMClient
from common.json_rpc import JsonRpcCaller
from functions.service_functions import enumerate_apps, get_service_info

# Service catalog built once on first use and reused for all subsequent calls
# Since the service catalog is the same for all users, we build it once and keep it
_service_catalog: Optional[Dict[str, Any]] = None


def load_config_file(filename: str) -> Dict:
    """Load a JSON config file from the config directory."""
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(script_dir, 'config', filename)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_prompt_file(filename: str) -> str:
    """Load a prompt file from the prompts directory."""
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompt_path = os.path.join(script_dir, 'prompts', filename)
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def clear_service_catalog():
    """
    Clear the service catalog.
    
    Useful for testing or forcing a refresh of the catalog.
    """
    global _service_catalog
    _service_catalog = None
    print("Service catalog cleared", file=sys.stderr)


async def initialize_service_catalog(api: JsonRpcCaller, token: str, user_id: str = None) -> bool:
    """
    Initialize the service catalog at startup if a token is available.
    
    This is optional - the catalog will be built on first use if not initialized here.
    Building it at startup ensures the first request is fast.
    
    Args:
        api: JsonRpcCaller instance
        token: Authentication token
        user_id: User ID (optional)
        
    Returns:
        True if catalog was built successfully, False otherwise
    """
    try:
        print("Initializing service catalog at startup...", file=sys.stderr)
        await build_service_catalog(api, token, user_id, force_rebuild=False)
        return True
    except Exception as e:
        print(f"Warning: Could not initialize service catalog at startup: {e}", file=sys.stderr)
        print("  Catalog will be built on first use instead", file=sys.stderr)
        return False


async def build_service_catalog(api: JsonRpcCaller, token: str, user_id: str = None, force_rebuild: bool = False) -> Dict[str, Any]:
    """
    Build a comprehensive service catalog with names, descriptions, and schemas.
    
    The catalog is built once on first use and reused for all subsequent calls,
    since the service catalog is the same for all users. This avoids expensive
    API calls on every request.
    
    Args:
        api: JsonRpcCaller instance
        token: Authentication token (required for first build)
        user_id: User ID (optional, not used but kept for API compatibility)
        force_rebuild: Force rebuilding the catalog even if it exists (default: False)
        
    Returns:
        Dictionary with service information
    """
    global _service_catalog
    
    # Return cached catalog if it exists and we're not forcing a rebuild
    if not force_rebuild and _service_catalog is not None:
        print("Using pre-built service catalog", file=sys.stderr)
        return _service_catalog
    
    # Build the catalog (only happens once, or when force_rebuild=True)
    print("Building service catalog (this happens once at startup or first use)...", file=sys.stderr)
    start_time = time.time()
    services_json = await enumerate_apps(api, token, user_id)
    api_time = time.time() - start_time
    print(f"API call took {api_time:.2f} seconds", file=sys.stderr)
    
    services_data = json.loads(services_json) if isinstance(services_json, str) else services_json
    
    # Load service name mapping
    service_mapping = load_config_file('service_mapping.json')
    friendly_to_api = service_mapping['friendly_to_api']
    api_to_friendly = {v: k for k, v in friendly_to_api.items()}
    
    # Extract service list
    if isinstance(services_data, list) and len(services_data) > 0:
        if isinstance(services_data[0], list):
            apps_list = services_data[0]
        else:
            apps_list = services_data
    else:
        apps_list = []
    
    # Build catalog
    catalog = {
        "services": [],
        "mapping": {
            "friendly_to_api": friendly_to_api,
            "api_to_friendly": api_to_friendly
        }
    }
    
    for app in apps_list:
        if not isinstance(app, dict) or 'id' not in app:
            continue
            
        api_name = app['id']
        friendly_name = api_to_friendly.get(api_name)
        
        if not friendly_name:
            continue
        
        # Get service description from prompt file
        try:
            description = get_service_info(friendly_name)
        except Exception:
            description = f"Service: {friendly_name}"
        
        catalog["services"].append({
            "friendly_name": friendly_name,
            "api_name": api_name,
            "description": description
        })
    
    # Store the catalog for reuse
    _service_catalog = catalog
    print(f"Service catalog built with {len(catalog['services'])} services", file=sys.stderr)
    
    return catalog


async def select_services_for_workflow(
    user_query: str,
    api: JsonRpcCaller,
    token: str,
    user_id: str,
    llm_client: LLMClient
) -> Dict[str, Any]:
    """
    STEP 1: Select which services are needed for the workflow.
    
    Args:
        user_query: User's workflow description
        api: JsonRpcCaller instance
        token: Authentication token
        user_id: User ID for workspace paths
        llm_client: LLM client instance
        
    Returns:
        Dictionary with:
        - services: List of friendly service names
        - reasoning: Why these services were selected
        - error: Error message if selection failed
    """
    try:
        print("STEP 1: Selecting services for workflow...", file=sys.stderr)
        
        # Load service selection system prompt
        system_prompt = load_prompt_file('workflow_service_selection.txt')
        
        # Build simple user prompt with just the query
        user_prompt = f"""Select the appropriate BV-BRC services for this user request:

USER QUERY: {user_query}

Return a JSON object with "services" array and "reasoning" string."""
        
        # Make the LLM call
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = llm_client.chat_completion(messages)
        print(f"Service selection response: {response[:200]}...", file=sys.stderr)
        
        # Parse the response
        response_text = response.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            lines = lines[1:]  # Remove first line
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]  # Remove last line
            response_text = "\n".join(lines)
        
        selection = json.loads(response_text)
        
        if "services" not in selection or not isinstance(selection["services"], list):
            return {
                "error": "Invalid service selection format",
                "raw_response": response
            }
        
        print(f"Selected services: {selection['services']}", file=sys.stderr)
        print(f"Reasoning: {selection.get('reasoning', 'N/A')}", file=sys.stderr)
        
        return selection
        
    except json.JSONDecodeError as e:
        print(f"Failed to parse service selection as JSON: {e}", file=sys.stderr)
        return {
            "error": f"Failed to parse service selection: {str(e)}",
            "raw_response": response
        }
    except Exception as e:
        import traceback
        print(f"Error in select_services_for_workflow: {traceback.format_exc()}", file=sys.stderr)
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


async def generate_workflow_with_services(
    user_query: str,
    selected_services: List[str],
    api: JsonRpcCaller,
    token: str,
    user_id: str,
    llm_client: LLMClient
) -> str:
    """
    STEP 2: Generate workflow manifest with detailed parameters for selected services.
    
    Args:
        user_query: User's workflow description
        selected_services: List of friendly service names (from step 1)
        api: JsonRpcCaller instance
        token: Authentication token
        user_id: User ID for workspace paths
        llm_client: LLM client instance
        
    Returns:
        JSON string containing the workflow manifest
    """
    try:
        print(f"STEP 2: Generating workflow for services: {selected_services}", file=sys.stderr)
        
        # Get service catalog
        catalog = await build_service_catalog(api, token, user_id, force_rebuild=False)
        
        # Load configuration files
        output_patterns = load_config_file('service_outputs.json')
        
        # Add job_output_path field to every service output
        for service_name, service_outputs in output_patterns.items():
            if isinstance(service_outputs, dict):
                service_outputs['job_output_path'] = "${params.output_path}/.${params.output_file}"
        
        # Get DETAILED specifications for ONLY the selected services
        selected_service_specs = []
        for service in catalog['services']:
            if service['friendly_name'] in selected_services:
                selected_service_specs.append({
                    "friendly_name": service['friendly_name'],
                    "api_name": service['api_name'],
                    "description": service['description']
                })
        
        # Get output patterns for ONLY the selected services
        selected_output_patterns = {}
        for service_name in selected_services:
            if service_name in output_patterns:
                selected_output_patterns[service_name] = output_patterns[service_name]
        
        # Load parameter generation system prompt
        system_prompt = load_prompt_file('workflow_parameter_generation.txt')
        
        # Build focused user prompt with ONLY selected service details
        user_prompt = f"""Generate a complete workflow manifest for the following user request using THESE SPECIFIC SERVICES:

USER QUERY: {user_query}

SELECTED SERVICES: {json.dumps(selected_services)}

DETAILED SERVICE SPECIFICATIONS (read these carefully for exact parameter names and requirements):
{json.dumps(selected_service_specs, indent=2)}

SERVICE OUTPUT PATTERNS for selected services:
{json.dumps(selected_output_patterns, indent=2)}

Generate a complete workflow manifest with ALL required parameters for each service. Return ONLY the JSON manifest."""
        
        # Make the LLM call
        print("Generating workflow parameters...", file=sys.stderr)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        llm_start = time.time()
        response = llm_client.chat_completion(messages)
        llm_time = time.time() - llm_start
        print(f"Parameter generation completed in {llm_time:.2f} seconds", file=sys.stderr)
        
        # Parse the response
        response_text = response.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            response_text = "\n".join(lines)
        
        # Parse JSON
        workflow_manifest = json.loads(response_text)
        
        # Update workspace_output_folder with actual user_id
        if 'base_context' in workflow_manifest:
            if 'workspace_output_folder' in workflow_manifest['base_context']:
                workspace_path = workflow_manifest['base_context']['workspace_output_folder']
                workflow_manifest['base_context']['workspace_output_folder'] = workspace_path.replace('/USERNAME/', f'/{user_id}/')
            else:
                workflow_manifest['base_context']['workspace_output_folder'] = f"/{user_id}/home/WorkspaceOutputFolder"
            # Remove workspace_root if it exists
            if 'workspace_root' in workflow_manifest['base_context']:
                del workflow_manifest['base_context']['workspace_root']
        else:
            workflow_manifest['base_context'] = {
                "base_url": "https://www.bv-brc.org",
                "workspace_output_folder": f"/{user_id}/home/WorkspaceOutputFolder"
            }
        
        return json.dumps(workflow_manifest, indent=2)
        
    except json.JSONDecodeError as e:
        print(f"Failed to parse workflow as JSON: {e}", file=sys.stderr)
        return json.dumps({
            "error": f"Failed to parse workflow: {str(e)}",
            "raw_response": response,
            "hint": "The LLM response was not valid JSON"
        }, indent=2)
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in generate_workflow_with_services: {error_trace}", file=sys.stderr)
        return json.dumps({
            "error": str(e),
            "traceback": error_trace
        }, indent=2)


async def generate_workflow_manifest_internal(
    user_query: str,
    api: JsonRpcCaller,
    token: str,
    user_id: str,
    llm_client: LLMClient
) -> str:
    """
    Generate a workflow manifest using TWO-STEP LLM-based planning.
    
    Step 1: Select appropriate services
    Step 2: Generate detailed parameters for selected services
    
    Args:
        user_query: User's workflow description
        api: JsonRpcCaller instance
        token: Authentication token
        user_id: User ID for workspace paths
        llm_client: LLM client instance
        
    Returns:
        JSON string containing the workflow manifest
    """
    try:
        # STEP 1: Select services
        selection_result = await select_services_for_workflow(
            user_query=user_query,
            api=api,
            token=token,
            user_id=user_id,
            llm_client=llm_client
        )
        
        # Check if selection failed
        if "error" in selection_result:
            return json.dumps({
                "error": "Service selection failed",
                "details": selection_result.get("error"),
                "hint": "Failed to determine which services are needed"
            }, indent=2)
        
        selected_services = selection_result.get("services", [])
        if not selected_services:
            return json.dumps({
                "error": "No services selected",
                "hint": "Could not identify appropriate services for the query"
            }, indent=2)
        
        # STEP 2: Generate workflow with detailed parameters
        workflow_json_str = await generate_workflow_with_services(
            user_query=user_query,
            selected_services=selected_services,
            api=api,
            token=token,
            user_id=user_id,
            llm_client=llm_client
        )
        
        return workflow_json_str
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in generate_workflow_manifest_internal: {error_trace}", file=sys.stderr)
        return json.dumps({
            "error": str(e),
            "traceback": error_trace
        }, indent=2)


def validate_workflow_structure(workflow_json: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Perform basic structural validation of workflow JSON before submission.
    
    This only checks for basic required fields. Complex validation (circular dependencies,
    DAG structure, parameter validation, etc.) is handled by the workflow engine itself.
    
    Args:
        workflow_json: The workflow manifest to validate
        
    Returns:
        Tuple of (is_valid, error_message)
        - (True, None) if basic structure is valid
        - (False, error_message) if basic structure is invalid
    """
    try:
        # Check required top-level fields
        required_fields = ['workflow_name', 'steps', 'base_context']
        for field in required_fields:
            if field not in workflow_json:
                return False, f"Missing required field: {field}"
        
        # Check steps is a non-empty list
        steps = workflow_json.get('steps', [])
        if not isinstance(steps, list):
            return False, "steps must be a list"
        if len(steps) == 0:
            return False, "Workflow must contain at least one step"
        
        # Basic step structure check
        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                return False, f"Step {i} is not a dictionary"
            
            # Check required step fields
            step_required = ['step_name', 'app', 'params']
            for field in step_required:
                if field not in step:
                    return False, f"Step {i} missing required field: {field}"
        
        # All basic checks passed - let workflow engine handle complex validation
        return True, None
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"


async def create_and_execute_workflow_internal(
    user_query: str,
    api: JsonRpcCaller,
    token: str,
    user_id: str,
    llm_client: LLMClient,
    auto_execute: bool = True,
    workflow_engine_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate a workflow from natural language, validate it, and optionally submit it for execution.
    
    This is the complete end-to-end workflow creation and submission function.
    
    Args:
        user_query: User's workflow description
        api: JsonRpcCaller instance
        token: Authentication token
        user_id: User ID for workspace paths
        llm_client: LLM client instance
        auto_execute: If True, submit workflow to engine after generation (default: True)
        workflow_engine_config: Workflow engine configuration dict (optional)
        
    Returns:
        Dictionary with workflow_id, status, and workflow_json if successful,
        or error information if any stage fails
    """
    from common.workflow_engine_client import WorkflowEngineClient, WorkflowEngineError
    
    # Stage 1: Generate workflow
    print("Stage 1: Generating workflow manifest...", file=sys.stderr)
    try:
        workflow_json_str = await generate_workflow_manifest_internal(
            user_query=user_query,
            api=api,
            token=token,
            user_id=user_id,
            llm_client=llm_client
        )
        
        # Parse the JSON string
        workflow_json = json.loads(workflow_json_str)
        
        # Check if generation returned an error
        if "error" in workflow_json:
            return {
                "error": workflow_json.get("error"),
                "errorType": "GENERATION_FAILED",
                "stage": "generation",
                "hint": workflow_json.get("hint", "Failed to generate workflow"),
                "source": "bvbrc-service"
            }
        
        print("Stage 1: Workflow generated successfully", file=sys.stderr)
        
    except json.JSONDecodeError as e:
        return {
            "error": f"Failed to parse generated workflow: {str(e)}",
            "errorType": "GENERATION_FAILED",
            "stage": "generation",
            "source": "bvbrc-service"
        }
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "errorType": "GENERATION_FAILED",
            "stage": "generation",
            "traceback": traceback.format_exc(),
            "source": "bvbrc-service"
        }
    
    # Stage 2: Basic validation (complex validation happens in workflow engine)
    print("Stage 2: Validating basic workflow structure...", file=sys.stderr)
    is_valid, error_message = validate_workflow_structure(workflow_json)
    
    if not is_valid:
        return {
            "error": error_message,
            "errorType": "VALIDATION_FAILED",
            "stage": "validation",
            "partial_workflow": workflow_json,
            "hint": "The generated workflow has basic structural issues",
            "source": "bvbrc-service"
        }
    
    print("Stage 2: Basic validation passed (detailed validation will occur in workflow engine)", file=sys.stderr)
    
    # If auto_execute is False, return the validated workflow JSON only
    if not auto_execute:
        print("auto_execute=False, returning workflow JSON only", file=sys.stderr)
        return {
            "workflow_json": workflow_json,
            "message": "Workflow manifest generated and validated (not submitted for execution)",
            "source": "bvbrc-service"
        }
    
    # Check if workflow engine is enabled
    if not workflow_engine_config or not workflow_engine_config.get('enabled', False):
        print("Workflow engine is disabled, returning workflow JSON only", file=sys.stderr)
        return {
            "workflow_json": workflow_json,
            "warning": "Workflow engine is disabled in configuration",
            "message": "Workflow generated but not submitted",
            "hint": "Enable workflow_engine in config.json to submit workflows for execution",
            "source": "bvbrc-service"
        }
    
    # Stage 3: Submit to workflow engine
    print("Stage 3: Submitting workflow to execution engine...", file=sys.stderr)
    try:
        engine_url = workflow_engine_config.get('api_url', 'http://localhost:8000/api/v1')
        engine_timeout = workflow_engine_config.get('timeout', 30)
        
        client = WorkflowEngineClient(base_url=engine_url, timeout=engine_timeout)
        
        # First check if engine is healthy
        is_healthy = await client.health_check()
        if not is_healthy:
            print("Workflow engine health check failed", file=sys.stderr)
            return {
                "workflow_json": workflow_json,
                "warning": "Workflow engine is not available",
                "message": "Workflow generated but not submitted",
                "hint": f"Ensure workflow engine is running at {engine_url}",
                "submission_url": f"{engine_url}/workflows/submit",
                "source": "bvbrc-service"
            }
        
        # Clean the workflow before submission - remove any fields that workflow engine assigns
        # The workflow engine assigns workflow_id and step_ids, so we must remove them if present
        workflow_for_submission = workflow_json.copy()
        workflow_for_submission.pop('workflow_id', None)  # Remove if LLM added it
        workflow_for_submission.pop('status', None)  # Remove if present
        workflow_for_submission.pop('created_at', None)  # Remove if present
        workflow_for_submission.pop('updated_at', None)  # Remove if present
        
        # Clean steps - remove execution metadata
        if 'steps' in workflow_for_submission:
            for step in workflow_for_submission['steps']:
                step.pop('step_id', None)  # Workflow engine assigns this
                step.pop('status', None)  # Execution metadata
                step.pop('task_id', None)  # Execution metadata
        
        # Submit the workflow
        result = await client.submit_workflow(workflow_for_submission, token)
        
        print(f"Stage 3: Workflow submitted successfully: {result.get('workflow_id')}", file=sys.stderr)
        
        # Return success response
        return {
            "workflow_id": result.get('workflow_id'),
            "status": result.get('status', 'pending'),
            "workflow_json": workflow_json,
            "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "message": result.get('message', 'Workflow created and submitted for execution'),
            "status_url": f"{engine_url}/workflows/{result.get('workflow_id')}/status",
            "source": "bvbrc-service"
        }
        
    except WorkflowEngineError as e:
        print(f"Workflow engine error: {e}", file=sys.stderr)
        return {
            "error": str(e),
            "errorType": e.error_type,
            "stage": "submission",
            "workflow_json": workflow_json,
            "hint": "Workflow was generated and validated but could not be submitted. You can submit it manually.",
            "submission_url": f"{workflow_engine_config.get('api_url', 'http://localhost:8000/api/v1')}/workflows/submit",
            "source": "bvbrc-service"
        }
    except Exception as e:
        import traceback
        print(f"Unexpected error during submission: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return {
            "error": str(e),
            "errorType": "SUBMISSION_FAILED",
            "stage": "submission",
            "workflow_json": workflow_json,
            "hint": "Workflow was generated and validated but submission failed unexpectedly",
            "traceback": traceback.format_exc(),
            "source": "bvbrc-service"
        }

