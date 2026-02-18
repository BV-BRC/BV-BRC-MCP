"""
BVBRC Service MCP Tools

This module contains all the MCP tool functions for the BVBRC Service MCP Server.
All tools are registered with the FastMCP server instance.

Refactored to use a unified service submission pattern similar to data_tools.py
"""
import sys
import json
import time
import os
import traceback
from fastmcp import FastMCP
from common.json_rpc import JsonRpcCaller
from common.llm_client import create_llm_client_from_config
from common.workflow_engine_client import WorkflowEngineClient, WorkflowEngineError
from functions.service_functions import (
    enumerate_apps, start_date_app, start_genome_annotation_app, query_tasks, list_jobs,
    start_genome_assembly_app, start_comprehensive_genome_analysis_app, start_blast_app,
    start_primer_design_app, start_variation_app, start_tnseq_app, start_bacterial_genome_tree_app,
    start_gene_tree_app, start_core_genome_mlst_app, start_whole_genome_snp_app,
    start_taxonomic_classification_app, start_metagenomic_binning_app, start_metagenomic_read_mapping_app,
    start_rnaseq_app, start_expression_import_app, start_sars_wastewater_analysis_app,
    start_sequence_submission_app, start_influenza_ha_subtype_conversion_app,
    start_subspecies_classification_app, start_viral_assembly_app, start_fastq_utils_app,
    start_genome_alignment_app, start_sars_genome_analysis_app, start_msa_snp_analysis_app,
    start_metacats_app, start_proteome_comparison_app, start_comparative_systems_app,
    start_docking_app, start_similar_genome_finder_app, get_service_info
)
from functions.workflow_functions import (
    generate_workflow_manifest_internal,
    create_and_execute_workflow_internal,
    prepare_workflow_for_engine_validation,
)
from typing import Any, List, Dict, Optional, Union


def extract_userid_from_token(token: str = None) -> str:
    """
    Extract user ID from JWT token.
    Returns a default user ID if token is None or invalid.
    """
    if not token:
        return None

    try:
        user_id = token.split('|')[0].replace('un=','')
        return user_id

    except Exception as e:
        print(f"Error extracting user ID from token: {e}")
        return None


def register_service_tools(mcp: FastMCP, api: JsonRpcCaller, similar_genome_finder_api: JsonRpcCaller, token_provider):
    """
    Register all MCP tools with the FastMCP server instance.

    Args:
        mcp: FastMCP server instance
        api: Main service API caller
        similar_genome_finder_api: Similar genome finder API caller
        token_provider: TokenProvider instance for handling authentication tokens
    """

    # Mapping between BV-BRC API service names and user-friendly names
    # API_NAME -> user_friendly_name
    BVBRC_TO_FRIENDLY = {
        'Date': 'date',
        'GenomeAssembly2': 'genome_assembly',
        'GenomeAnnotation': 'genome_annotation',
        'ComprehensiveGenomeAnalysis': 'comprehensive_genome_analysis',
        'Homology': 'blast',
        'PrimerDesign': 'primer_design',
        'Variation': 'variation',
        'TnSeq': 'tnseq',
        'CodonTree': 'bacterial_genome_tree',
        'GeneTree': 'gene_tree',
        'CoreGenomeMLST': 'core_genome_mlst',
        'WholeGenomeSNPAnalysis': 'whole_genome_snp',
        'TaxonomicClassification': 'taxonomic_classification',
        'MetagenomeBinning': 'metagenomic_binning',
        'MetagenomicReadMapping': 'metagenomic_read_mapping',
        'RNASeq': 'rnaseq',
        'ExpressionImport': 'expression_import',
        'SARSWastewaterAnalysis': 'sars_wastewater_analysis',
        'SequenceSubmission': 'sequence_submission',
        'InfluenzaHASubtypeConversion': 'influenza_ha_subtype_conversion',
        'HASubtypeNumberingConversion': 'influenza_ha_subtype_conversion',
        'SubspeciesClassification': 'subspecies_classification',
        'ViralAssembly': 'viral_assembly',
        'GenomeAlignment': 'genome_alignment',
        'SARS2Assembly': 'sars_genome_analysis',
        'MSA': 'msa_snp_analysis',
        'MetaCATS': 'metacats',
        'GenomeComparison': 'proteome_comparison',
        'ComparativeSystems': 'comparative_systems',
        'Docking': 'docking',
        'SimilarGenomeFinder': 'similar_genome_finder',
        'FastqUtils': 'fastqutils',
    }

    # Reverse mapping: user_friendly_name -> API_NAME
    FRIENDLY_TO_BVBRC = {v: k for k, v in BVBRC_TO_FRIENDLY.items()}

    # Service mapping: maps user-friendly service names to their handler functions
    SERVICE_MAP = {
        # Basic Services
        'date': start_date_app,

        # Genomics Analysis Services
        'genome_assembly': start_genome_assembly_app,
        'genome_annotation': start_genome_annotation_app,
        'comprehensive_genome_analysis': start_comprehensive_genome_analysis_app,
        'blast': start_blast_app,
        'primer_design': start_primer_design_app,
        'variation': start_variation_app,
        'tnseq': start_tnseq_app,

        # Phylogenomics Services
        'bacterial_genome_tree': start_bacterial_genome_tree_app,
        'gene_tree': start_gene_tree_app,
        'core_genome_mlst': start_core_genome_mlst_app,
        'whole_genome_snp': start_whole_genome_snp_app,

        # Metagenomics Services
        'taxonomic_classification': start_taxonomic_classification_app,
        'metagenomic_binning': start_metagenomic_binning_app,
        'metagenomic_read_mapping': start_metagenomic_read_mapping_app,

        # Transcriptomics Services
        'rnaseq': start_rnaseq_app,
        'expression_import': start_expression_import_app,

        # Viral Services
        'sars_wastewater_analysis': start_sars_wastewater_analysis_app,
        'sequence_submission': start_sequence_submission_app,
        'influenza_ha_subtype_conversion': start_influenza_ha_subtype_conversion_app,
        'subspecies_classification': start_subspecies_classification_app,
        'viral_assembly': start_viral_assembly_app,

        # Additional Services
        'genome_alignment': start_genome_alignment_app,
        'sars_genome_analysis': start_sars_genome_analysis_app,
        'msa_snp_analysis': start_msa_snp_analysis_app,
        'metacats': start_metacats_app,
        'proteome_comparison': start_proteome_comparison_app,
        'comparative_systems': start_comparative_systems_app,
        'docking': start_docking_app,
        'fastqutils': start_fastq_utils_app,
    }

    # Special case for similar_genome_finder which uses a different API
    SPECIAL_API_SERVICES = {
        'similar_genome_finder': (similar_genome_finder_api, start_similar_genome_finder_app),
    }

    # Helper Tools

    @mcp.tool(name="list_service_apps", annotations={"readOnlyHint": True})
    async def service_enumerate_apps(token: Optional[str] = None) -> str:
        """
        Enumerate all available BV-BRC service apps.

        Returns:
            JSON array of user-friendly service names (e.g., ["blast", "genome_assembly", "rnaseq", ...])
        """
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return "Error: No authentication token available"

        user_id = extract_userid_from_token(auth_token)
        result = await enumerate_apps(api, auth_token, user_id=user_id)

        # Parse the result and extract only the service IDs
        try:
            # Result is a JSON string, parse it
            apps_data = json.loads(result) if isinstance(result, str) else result

            # Extract IDs from the apps list
            # The structure is typically [[app1, app2, ...]] or [app1, app2, ...]
            if isinstance(apps_data, list) and len(apps_data) > 0:
                # If it's nested (list of lists), flatten it
                if isinstance(apps_data[0], list):
                    apps_list = apps_data[0]
                else:
                    apps_list = apps_data

                # Extract the 'id' field from each app and map to user-friendly names
                bvbrc_service_ids = [app.get('id') for app in apps_list if isinstance(app, dict) and 'id' in app]

                # Convert BV-BRC service names to user-friendly names
                # Only include services we have mappings for
                friendly_names = []
                for bvbrc_id in bvbrc_service_ids:
                    if bvbrc_id in BVBRC_TO_FRIENDLY:
                        friendly_names.append(BVBRC_TO_FRIENDLY[bvbrc_id])

                return {
                    "services": sorted(friendly_names),
                    "count": len(friendly_names),
                    "source": "bvbrc-service"
                }

            return {
                "services": [],
                "count": 0,
                "source": "bvbrc-service"
            }
        except Exception as e:
            print(f"Error parsing service list: {e}", file=sys.stderr)
            return {
                "error": f"Error parsing service list: {str(e)}",
                "errorType": "API_ERROR",
                "source": "bvbrc-service"
            }

    @mcp.tool(name="get_job_details", annotations={"readOnlyHint": True})
    async def service_get_job_details(task_ids: Optional[List[Union[str, int]]] = None, token: Optional[str] = None) -> Dict[str, Any]:
        """
        Query task details by task IDs.

        Args:
            task_ids: List of task IDs to query (can be strings or numbers)
            token: Authentication token (optional - will use default if not provided)
        """
        if not task_ids or not isinstance(task_ids, list):
            return {
                "error": "task_ids (list) parameter is required",
                "errorType": "INVALID_PARAMETERS",
                "source": "bvbrc-service"
            }

        auth_token = token_provider.get_token(token)
        if not auth_token:
            return {
                "error": "No authentication token available",
                "errorType": "AUTHENTICATION_FAILED",
                "source": "bvbrc-service"
            }

        # Convert all task IDs to strings (handles both string and numeric IDs)
        task_ids_str = [str(task_id) for task_id in task_ids]

        user_id = extract_userid_from_token(auth_token)
        return await query_tasks(
            api,
            auth_token,
            user_id=user_id,
            params={"task_ids": task_ids_str}
        )

    @mcp.tool(name="list_jobs", annotations={"readOnlyHint": True})
    async def service_list_jobs(
        token: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
        sort_by: str = "submit_time",
        sort_dir: str = "desc",
        status: Optional[str] = None,
        service: Optional[str] = None,
        search: Optional[str] = None,
        include_archived: bool = False
    ) -> Dict[str, Any]:
        """
        List recent jobs with sorting/filtering support.

        Args:
            token: Authentication token (optional)
            limit: Maximum number of jobs to return
            offset: Number of jobs to skip (for pagination)
            sort_by: Field to sort by (e.g., "submit_time", "status")
            sort_dir: Sort direction ("asc" or "desc")
            status: Filter by job status (e.g., "completed", "running", "failed")
            service: Filter by service name. Valid service names include:
                bacterial_genome_tree, blast, comparative_systems, comprehensive_genome_analysis,
                core_genome_mlst, date, docking, expression_import, fastqutils, gene_tree,
                genome_alignment, genome_annotation, genome_assembly, influenza_ha_subtype_conversion,
                metacats, metagenomic_binning, metagenomic_read_mapping, msa_snp_analysis,
                primer_design, proteome_comparison, rnaseq, sars_genome_analysis, sars_wastewater_analysis,
                sequence_submission, similar_genome_finder, subspecies_classification, taxonomic_classification,
                tnseq, variation, viral_assembly, whole_genome_snp
            search: Search term to filter jobs by name or description
            include_archived: Whether to include archived jobs
        """
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return {
                "error": "No authentication token available",
                "errorType": "AUTHENTICATION_FAILED",
                "source": "bvbrc-service"
            }

        # Convert friendly service name to BVBRC API name if needed
        if service and service in FRIENDLY_TO_BVBRC:
            service = FRIENDLY_TO_BVBRC[service]

        user_id = extract_userid_from_token(auth_token)
        return await list_jobs(
            api=api,
            token=auth_token,
            user_id=user_id,
            limit=limit,
            offset=offset,
            sort_by=sort_by,
            sort_dir=sort_dir,
            status=status,
            service=service,
            search=search,
            include_archived=include_archived
        )

    @mcp.tool(name="get_service_submission_schema", annotations={"readOnlyHint": True})
    def service_get_service_submission_schema(service_name: str = None, token: Optional[str] = None) -> str:
        """
        Fetch the parameter/schema details needed immediately before submitting a service job.
        Use the helpdesk tool for any other guidance or questions about which service to run.

        Args:
            service_name: Name of the service to get submission schema for (e.g., 'genome_assembly', 'blast', 'primer_design')
            token: Authentication token (optional - will use default if not provided)

        Returns:
            Structured parameter/schema details required for submission
        """
        if not service_name:
            return "Error: service_name parameter is required"

        auth_token = token_provider.get_token(token)
        if not auth_token:
            return "Error: No authentication token available"

        # Convert BV-BRC name to friendly name if needed
        if service_name in BVBRC_TO_FRIENDLY:
            service_name = BVBRC_TO_FRIENDLY[service_name]

        try:
            return get_service_info(service_name=service_name)
        except Exception as e:
            available = sorted(list(SERVICE_MAP.keys()) + list(SPECIAL_API_SERVICES.keys()))
            return f"Error getting service submission schema: {str(e)}\n\nAvailable services: {', '.join(available)}"



    # Workflow Tools

    @mcp.tool(name="plan_workflow")
    async def plan_workflow(
        user_query: str = None,
        token: Optional[str] = None,
        session_id: Optional[str] = None,
        workspace_items: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Plan a workflow from natural language description without executing it.

        This tool generates a workflow manifest (plan) based on your natural language request.
        The workflow is fully planned and validated (including workflow-engine validation
        when available) but NOT submitted for execution.
        Use submit_workflow() to actually execute the planned workflow.

        This two-step approach allows you to:
        1. Review the planned workflow before execution
        2. Modify parameters if needed
        3. Reuse workflow plans for similar tasks

        Args:
            user_query: Natural language description of the desired workflow.
                       Examples:
                       - "Assemble my reads and then annotate the resulting genome"
                       - "Perform comprehensive genome analysis for E. coli"
                       - "Map RNA-seq reads to reference genome and analyze expression"
                       - "Run BLAST on my sequences then build a phylogenetic tree"

            token: Authentication token (optional - will use default if not provided)

            session_id: Optional session ID for retrieving session facts to enhance workflow generation

            workspace_items: Optional list of workspace items (files, directories, etc.) to include in workflow planning prompts

        Returns:
            Dictionary with workflow identity and summary fields:
              {
                "workflow_id": "wf_...",
                "status": "planned",
                "workflow_name": "...",
                "step_count": 3,
                "workflow_description": "Planned workflow with ...",
                "message": "Workflow planned and saved. Use submit_workflow(workflow_id=...) to execute."
              }
        """
        if not user_query:
            return {
                "error": "user_query parameter is required",
                "errorType": "INVALID_PARAMETERS",
                "example": "plan_workflow(user_query='Assemble reads and annotate the genome')",
                "hint": "Provide a natural language description of your desired workflow",
                "source": "bvbrc-service"
            }

        # Get authentication token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return {
                "error": "No authentication token available",
                "errorType": "AUTHENTICATION_FAILED",
                "hint": "Please provide a valid authentication token",
                "source": "bvbrc-service"
            }

        # Extract user ID
        user_id = extract_userid_from_token(auth_token)
        if not user_id:
            return {
                "error": "Could not extract user ID from token",
                "errorType": "AUTHENTICATION_FAILED",
                "hint": "The provided token is invalid or malformed",
                "source": "bvbrc-service"
            }

        try:
            # Load configuration
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.json')

            with open(config_path, 'r') as f:
                config = json.load(f)

            # Create LLM client
            llm_client = create_llm_client_from_config(config)

            # Get workflow engine configuration
            workflow_engine_config = config.get('workflow_engine', {})

            # Generate and validate workflow manifest first (no execution)
            generated = await create_and_execute_workflow_internal(
                user_query=user_query,
                api=api,
                token=auth_token,
                user_id=user_id,
                llm_client=llm_client,
                auto_execute=False,  # Only plan, don't execute
                workflow_engine_config=workflow_engine_config,
                session_id=session_id,
                workspace_items=workspace_items
            )
            if "error" in generated:
                return generated

            workflow_json = generated.get("workflow_json")
            if not workflow_json or not isinstance(workflow_json, dict):
                return {
                    "error": "Failed to generate workflow plan payload",
                    "errorType": "GENERATION_FAILED",
                    "stage": "planning",
                    "source": "bvbrc-service"
                }

            if not workflow_engine_config or not workflow_engine_config.get('enabled', False):
                return {
                    "error": "Workflow engine is disabled in configuration",
                    "errorType": "ENGINE_UNAVAILABLE",
                    "hint": "Enable workflow_engine in config.json to persist workflow plans",
                    "source": "bvbrc-service"
                }

            engine_url = workflow_engine_config.get('api_url', 'http://localhost:8000/api/v1')
            engine_timeout = workflow_engine_config.get('timeout', 30)
            client = WorkflowEngineClient(base_url=engine_url, timeout=engine_timeout)

            is_healthy = await client.health_check()
            if not is_healthy:
                return {
                    "error": "Workflow engine is not available",
                    "errorType": "ENGINE_UNAVAILABLE",
                    "hint": f"Ensure workflow engine is running at {engine_url}",
                    "source": "bvbrc-service"
                }

            planned = await client.plan_workflow(workflow_json, auth_token)
            workflow_id = planned.get("workflow_id")
            print(f"[DEBUG plan_workflow] Generated workflow_id: {workflow_id}", file=sys.stderr)
            print(f"[DEBUG plan_workflow] Full planned response: {planned}", file=sys.stderr)
            return {
                "workflow_id": workflow_id,
                "status": planned.get("status", "planned"),
                "workflow_name": planned.get("workflow_name", workflow_json.get("workflow_name", "Workflow")),
                "step_count": planned.get("step_count", len(workflow_json.get("steps", []) if isinstance(workflow_json.get("steps"), list) else [])),
                "workflow_description": generated.get("workflow_description"),
                "message": "Workflow planned and saved. Use submit_workflow(workflow_id=...) to execute.",
                "call": {
                    "tool": "plan_workflow",
                    "arguments_executed": {
                        "user_query": user_query,
                        "session_id": session_id,
                        "workspace_items_count": len(workspace_items) if isinstance(workspace_items, list) else 0
                    },
                    "replayable": True
                },
                "source": "bvbrc-service"
            }

        except FileNotFoundError as e:
            return {
                "error": f"Configuration file not found: {str(e)}",
                "errorType": "CONFIGURATION_ERROR",
                "hint": "Ensure config/config.json exists with 'llm' and 'workflow_engine' sections",
                "source": "bvbrc-service"
            }
        except Exception as e:
            error_trace = traceback.format_exc()
            print(f"Error in plan_workflow: {error_trace}", file=sys.stderr)
            return {
                "error": str(e),
                "errorType": "UNKNOWN_ERROR",
                "stage": "initialization",
                "traceback": error_trace,
                "source": "bvbrc-service"
            }

    @mcp.tool(name="submit_workflow")
    async def submit_workflow(
        workflow_id: Optional[str] = None,
        workflow_json: Optional[Dict[str, Any]] = None,
        token: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Submit a planned workflow for execution.

        This tool takes a workflow manifest (typically generated by plan_workflow) and
        submits it to the workflow engine for execution. The workflow engine will:
        1. Perform detailed validation
        2. Schedule and execute workflow steps in sequence
        3. Handle dependencies between steps
        4. Track progress and results

        Args:
            workflow_id: Planned workflow ID (preferred)
            workflow_json: Complete workflow manifest dictionary (fallback)
            token: Authentication token (optional - will use default if not provided)

        Returns:
            Dictionary with:
            - On success:
              {
                "workflow_id": "wf_123...",
                "status": "pending",
                "submitted_at": "2026-02-04T10:30:00Z",
                "message": "Workflow submitted for execution",
                "status_url": "http://.../workflows/wf_123/status"
              }

            - On error:
              {
                "error": "Error description",
                "errorType": "SUBMISSION_FAILED | VALIDATION_FAILED | ENGINE_UNAVAILABLE",
                "hint": "Helpful suggestion"
              }

        Notes:
            - The workflow engine must be running and configured for execution
            - The workflow will be validated by the engine before execution
            - Use workflow monitoring tools to track execution progress
            - The workflow_id can be used to query status and retrieve results

        Example:
            # First, plan the workflow
            plan = plan_workflow(user_query="Assemble and annotate genome")

            # Review the plan, then submit it
            result = submit_workflow(workflow_json=plan["workflow_json"])
        """
        if not workflow_id and not workflow_json:
            return {
                "error": "workflow_id or workflow_json parameter is required",
                "errorType": "INVALID_PARAMETERS",
                "hint": "Provide a planned workflow_id (preferred) or a workflow manifest dictionary",
                "example": "submit_workflow(workflow_id='wf_...')",
                "source": "bvbrc-service"
            }

        workflow_manifest = workflow_json

        # Get authentication token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return {
                "error": "No authentication token available",
                "errorType": "AUTHENTICATION_FAILED",
                "hint": "Please provide a valid authentication token",
                "source": "bvbrc-service"
            }

        # Extract user ID
        user_id = extract_userid_from_token(auth_token)
        if not user_id:
            return {
                "error": "Could not extract user ID from token",
                "errorType": "AUTHENTICATION_FAILED",
                "hint": "The provided token is invalid or malformed",
                "source": "bvbrc-service"
            }

        try:
            # Load configuration
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.json')

            with open(config_path, 'r') as f:
                config = json.load(f)

            # Get workflow engine configuration
            workflow_engine_config = config.get('workflow_engine', {})

            # Check if workflow engine is enabled
            if not workflow_engine_config or not workflow_engine_config.get('enabled', False):
                return {
                    "error": "Workflow engine is disabled in configuration",
                    "errorType": "ENGINE_UNAVAILABLE",
                    "hint": "Enable workflow_engine in config.json to submit workflows for execution",
                    "source": "bvbrc-service"
                }

            # Setup workflow engine client
            engine_url = workflow_engine_config.get('api_url', 'http://localhost:8000/api/v1')
            engine_timeout = workflow_engine_config.get('timeout', 30)

            client = WorkflowEngineClient(base_url=engine_url, timeout=engine_timeout)

            # Check if engine is healthy
            print("Checking workflow engine health...", file=sys.stderr)
            is_healthy = await client.health_check()
            if not is_healthy:
                print("Workflow engine health check failed", file=sys.stderr)
                return {
                    "error": "Workflow engine is not available",
                    "errorType": "ENGINE_UNAVAILABLE",
                    "hint": f"Ensure workflow engine is running at {engine_url}",
                    "submission_url": f"{engine_url}/workflows/submit",
                    "source": "bvbrc-service"
                }

            # Preferred path: submit persisted planned workflow by ID.
            if workflow_id:
                result = await client.submit_planned_workflow(workflow_id, auth_token)
                return {
                    "workflow_id": result.get('workflow_id', workflow_id),
                    "status": result.get('status', 'pending'),
                    "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "message": result.get('message', 'Workflow submitted for execution'),
                    "status_url": f"{engine_url}/workflows/{result.get('workflow_id', workflow_id)}/status",
                    "call": {
                        "tool": "submit_workflow",
                        "arguments_executed": {
                            "workflow_id": workflow_id
                        },
                        "replayable": True
                    },
                    "source": "bvbrc-service"
                }

            # Fallback raw manifest path
            if (
                isinstance(workflow_manifest, dict)
                and isinstance(workflow_manifest.get('workflow_json'), dict)
            ):
                workflow_manifest = workflow_manifest['workflow_json']

            if not isinstance(workflow_manifest, dict):
                return {
                    "error": "workflow_json must be a dictionary when workflow_id is not provided",
                    "errorType": "INVALID_PARAMETERS",
                    "source": "bvbrc-service"
                }

            # Clean workflow before submission using shared helper used in planning path.
            # This strips engine-assigned/execution metadata and any wrapper fields
            # that do not belong to WorkflowDefinition.
            workflow_for_submission = prepare_workflow_for_engine_validation(workflow_manifest)

            # Submit the workflow
            print(f"Submitting workflow to {engine_url}...", file=sys.stderr)
            result = await client.submit_workflow(workflow_for_submission, auth_token)

            print(f"Workflow submitted successfully: {result.get('workflow_id')}", file=sys.stderr)

            # Return success response
            return {
                "workflow_id": result.get('workflow_id'),
                "status": result.get('status', 'pending'),
                "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "message": result.get('message', 'Workflow submitted for execution'),
                "status_url": f"{engine_url}/workflows/{result.get('workflow_id')}/status",
                "call": {
                    "tool": "submit_workflow",
                    "arguments_executed": {
                        "workflow_json": workflow_for_submission
                    },
                    "replayable": True
                },
                "source": "bvbrc-service"
            }

        except WorkflowEngineError as e:
            print(f"Workflow engine error: {e}", file=sys.stderr)
            return {
                "error": str(e),
                "errorType": e.error_type if hasattr(e, 'error_type') else "SUBMISSION_FAILED",
                "hint": "The workflow engine rejected the workflow. Check the error message for details.",
                "source": "bvbrc-service"
            }
        except FileNotFoundError as e:
            return {
                "error": f"Configuration file not found: {str(e)}",
                "errorType": "CONFIGURATION_ERROR",
                "hint": "Ensure config/config.json exists with 'workflow_engine' section",
                "source": "bvbrc-service"
            }
        except Exception as e:
            error_trace = traceback.format_exc()
            print(f"Error in submit_workflow: {error_trace}", file=sys.stderr)
            return {
                "error": str(e),
                "errorType": "UNKNOWN_ERROR",
                "traceback": error_trace,
                "source": "bvbrc-service"
            }
