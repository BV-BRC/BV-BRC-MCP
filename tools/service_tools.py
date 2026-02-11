"""
BVBRC Service MCP Tools

This module contains all the MCP tool functions for the BVBRC Service MCP Server.
All tools are registered with the FastMCP server instance.

Refactored to use a unified service submission pattern similar to data_tools.py
"""
import sys
import json
from fastmcp import FastMCP
from common.json_rpc import JsonRpcCaller
from functions.service_functions import (
    enumerate_apps, start_date_app, start_genome_annotation_app, query_tasks,
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
from functions.workflow_functions import generate_workflow_manifest_internal, create_and_execute_workflow_internal
from typing import Any, List, Dict, Optional


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

    @mcp.tool(name="get_job_details")
    async def service_get_job_details(task_ids: List[str] = None, token: Optional[str] = None) -> str:
        """
        Query the status and details of submitted jobs/tasks.
        
        Args:
            task_ids: List of task IDs to query (obtained from service submission results)
            token: Authentication token (optional - will use default if not provided)
            
        Returns:
            JSON string with detailed information about each task including status, progress, and results
        """
        if not task_ids:
            return "Error: task_ids parameter is required"
            
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return "Error: No authentication token available"
        
        user_id = extract_userid_from_token(auth_token)
        params = {"task_ids": task_ids}
        return await query_tasks(api, token=auth_token, user_id=user_id, params=params)

    # Main Service Submission Tool
    
    @mcp.tool(name="submit_service")
    async def submit_service(service_name: str = None, parameters: Dict[str, Any] = None, token: Optional[str] = None) -> str:
        """
        Submit a service job to BV-BRC. This is the unified tool for submitting any BV-BRC service.
        
        IMPORTANT: Always use the get_service_submission_schema tool first to understand the required parameters for the service.
        
        Args:
            service_name: Name of the service to submit (e.g., 'genome_assembly', 'blast', 'rnaseq')
            parameters: Dictionary of service-specific parameters as documented by get_service_submission_schema
            token: Authentication token (optional - will use default if not provided)
            
        Returns:
            JSON string with job submission result including task ID for tracking
            
        Available services:
            Genomics: genome_assembly, genome_annotation, comprehensive_genome_analysis, blast, 
                     primer_design, variation, tnseq
            Phylogenomics: bacterial_genome_tree, gene_tree, core_genome_mlst, whole_genome_snp
            Metagenomics: taxonomic_classification, metagenomic_binning, metagenomic_read_mapping
            Transcriptomics: rnaseq, expression_import
            Viral: sars_wastewater_analysis, sequence_submission, influenza_ha_subtype_conversion,
                   subspecies_classification, viral_assembly, sars_genome_analysis
            Other: genome_alignment, msa_snp_analysis, metacats, proteome_comparison,
                   comparative_systems, docking, similar_genome_finder, fastqutils, date
        
        Example usage:
            1. First call: get_service_submission_schema(service_name="blast")
            2. Then call: submit_service(
                service_name="blast",
                parameters={
                    "input_type": "dna_fasta",
                    "input_source": "fasta_data",
                    "input_fasta_data": ">seq1\\nATCG...",
                    "db_type": "dna",
                    "db_source": "genome_list",
                    "db_genome_list": ["123.456"],
                    "blast_program": "blastn",
                    "output_path": "MyFolder",
                    "output_file": "blast_results"
                }
            )
        """
        if not service_name:
            available = sorted(list(SERVICE_MAP.keys()) + list(SPECIAL_API_SERVICES.keys()))
            return f"Error: service_name parameter is required\n\nAvailable services: {', '.join(available)}"
        
        # Convert BV-BRC name to friendly name if needed
        if service_name in BVBRC_TO_FRIENDLY:
            service_name = BVBRC_TO_FRIENDLY[service_name]
        
        if parameters is None:
            return f"Error: parameters dictionary is required. Use get_service_submission_schema(service_name='{service_name}') to see required parameters."
        
        # Get authentication token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return "Error: No authentication token available"
        
        # Extract user ID for path resolution
        user_id = extract_userid_from_token(auth_token)
        
        # Check if service exists in special API services
        if service_name in SPECIAL_API_SERVICES:
            special_api, service_func = SPECIAL_API_SERVICES[service_name]
            try:
                return await service_func(special_api, token=auth_token, user_id=user_id, **parameters)
            except TypeError as e:
                return f"Error: Invalid parameters for service '{service_name}'. Use get_service_submission_schema(service_name='{service_name}') to see correct parameters.\n\nDetails: {str(e)}"
            except Exception as e:
                return f"Error submitting service '{service_name}': {str(e)}"
        
        # Check if service exists in regular services
        if service_name not in SERVICE_MAP:
            available = sorted(list(SERVICE_MAP.keys()) + list(SPECIAL_API_SERVICES.keys()))
            return f"Error: Unknown service '{service_name}'.\n\nAvailable services:\n" + "\n".join(f"  - {svc}" for svc in available)
        
        # Get the service function
        service_func = SERVICE_MAP[service_name]
        
        # Submit the service with the provided parameters
        try:
            result = await service_func(api, token=auth_token, user_id=user_id, **parameters)
            return result
        except TypeError as e:
            return f"Error: Invalid parameters for service '{service_name}'. Use get_service_submission_schema(service_name='{service_name}') to see correct parameters.\n\nDetails: {str(e)}"
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Exception in submit_service for '{service_name}': {error_trace}", file=sys.stderr)
            return f"Error submitting service '{service_name}': {str(e)}\n\nUse get_service_submission_schema(service_name='{service_name}') to verify parameters."

    @mcp.tool(name="create_and_execute_workflow")
    async def create_and_execute_workflow(
        user_query: str = None, 
        auto_execute: bool = True,
        token: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a workflow from natural language description and optionally execute it.
        
        This tool provides complete end-to-end workflow lifecycle management:
        1. Generation: Uses LLM to analyze your request and generate a workflow manifest
        2. Basic Validation: Checks basic structure (detailed validation by workflow engine)
        3. Execution: Submits the workflow to the workflow engine for execution
        
        Args:
            user_query: Natural language description of the desired workflow.
                       Examples:
                       - "Assemble my reads and then annotate the resulting genome"
                       - "Perform comprehensive genome analysis for E. coli"
                       - "Map RNA-seq reads to reference genome and analyze expression"
                       - "Run BLAST on my sequences then build a phylogenetic tree"
            
            auto_execute: If True (default), submit workflow for execution after generation.
                         If False, only generate and validate the workflow without executing.
                         
            token: Authentication token (optional - will use default if not provided)
            
            session_id: Optional session ID for retrieving session facts to enhance workflow generation
            
        Returns:
            Dictionary with:
            - If auto_execute=True and successful:
              {
                "workflow_id": "wf_123...",
                "status": "pending",
                "workflow_json": {...},
                "submitted_at": "2026-02-04T10:30:00Z",
                "message": "Workflow created and submitted for execution",
                "status_url": "http://.../workflows/wf_123/status"
              }
              
            - If auto_execute=False:
              {
                "workflow_json": {...},
                "message": "Workflow generated and validated (not submitted)"
              }
              
            - On error:
              {
                "error": "Error description",
                "errorType": "GENERATION_FAILED | VALIDATION_FAILED | SUBMISSION_FAILED",
                "stage": "generation | validation | submission",
                "hint": "Helpful suggestion",
                "partial_workflow": {...}  // If available
              }
        
        Notes:
            - The workflow engine must be running and configured for execution
            - If the workflow engine is unavailable, the tool will return the generated 
              workflow JSON with a warning
            - Use get_job_details() to check the status of executed workflows
        """
        if not user_query:
            return {
                "error": "user_query parameter is required",
                "errorType": "INVALID_PARAMETERS",
                "example": "create_and_execute_workflow(user_query='Assemble reads and annotate the genome')",
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
            import os
            import json as json_lib
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.json')
            
            with open(config_path, 'r') as f:
                config = json_lib.load(f)
            
            # Create LLM client
            from common.llm_client import create_llm_client_from_config
            llm_client = create_llm_client_from_config(config)
            
            # Get workflow engine configuration
            workflow_engine_config = config.get('workflow_engine', {})
            
            # Create and optionally execute workflow
            result = await create_and_execute_workflow_internal(
                user_query=user_query,
                api=api,
                token=auth_token,
                user_id=user_id,
                llm_client=llm_client,
                auto_execute=auto_execute,
                workflow_engine_config=workflow_engine_config,
                session_id=session_id
            )
            
            return result
            
        except FileNotFoundError as e:
            return {
                "error": f"Configuration file not found: {str(e)}",
                "errorType": "CONFIGURATION_ERROR",
                "hint": "Ensure config/config.json exists with 'llm' and 'workflow_engine' sections",
                "source": "bvbrc-service"
            }
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error in create_and_execute_workflow: {error_trace}", file=sys.stderr)
            return {
                "error": str(e),
                "errorType": "UNKNOWN_ERROR",
                "stage": "initialization",
                "traceback": error_trace,
                "source": "bvbrc-service"
            }