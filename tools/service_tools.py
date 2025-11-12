"""
BVBRC Service MCP Tools

This module contains all the MCP tool functions for the BVBRC Service MCP Server.
All tools are registered with the FastMCP server instance.
"""

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
    
    # Basic Services
    @mcp.tool(name="list_service_apps", description="Enumerate all apps. Parameters: token (str, optional) - Authentication token for API access")
    def service_enumerate_apps(token: Optional[str] = None) -> str:
        # Get the appropriate token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return ["Error: No authentication token available"]

        user_id = extract_userid_from_token(auth_token)
        return str(enumerate_apps(api, auth_token, user_id=user_id))

    @mcp.tool(name="submit_date_app", description="Start the date app. Parameters: token (str, optional) - Authentication token for API access, output_path (str, required) - Path for output files, output_file (str, required) - Name of output file")
    def submit_date_app(token: Optional[str] = None, output_path: str = None, output_file: str = None) -> str:
        # Get the appropriate token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return "Error: No authentication token available"
        
        user_id = extract_userid_from_token(auth_token)
        return start_date_app(api, token=auth_token, user_id=user_id, output_path=output_path, output_file=output_file)

    @mcp.tool(name="get_service_info", description="Get information about the service. Parameters: Service Name (str, required) - Name of the service to get information about. example: genome_assembly, blast, primer_design, etc.")
    def service_get_service_info(token: Optional[str] = None, service_name: str = None) -> str:
        # Get the appropriate token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return "Error: No authentication token available"
        
        user_id = extract_userid_from_token(auth_token)
        return get_service_info(service_name=service_name)

    # Query Tasks
    @mcp.tool(name="get_job_details", description="Query tasks. Parameters: token (str, optional) - Authentication token for API access, task_ids (List[str]) - List of task IDs to query")
    def get_job_details(token: Optional[str] = None, task_ids: List[str] = None) -> str:
        # Get the appropriate token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return "Error: No authentication token available"
        
        user_id = extract_userid_from_token(auth_token)
        params = {"task_ids": task_ids} if task_ids else None
        return query_tasks(api, token=auth_token, user_id=user_id, params=params)

    # Genomics Analysis Services
    @mcp.tool(name="genome_assembly", description="Get parameters from get_service_info tool.")
    def submit_genome_assembly_app(token: Optional[str] = None, paired_end_libs: List[Dict] = None, single_end_libs: List[Dict] = None, srr_ids: List[str] = None, max_bases: int = 10000000000, recipe: str = "auto", racon_iter: int = 2, pilon_iter: int = 2, trim: bool = False, target_depth: int = 200, normalize: bool = False, filtlong: bool = False, genome_size: int = 5000000, min_contig_len: int = 300, min_contig_cov: float = 5.0, output_path: str = None, output_file: str = None) -> str:
        # Get the appropriate token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return "Error: No authentication token available"
        
        user_id = extract_userid_from_token(auth_token)
        return start_genome_assembly_app(api, token=auth_token, user_id=user_id, paired_end_libs=paired_end_libs, single_end_libs=single_end_libs, srr_ids=srr_ids, max_bases=max_bases, recipe=recipe, racon_iter=racon_iter, pilon_iter=pilon_iter, trim=trim, target_depth=target_depth, normalize=normalize, filtlong=filtlong, genome_size=genome_size, min_contig_len=min_contig_len, min_contig_cov=min_contig_cov, output_path=output_path, output_file=output_file)

    @mcp.tool(name="genome_annotation", description="Get parameters from get_service_info tool.")
    def submit_genome_annotation_app(token: Optional[str] = None, genome_id: str = None, contigs: str = None, scientific_name: str = None, tax_id: str = None, my_label: str = None, taxonomy_id: int = None, code: int = 0, domain: str = "auto", public: bool = False, queue_nowait: bool = False, skip_indexing: bool = False, skip_workspace_output: bool = False, output_path: str = None, output_file: str = None, reference_genome_id: str = None, lowvan_min_contig_length: int = 300, lowvan_max_contig_length: int = 35000, reference_virus_name: str = None, fix_errors: bool = None, fix_frameshifts: bool = None, verbose_level: int = None, workflow: str = None, recipe: str = None, disable_replication: bool = None, analyze_quality: bool = None, assembly_output: str = None, custom_pipeline: Dict = None) -> str:
        # Get the appropriate token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return "Error: No authentication token available"
        
        user_id = extract_userid_from_token(auth_token)
        return start_genome_annotation_app(api, token=auth_token, user_id=user_id, genome_id=genome_id, contigs=contigs, scientific_name=scientific_name, tax_id=tax_id, my_label=my_label, taxonomy_id=taxonomy_id, code=code, domain=domain, public=public, queue_nowait=queue_nowait, skip_indexing=skip_indexing, skip_workspace_output=skip_workspace_output, output_path=output_path, output_file=output_file, reference_genome_id=reference_genome_id, lowvan_min_contig_length=lowvan_min_contig_length, lowvan_max_contig_length=lowvan_max_contig_length, reference_virus_name=reference_virus_name, fix_errors=fix_errors, fix_frameshifts=fix_frameshifts, verbose_level=verbose_level, workflow=workflow, recipe=recipe, disable_replication=disable_replication, analyze_quality=analyze_quality, assembly_output=assembly_output, custom_pipeline=custom_pipeline)


    @mcp.tool(name="comprehensive_genome_analysis", description="Get parameters from get_service_info tool.")
    def submit_comprehensive_genome_analysis_app(token: Optional[str] = None, input_type: str = None, output_path: str = None, output_file: str = None, paired_end_libs: List[Dict] = None, single_end_libs: List[Dict] = None, srr_ids: List[str] = None, reference_assembly: str = None, recipe: str = "auto", racon_iter: int = 2, pilon_iter: int = 2, trim: bool = False, normalize: bool = False, filtlong: bool = False, target_depth: int = 200, genome_size: int = 5000000, min_contig_len: int = 300, min_contig_cov: float = 5.0, gto: str = None, genbank_file: str = None, contigs: str = None, scientific_name: str = None, taxonomy_id: int = None, code: int = 0, domain: str = "auto", public: bool = False, queue_nowait: bool = False, skip_indexing: bool = False, reference_genome_id: str = None, analyze_quality: bool = None) -> str:
        # Get the appropriate token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return "Error: No authentication token available"
        
        user_id = extract_userid_from_token(auth_token)
        return start_comprehensive_genome_analysis_app(api, token=auth_token, user_id=user_id, input_type=input_type, output_path=output_path, output_file=output_file, paired_end_libs=paired_end_libs, single_end_libs=single_end_libs, srr_ids=srr_ids, reference_assembly=reference_assembly, recipe=recipe, racon_iter=racon_iter, pilon_iter=pilon_iter, trim=trim, normalize=normalize, filtlong=filtlong, target_depth=target_depth, genome_size=genome_size, min_contig_len=min_contig_len, min_contig_cov=min_contig_cov, gto=gto, genbank_file=genbank_file, contigs=contigs, scientific_name=scientific_name, taxonomy_id=taxonomy_id, code=code, domain=domain, public=public, queue_nowait=queue_nowait, skip_indexing=skip_indexing, reference_genome_id=reference_genome_id, analyze_quality=analyze_quality)

    @mcp.tool(name="blast", description="Get parameters from get_service_info tool.")
    def submit_blast_app(token: Optional[str] = None, input_type: str = None, input_source: str = None, input_fasta_data: str = None, input_id_list: List[str] = None, input_fasta_file: str = None, input_feature_group: str = None, input_genome_group: str = None, db_type: str = None, db_source: str = None, db_fasta_data: str = None, db_fasta_file: str = None, db_id_list: List[str] = None, db_feature_group: str = None, db_genome_group: str = None, db_genome_list: List[str] = None, db_taxon_list: List[str] = None, db_precomputed_database: str = None, blast_program: str = None, blast_evalue_cutoff: float = 1e-5, blast_max_hits: int = 300, blast_min_coverage: int = None, output_path: str = None, output_file: str = None) -> str:
        # Get the appropriate token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return "Error: No authentication token available"
        
        user_id = extract_userid_from_token(auth_token)
        return start_blast_app(api, token=auth_token, user_id=user_id, input_type=input_type, input_source=input_source, input_fasta_data=input_fasta_data, input_id_list=input_id_list, input_fasta_file=input_fasta_file, input_feature_group=input_feature_group, input_genome_group=input_genome_group, db_type=db_type, db_source=db_source, db_fasta_data=db_fasta_data, db_fasta_file=db_fasta_file, db_id_list=db_id_list, db_feature_group=db_feature_group, db_genome_group=db_genome_group, db_genome_list=db_genome_list, db_taxon_list=db_taxon_list, db_precomputed_database=db_precomputed_database, blast_program=blast_program, blast_evalue_cutoff=blast_evalue_cutoff, blast_max_hits=blast_max_hits, blast_min_coverage=blast_min_coverage, output_path=output_path, output_file=output_file)

    @mcp.tool(name="primer_design", description="Get parameters from get_service_info tool.")
    def submit_primer_design_app(token: Optional[str] = None, output_file: str = None, output_path: str = None, input_type: str = None, sequence_input: str = None, SEQUENCE_ID: str = None, SEQUENCE_TARGET: List[List[int]] = None, SEQUENCE_INCLUDED_REGION: List[int] = None, SEQUENCE_EXCLUDED_REGION: List[int] = None, SEQUENCE_OVERLAP_JUNCTION_LIST: List[List[int]] = None, PRIMER_PICK_INTERNAL_OLIGO: int = None, PRIMER_PRODUCT_SIZE_RANGE: List[List[int]] = None, PRIMER_NUM_RETURN: int = None, PRIMER_MIN_SIZE: int = None, PRIMER_OPT_SIZE: int = None, PRIMER_MAX_SIZE: int = None, PRIMER_MAX_TM: float = None, PRIMER_MIN_TM: float = None, PRIMER_OPT_TM: float = None, PRIMER_PAIR_MAX_DIFF_TM: float = None, PRIMER_MAX_GC: float = None, PRIMER_MIN_GC: float = None, PRIMER_OPT_GC: float = None, PRIMER_SALT_MONOVALENT: float = None, PRIMER_SALT_DIVALENT: float = None, PRIMER_DNA_CONC: float = None, PRIMER_DNTP_CONC: float = None) -> str:
        # Get the appropriate token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return "Error: No authentication token available"
        
        user_id = extract_userid_from_token(auth_token)
        return start_primer_design_app(api, token=auth_token, user_id=user_id, output_file=output_file, output_path=output_path, input_type=input_type, sequence_input=sequence_input, SEQUENCE_ID=SEQUENCE_ID, SEQUENCE_TARGET=SEQUENCE_TARGET, SEQUENCE_INCLUDED_REGION=SEQUENCE_INCLUDED_REGION, SEQUENCE_EXCLUDED_REGION=SEQUENCE_EXCLUDED_REGION, SEQUENCE_OVERLAP_JUNCTION_LIST=SEQUENCE_OVERLAP_JUNCTION_LIST, PRIMER_PICK_INTERNAL_OLIGO=PRIMER_PICK_INTERNAL_OLIGO, PRIMER_PRODUCT_SIZE_RANGE=PRIMER_PRODUCT_SIZE_RANGE, PRIMER_NUM_RETURN=PRIMER_NUM_RETURN, PRIMER_MIN_SIZE=PRIMER_MIN_SIZE, PRIMER_OPT_SIZE=PRIMER_OPT_SIZE, PRIMER_MAX_SIZE=PRIMER_MAX_SIZE, PRIMER_MAX_TM=PRIMER_MAX_TM, PRIMER_MIN_TM=PRIMER_MIN_TM, PRIMER_OPT_TM=PRIMER_OPT_TM, PRIMER_PAIR_MAX_DIFF_TM=PRIMER_PAIR_MAX_DIFF_TM, PRIMER_MAX_GC=PRIMER_MAX_GC, PRIMER_MIN_GC=PRIMER_MIN_GC, PRIMER_OPT_GC=PRIMER_OPT_GC, PRIMER_SALT_MONOVALENT=PRIMER_SALT_MONOVALENT, PRIMER_SALT_DIVALENT=PRIMER_SALT_DIVALENT, PRIMER_DNA_CONC=PRIMER_DNA_CONC, PRIMER_DNTP_CONC=PRIMER_DNTP_CONC)

    @mcp.tool(name="variation", description="Get parameters from get_service_info tool.")
    def submit_variation_app(token: Optional[str] = None, reference_genome_id: str = None, paired_end_libs: List[Dict] = None, single_end_libs: List[Dict] = None, srr_ids: List[str] = None, mapper: str = "BWA-mem", caller: str = "FreeBayes", output_path: str = None, output_file: str = None) -> str:
        # Get the appropriate token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return "Error: No authentication token available"
        
        user_id = extract_userid_from_token(auth_token)
        return start_variation_app(api, token=auth_token, user_id=user_id, reference_genome_id=reference_genome_id, paired_end_libs=paired_end_libs, single_end_libs=single_end_libs, srr_ids=srr_ids, mapper=mapper, caller=caller, output_path=output_path, output_file=output_file)

    @mcp.tool(name="tnseq", description="Get parameters from get_service_info tool.")
    def submit_tnseq_app(token: Optional[str] = None, experimental_conditions: List[str] = None, contrasts: List[str] = None, read_files: List[Dict] = None, reference_genome_id: str = None, recipe: str = "gumbel", protocol: str = "sassetti", primer: str = "", output_path: str = None, output_file: str = None) -> str:
        # Get the appropriate token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return "Error: No authentication token available"
        
        user_id = extract_userid_from_token(auth_token)
        return start_tnseq_app(api, token=auth_token, user_id=user_id, experimental_conditions=experimental_conditions, contrasts=contrasts, read_files=read_files, reference_genome_id=reference_genome_id, recipe=recipe, protocol=protocol, primer=primer, output_path=output_path, output_file=output_file)

# Phylogenomics Services

    @mcp.tool(name="bacterial_genome_tree", description="Get parameters from get_service_info tool.")
    def submit_bacterial_genome_tree_app(token: Optional[str] = None, output_path: str = None, output_file: str = None, genome_ids: List[str] = None, genome_groups: List[str] = None, optional_genome_ids: List[str] = None, genome_metadata_fields: str = None, number_of_genes: int = 20, bootstraps: int = 100, max_genomes_missing: int = 0, max_allowed_dups: int = 0) -> str:
        # Get the appropriate token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return "Error: No authentication token available"
        
        user_id = extract_userid_from_token(auth_token)
        return start_bacterial_genome_tree_app(api, token=auth_token, user_id=user_id, output_path=output_path, output_file=output_file, genome_ids=genome_ids, genome_groups=genome_groups, optional_genome_ids=optional_genome_ids, genome_metadata_fields=genome_metadata_fields, number_of_genes=number_of_genes, bootstraps=bootstraps, max_genomes_missing=max_genomes_missing, max_allowed_dups=max_allowed_dups)

    @mcp.tool(name="gene_tree", description="Get parameters from get_service_info tool.")
    def submit_gene_tree_app(token: Optional[str] = None, sequences: List[str] = None, alignment_program: str = None, trim_threshold: float = None, gap_threshold: float = None, alphabet: str = None, substitution_model: str = None, bootstrap: int = None, recipe: str = "RAxML", tree_type: str = None, feature_metadata_fields: str = None, genome_metadata_fields: str = None, output_path: str = None, output_file: str = None) -> str:
        # Get the appropriate token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return "Error: No authentication token available"
        
        user_id = extract_userid_from_token(auth_token)
        return start_gene_tree_app(api, token=auth_token, user_id=user_id, sequences=sequences, alignment_program=alignment_program, trim_threshold=trim_threshold, gap_threshold=gap_threshold, alphabet=alphabet, substitution_model=substitution_model, bootstrap=bootstrap, recipe=recipe, tree_type=tree_type, feature_metadata_fields=feature_metadata_fields, genome_metadata_fields=genome_metadata_fields, output_path=output_path, output_file=output_file)

    @mcp.tool(name="core_genome_mlst", description="Get parameters from get_service_info tool.")
    def submit_core_genome_mlst_app(token: Optional[str] = None, input_genome_type: str = "genome_group", analysis_type: str = "chewbbaca", input_genome_group: str = None, input_genome_fasta: str = None, schema_location: str = None, input_schema_selection: str = None, output_path: str = None, output_file: str = None) -> str:
        # Get the appropriate token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return "Error: No authentication token available"
        
        user_id = extract_userid_from_token(auth_token)
        return start_core_genome_mlst_app(api, token=auth_token, user_id=user_id, input_genome_type=input_genome_type, analysis_type=analysis_type, input_genome_group=input_genome_group, input_genome_fasta=input_genome_fasta, schema_location=schema_location, input_schema_selection=input_schema_selection, output_path=output_path, output_file=output_file)

    @mcp.tool(name="whole_genome_snp", description="Get parameters from get_service_info tool.")
    def submit_whole_genome_snp_app(token: Optional[str] = None, input_genome_type: str = None, majority_threshold: float = 0.5, min_mid_linkage: int = 10, max_mid_linkage: int = 40, analysis_type: str = "Whole Genome SNP Analysis", input_genome_group: str = None, input_genome_fasta: str = None, output_path: str = None, output_file: str = None) -> str:
        # Get the appropriate token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return "Error: No authentication token available"
        
        user_id = extract_userid_from_token(auth_token)
        return start_whole_genome_snp_app(api, token=auth_token, user_id=user_id, input_genome_type=input_genome_type, majority_threshold=majority_threshold, min_mid_linkage=min_mid_linkage, max_mid_linkage=max_mid_linkage, analysis_type=analysis_type, input_genome_group=input_genome_group, input_genome_fasta=input_genome_fasta, output_path=output_path, output_file=output_file)

# Metagenomics Services

    @mcp.tool(name="taxonomic_classification", description="Get parameters from get_service_info tool.")
    def submit_taxonomic_classification_app(token: Optional[str] = None, host_genome: str = "no_host", analysis_type: str = "16S", paired_end_libs: List[Dict] = None, single_end_libs: List[Dict] = None, srr_libs: List[Dict] = None, database: str = "SILVA", save_classified_sequences: bool = False, save_unclassified_sequences: bool = False, confidence_interval: float = 0.1, output_path: str = None, output_file: str = None) -> str:
        # Get the appropriate token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return "Error: No authentication token available"
        
        user_id = extract_userid_from_token(auth_token)
        return start_taxonomic_classification_app(api, token=auth_token, user_id=user_id, host_genome=host_genome, analysis_type=analysis_type, paired_end_libs=paired_end_libs, single_end_libs=single_end_libs, srr_libs=srr_libs, database=database, save_classified_sequences=save_classified_sequences, save_unclassified_sequences=save_unclassified_sequences, confidence_interval=confidence_interval, output_path=output_path, output_file=output_file)

    @mcp.tool(name="metagenomic_binning", description="Get parameters from get_service_info tool.")
    def submit_metagenomic_binning_app(token: Optional[str] = None, paired_end_libs: Dict = None, single_end_libs: Dict = None, srr_ids: str = None, contigs: str = None, genome_group: str = None, skip_indexing: bool = False, recipe: str = None, viral_recipe: str = None, output_path: str = None, output_file: str = None, force_local_assembly: bool = False, force_inline_annotation: bool = True, perform_bacterial_binning: bool = True, perform_viral_binning: bool = False, perform_viral_annotation: bool = False, perform_bacterial_annotation: bool = True, assembler: str = "", danglen: str = "50", min_contig_len: int = 400, min_contig_cov: float = 4.0) -> str:
        # Get the appropriate token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return "Error: No authentication token available"
        
        user_id = extract_userid_from_token(auth_token)
        return start_metagenomic_binning_app(api, token=auth_token, user_id=user_id, paired_end_libs=paired_end_libs, single_end_libs=single_end_libs, srr_ids=srr_ids, contigs=contigs, genome_group=genome_group, skip_indexing=skip_indexing, recipe=recipe, viral_recipe=viral_recipe, output_path=output_path, output_file=output_file, force_local_assembly=force_local_assembly, force_inline_annotation=force_inline_annotation, perform_bacterial_binning=perform_bacterial_binning, perform_viral_binning=perform_viral_binning, perform_viral_annotation=perform_viral_annotation, perform_bacterial_annotation=perform_bacterial_annotation, assembler=assembler, danglen=danglen, min_contig_len=min_contig_len, min_contig_cov=min_contig_cov)

    @mcp.tool(name="metagenomic_read_mapping", description="Get parameters from get_service_info tool.")
    def submit_metagenomic_read_mapping_app(token: Optional[str] = None, gene_set_type: str = None, gene_set_name: str = None, gene_set_fasta: str = None, gene_set_feature_group: str = None, paired_end_libs: Dict = None, single_end_libs: Dict = None, srr_ids: str = None, output_path: str = None, output_file: str = None) -> str:
        # Get the appropriate token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return "Error: No authentication token available"
        
        user_id = extract_userid_from_token(auth_token)
        return start_metagenomic_read_mapping_app(api, token=auth_token, user_id=user_id, gene_set_type=gene_set_type, gene_set_name=gene_set_name, gene_set_fasta=gene_set_fasta, gene_set_feature_group=gene_set_feature_group, paired_end_libs=paired_end_libs, single_end_libs=single_end_libs, srr_ids=srr_ids, output_path=output_path, output_file=output_file)

# Transcriptomics Services

    @mcp.tool(name="rnaseq", description="Get parameters from get_service_info tool.")
    def submit_rnaseq_app(token: Optional[str] = None, experimental_conditions: List[str] = None, contrasts: str = None, strand_specific: bool = True, paired_end_libs: List[Dict] = None, single_end_libs: List[Dict] = None, srr_libs: List[Dict] = None, reference_genome_id: str = None, genome_type: str = None, recipe: str = "HTSeq-DESeq", host_ftp: str = None, output_path: str = None, output_file: str = None, trimming: bool = False, unit_test: str = None, skip_sampling: str = None) -> str:
        # Get the appropriate token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return "Error: No authentication token available"
        
        user_id = extract_userid_from_token(auth_token)
        return start_rnaseq_app(api, token=auth_token, user_id=user_id, experimental_conditions=experimental_conditions, contrasts=contrasts, strand_specific=strand_specific, paired_end_libs=paired_end_libs, single_end_libs=single_end_libs, srr_libs=srr_libs, reference_genome_id=reference_genome_id, genome_type=genome_type, recipe=recipe, host_ftp=host_ftp, output_path=output_path, output_file=output_file, trimming=trimming, unit_test=unit_test, skip_sampling=skip_sampling)

    @mcp.tool(name="expression_import", description="Get parameters from get_service_info tool.")
    def submit_expression_import_app(token: Optional[str] = None, xfile: str = None, mfile: str = None, ustring: str = None, output_path: str = None, output_file: str = None) -> str:
        # Get the appropriate token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return "Error: No authentication token available"
        
        user_id = extract_userid_from_token(auth_token)
        return start_expression_import_app(api, token=auth_token, user_id=user_id, xfile=xfile, mfile=mfile, ustring=ustring, output_path=output_path, output_file=output_file)

# Viral Services

    @mcp.tool(name="sars_wastewater_analysis", description="Get parameters from get_service_info tool.")
    def submit_sars_wastewater_analysis_app(token: Optional[str] = None, paired_end_libs: List[Dict] = None, single_end_libs: List[Dict] = None, srr_libs: List[Dict] = None, recipe: str = "auto", primers: str = "ARTIC", minimum_base_quality_score: int = 20, minimum_genome_coverage: int = 60, agg_minimum_lineage_abundance: float = 0.01, minimum_coverage_depth: int = 0, confirmedonly: bool = False, minimum_lineage_abundance: float = 0.001, coverage_estimate: int = 10, timeseries_plot_interval: str = "0", primer_version: str = None, barcode_csv: str = None, sample_metadata_csv: str = None, keep_intermediates: bool = True, output_path: str = None, output_file: str = None) -> str:
        # Get the appropriate token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return "Error: No authentication token available"
        
        user_id = extract_userid_from_token(auth_token)
        return start_sars_wastewater_analysis_app(api, token=auth_token, user_id=user_id, paired_end_libs=paired_end_libs, single_end_libs=single_end_libs, srr_libs=srr_libs, recipe=recipe, primers=primers, minimum_base_quality_score=minimum_base_quality_score, minimum_genome_coverage=minimum_genome_coverage, agg_minimum_lineage_abundance=agg_minimum_lineage_abundance, minimum_coverage_depth=minimum_coverage_depth, confirmedonly=confirmedonly, minimum_lineage_abundance=minimum_lineage_abundance, coverage_estimate=coverage_estimate, timeseries_plot_interval=timeseries_plot_interval, primer_version=primer_version, barcode_csv=barcode_csv, sample_metadata_csv=sample_metadata_csv, keep_intermediates=keep_intermediates, output_path=output_path, output_file=output_file)

    @mcp.tool(name="sequence_submission", description="Get parameters from get_service_info tool.")
    def submit_sequence_submission_app(token: Optional[str] = None, input_source: str = None, input_fasta_data: str = None, input_fasta_file: str = None, input_genome_group: str = None, metadata: str = None, affiliation: str = None, first_name: str = None, last_name: str = None, email: str = None, consortium: str = None, country: str = None, phoneNumber: str = None, street: str = None, postal_code: str = None, city: str = None, state: str = None, numberOfSequences: str = None, output_path: str = None, output_file: str = None) -> str:
        # Get the appropriate token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return "Error: No authentication token available"
        
        user_id = extract_userid_from_token(auth_token)
        return start_sequence_submission_app(api, token=auth_token, user_id=user_id, input_source=input_source, input_fasta_data=input_fasta_data, input_fasta_file=input_fasta_file, input_genome_group=input_genome_group, metadata=metadata, affiliation=affiliation, first_name=first_name, last_name=last_name, email=email, consortium=consortium, country=country, phoneNumber=phoneNumber, street=street, postal_code=postal_code, city=city, state=state, numberOfSequences=numberOfSequences, output_path=output_path, output_file=output_file)

    @mcp.tool(name="influenza_ha_subtype_conversion", description="Get parameters from get_service_info tool.")
    def submit_influenza_ha_subtype_conversion_app(token: Optional[str] = None, input_source: str = None, input_fasta_data: str = None, input_fasta_file: str = None, input_feature_group: str = None, input_feature_list: str = None, types: str = None, output_path: str = None, output_file: str = None) -> str:
        # Get the appropriate token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return "Error: No authentication token available"
        
        user_id = extract_userid_from_token(auth_token)
        return start_influenza_ha_subtype_conversion_app(api, token=auth_token, user_id=user_id, input_source=input_source, input_fasta_data=input_fasta_data, input_fasta_file=input_fasta_file, input_feature_group=input_feature_group, input_feature_list=input_feature_list, types=types, output_path=output_path, output_file=output_file)

    @mcp.tool(name="subspecies_classification", description="Get parameters from get_service_info tool.")
    def submit_subspecies_classification_app(token: Optional[str] = None, input_source: str = None, input_fasta_data: str = None, input_fasta_file: str = None, input_genome_group: str = None, ref_msa_fasta: str = None, virus_type: str = None, output_path: str = None, output_file: str = None) -> str:
        # Get the appropriate token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return "Error: No authentication token available"
        
        user_id = extract_userid_from_token(auth_token)
        return start_subspecies_classification_app(api, token=auth_token, user_id=user_id, input_source=input_source, input_fasta_data=input_fasta_data, input_fasta_file=input_fasta_file, input_genome_group=input_genome_group, ref_msa_fasta=ref_msa_fasta, virus_type=virus_type, output_path=output_path, output_file=output_file)

    @mcp.tool(name="viral_assembly", description="Get parameters from get_service_info tool.")
    def submit_viral_assembly_app(token: Optional[str] = None, paired_end_lib: Dict = None, single_end_lib: Dict = None, srr_id: str = None, recipe: str = "auto", module: str = None, viral_size: str = "5M", output_path: str = None, output_file: str = None) -> str:
        # Get the appropriate token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return "Error: No authentication token available"
        
        user_id = extract_userid_from_token(auth_token)
        return start_viral_assembly_app(api, token=auth_token, user_id=user_id, paired_end_lib=paired_end_lib, single_end_lib=single_end_lib, srr_id=srr_id, recipe=recipe, module=module, viral_size=viral_size, output_path=output_path, output_file=output_file)

    # Additional Services

    @mcp.tool(name="genome_alignment", description="Get parameters from get_service_info tool.")
    def submit_genome_alignment_app(token: Optional[str] = None, genome_ids: List[str] = None, recipe: str = "progressiveMauve", seedWeight: float = None, maxGappedAlignerLength: float = None, maxBreakpointDistanceScale: float = None, conservationDistanceScale: float = None, weight: float = None, minScaledPenalty: float = None, hmmPGoHomologous: float = None, hmmPGoUnrelated: float = None, output_path: str = None, output_file: str = None) -> str:
        # Get the appropriate token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return "Error: No authentication token available"
        
        user_id = extract_userid_from_token(auth_token)
        return start_genome_alignment_app(api, token=auth_token, user_id=user_id, genome_ids=genome_ids, recipe=recipe, seedWeight=seedWeight, maxGappedAlignerLength=maxGappedAlignerLength, maxBreakpointDistanceScale=maxBreakpointDistanceScale, conservationDistanceScale=conservationDistanceScale, weight=weight, minScaledPenalty=minScaledPenalty, hmmPGoHomologous=hmmPGoHomologous, hmmPGoUnrelated=hmmPGoUnrelated, output_path=output_path, output_file=output_file)

    @mcp.tool(name="sars_genome_analysis", description="Get parameters from get_service_info tool.")
    def submit_sars_genome_analysis_app(token: Optional[str] = None, paired_end_libs: List[Dict] = None, single_end_libs: List[Dict] = None, srr_ids: List[str] = None, recipe: str = "auto", primers: str = "ARTIC", primer_version: str = None, min_depth: int = 100, max_depth: int = 8000, keep_intermediates: int = 0, output_path: str = None, output_file: str = None) -> str:
        # Get the appropriate token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return "Error: No authentication token available"
        
        user_id = extract_userid_from_token(auth_token)
        return start_sars_genome_analysis_app(api, token=auth_token, user_id=user_id, paired_end_libs=paired_end_libs, single_end_libs=single_end_libs, srr_ids=srr_ids, recipe=recipe, primers=primers, primer_version=primer_version, min_depth=min_depth, max_depth=max_depth, keep_intermediates=keep_intermediates, output_path=output_path, output_file=output_file)

    @mcp.tool(name="msa_snp_analysis", description="Get parameters from get_service_info tool.")
    def submit_msa_snp_analysis_app(token: Optional[str] = None, input_status: str = "unaligned", input_type: str = "input_group", fasta_files: List[Dict] = None, select_genomegroup: List[str] = None, feature_groups: List[str] = None, feature_list: List[str] = None, genome_list: List[str] = None, aligner: str = "Muscle", alphabet: str = "dna", fasta_keyboard_input: str = "", ref_type: str = "none", strategy: str = "auto", ref_string: str = "", output_path: str = None, output_file: str = None) -> str:
        # Get the appropriate token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return "Error: No authentication token available"
        
        user_id = extract_userid_from_token(auth_token)
        return start_msa_snp_analysis_app(api, token=auth_token, user_id=user_id, input_status=input_status, input_type=input_type, fasta_files=fasta_files, select_genomegroup=select_genomegroup, feature_groups=feature_groups, feature_list=feature_list, genome_list=genome_list, aligner=aligner, alphabet=alphabet, fasta_keyboard_input=fasta_keyboard_input, ref_type=ref_type, strategy=strategy, ref_string=ref_string, output_path=output_path, output_file=output_file)

    @mcp.tool(name="metacats", description="Get parameters from get_service_info tool.")
    def submit_metacats_app(token: Optional[str] = None, output_path: str = None, output_file: str = None, p_value: float = 0.05, year_ranges: str = None, metadata_group: str = None, input_type: str = None, alphabet: str = "na", groups: List[str] = None, alignment_file: str = None, group_file: str = None, alignment_type: str = None, auto_groups: List[Dict] = None) -> str:
        # Get the appropriate token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return "Error: No authentication token available"
        
        user_id = extract_userid_from_token(auth_token)
        return start_metacats_app(api, token=auth_token, user_id=user_id, output_path=output_path, output_file=output_file, p_value=p_value, year_ranges=year_ranges, metadata_group=metadata_group, input_type=input_type, alphabet=alphabet, groups=groups, alignment_file=alignment_file, group_file=group_file, alignment_type=alignment_type, auto_groups=auto_groups)

    @mcp.tool(name="proteome_comparison", description="Get parameters from get_service_info tool.")
    def submit_proteome_comparison_app(token: Optional[str] = None, genome_ids: List[str] = None, user_genomes: List[str] = None, user_feature_groups: List[str] = None, reference_genome_index: int = 1, min_seq_cov: float = 0.30, max_e_val: float = 1e-5, min_ident: float = 0.1, min_positives: float = 0.2, output_path: str = None, output_file: str = None) -> str:
        # Get the appropriate token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return "Error: No authentication token available"
        
        user_id = extract_userid_from_token(auth_token)
        return start_proteome_comparison_app(api, token=auth_token, user_id=user_id, genome_ids=genome_ids, user_genomes=user_genomes, user_feature_groups=user_feature_groups, reference_genome_index=reference_genome_index, min_seq_cov=min_seq_cov, max_e_val=max_e_val, min_ident=min_ident, min_positives=min_positives, output_path=output_path, output_file=output_file)

    @mcp.tool(name="comparative_systems", description="Get parameters from get_service_info tool.")
    def submit_comparative_systems_app(token: Optional[str] = None, output_path: str = None, output_file: str = None, genome_ids: List[str] = None, genome_groups: List[str] = None) -> str:
        # Get the appropriate token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return "Error: No authentication token available"
        
        user_id = extract_userid_from_token(auth_token)
        return start_comparative_systems_app(api, token=auth_token, user_id=user_id, output_path=output_path, output_file=output_file, genome_ids=genome_ids, genome_groups=genome_groups)

    @mcp.tool(name="docking", description="Get parameters from get_service_info tool.")
    def submit_docking_app(token: Optional[str] = None, protein_input_type: str = None, input_pdb: List[str] = None, user_pdb_file: List[str] = None, ligand_library_type: str = None, ligand_named_library: str = None, ligand_smiles_list: List[str] = None, ligand_ws_file: str = None, top_n: int = None, batch_size: int = 10, output_path: str = None, output_file: str = None) -> str:
        # Get the appropriate token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return "Error: No authentication token available"
        
        user_id = extract_userid_from_token(auth_token)
        return start_docking_app(api, token=auth_token, user_id=user_id, protein_input_type=protein_input_type, input_pdb=input_pdb, user_pdb_file=user_pdb_file, ligand_library_type=ligand_library_type, ligand_named_library=ligand_named_library, ligand_smiles_list=ligand_smiles_list, ligand_ws_file=ligand_ws_file, top_n=top_n, batch_size=batch_size, output_path=output_path, output_file=output_file)

    @mcp.tool(name="similar_genome_finder", description="Get parameters from get_service_info tool.")
    def submit_similar_genome_finder_app(token: Optional[str] = None, selectedGenomeId: str = None, fasta_file: str = None, max_pvalue: float = None, max_distance: float = None, max_hits: int = None, include_reference: bool = None, include_representative: bool = None, include_bacterial: bool = None, include_viral: bool = None, output_path: str = None, output_file: str = None) -> str:
        # Get the appropriate token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return "Error: No authentication token available"
        
        user_id = extract_userid_from_token(auth_token)
        return start_similar_genome_finder_app(similar_genome_finder_api, token=auth_token, user_id=user_id, selectedGenomeId=selectedGenomeId, fasta_file=fasta_file, max_pvalue=max_pvalue, max_distance=max_distance, max_hits=max_hits, include_reference=include_reference, include_representative=include_representative, include_bacterial=include_bacterial, include_viral=include_viral, output_path=output_path, output_file=output_file)

    @mcp.tool(name="fastqutils", description="Get parameters from get_service_info tool.")
    def submit_fastqutils_app(token: Optional[str] = None, reference_genome_id: str = None, paired_end_libs: List[Dict] = None, single_end_libs: List[Dict] = None, srr_libs: List[Dict] = None, output_path: str = None, output_file: str = None, recipe: List[str] = None) -> str:
        # Get the appropriate token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return "Error: No authentication token available"
        
        user_id = extract_userid_from_token(auth_token)
        return start_fastq_utils_app(api, token=auth_token, user_id=user_id, reference_genome_id=reference_genome_id, paired_end_libs=paired_end_libs, single_end_libs=single_end_libs, srr_libs=srr_libs, output_path=output_path, output_file=output_file, recipe=recipe)

