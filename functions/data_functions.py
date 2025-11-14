"""
BV-BRC Data Functions

This module provides data query functions for the BV-BRC Solr API.
Combines mvp_functions and common_functions from the data-mcp-server.
"""

import os
import json
from typing import Any, Dict, List, Tuple
from bvbrc_solr_api import create_client, query
from bvbrc_solr_api.core.solr_http_client import select as solr_select
CURSOR_BATCH_SIZE = 100
# Timeout for Solr queries in seconds (default is 60, increase for large queries)
SOLR_QUERY_TIMEOUT = 300.0  # 5 minutes



def create_bvbrc_client(base_url: str = None, headers: Dict[str, str] = None) -> Any:
    """
    Create a BV-BRC client with optional configuration overrides.
    
    Args:
        base_url: Optional base URL override
        headers: Optional headers override
        
    Returns:
        BV-BRC client instance
    """
    context_overrides = {}
    if base_url:
        context_overrides["base_url"] = base_url
    if headers:
        context_overrides["headers"] = headers
    
    return create_client(context_overrides)


def query_direct(core: str, filter_str: str = "", options: Dict[str, Any] = None,
                base_url: str = None, headers: Dict[str, str] = None,
                cursorId: str | None = None, countOnly: bool = False) -> Dict[str, Any]:
    """
    Query BV-BRC data directly using core name and filter string with cursor-based streaming.
    
    Args:
        core: The core/collection name (e.g., "genome", "genome_feature")
        filter_str: Solr query expression (e.g., "genome_id:123.45" or "species:\"Escherichia coli\"")
        options: Optional query options (e.g., {"select": ["field1", "field2"], "sort": "field_name"})
        base_url: Optional base URL override
        headers: Optional headers override
        cursorId: Cursor ID for pagination (optional, use "*" or None for first page)
        countOnly: If True, iterate through all pages to compute total count without returning data
        
    Returns:
        Dict with keys depending on countOnly:
        - If countOnly is True: { "count": <total_count> }
        - Else: { "results": [ ...batch... ], "count": <batch_count>, "nextCursorId": <str|None> }
        
    Note:
        Batch size is hardcoded to 1000 entries per page. Use nextCursorId from the response
        to fetch the next batch by passing it as cursorId in a subsequent call.
    """
    client = create_bvbrc_client(base_url, headers)
    options = options or {}
    
    # Build context_overrides with timeout and any provided base_url/headers
    context_overrides = {"timeout": SOLR_QUERY_TIMEOUT}
    if base_url:
        context_overrides["base_url"] = base_url
    if headers:
        context_overrides["headers"] = headers
    
    # Prepare a configured CursorPager via the client (ensures correct unique_key/sort per collection)
    pager = getattr(client, core).stream_all_solr(
        rows=CURSOR_BATCH_SIZE,
        sort=options.get("sort"),
        fields=options.get("select"),
        q_expr=filter_str if filter_str else "*:*",
        start_cursor=cursorId or "*",
        context_overrides=context_overrides
    )

    # Helper to execute a single Solr select call based on the pager's current state
    def _single_page(cursor_mark: str) -> Tuple[List[Dict[str, Any]], str | None]:
        params = dict(pager.base_params)
        params["rows"] = pager.rows
        params["sort"] = pager.sort
        params["cursorMark"] = cursor_mark
        result = solr_select(
            pager.collection,
            params,
            base_url=pager.base_url,
            headers=pager.headers,
            auth=pager.auth,
            timeout=pager.timeout,
        )
        response = result.get("response", {})
        docs: List[Dict[str, Any]] = response.get("docs", [])
        next_cursor = result.get("nextCursorMark")
        return docs, next_cursor

    if countOnly:
        # Iterate through all pages to compute total count without returning data
        total_count = 0
        last_mark: str | None = None
        current_cursor = pager.cursor
        while True:
            docs, next_cursor = _single_page(current_cursor)
            if not docs:
                break
            total_count += len(docs)
            if next_cursor is None or next_cursor == current_cursor or next_cursor == last_mark:
                break
            last_mark = current_cursor
            current_cursor = next_cursor
        return {"count": total_count}

    # Fetch a single batch/page and return nextCursorId for optional continuation
    docs, next_cursor = _single_page(pager.cursor)
    return {
        "results": docs,
        "count": len(docs),
        "nextCursorId": next_cursor,
    }


def format_query_result(result: List[Dict[str, Any]], max_items: int = 10) -> str:
    """
    Format query result for display.
    
    Args:
        result: List of query results
        max_items: Maximum number of items to display
        
    Returns:
        Formatted string representation of results
    """
    if not result:
        return "No results found."
    
    total_count = len(result)
    display_count = min(total_count, max_items)
    
    formatted = f"Found {total_count} result(s). Showing first {display_count}:\n\n"
    
    for i, item in enumerate(result[:display_count]):
        formatted += f"Result {i+1}:\n"
        for key, value in item.items():
            if isinstance(value, (list, dict)):
                value = json.dumps(value, indent=2)
            formatted += f"  {key}: {value}\n"
        formatted += "\n"
    
    if total_count > max_items:
        formatted += f"... and {total_count - max_items} more results.\n"
    
    return formatted


def lookup_parameters(collection: str) -> str:
    """
    Lookup parameters for a given collection by loading from prompts folder.
    
    Args:
        collection: The collection name (without _functions.py suffix)
        
    Returns:
        String describing the parameters for the collection
    """
    # Get the directory of this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_dir = os.path.join(current_dir, '..', 'prompts')
    
    # Construct the file path for the collection
    prompt_file = os.path.join(prompts_dir, f"{collection}.txt")
    
    try:
        # Read the parameters from the file
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        # If file doesn't exist, list available endpoints
        try:
            available_files = [f.replace('.txt', '') for f in os.listdir(prompts_dir) if f.endswith('.txt')]
            return f"Unknown collection: {collection}. Available collections: {', '.join(sorted(available_files))}"
        except FileNotFoundError:
            return f"Unknown collection: {collection}. Prompts directory not found."
    except Exception as e:
        return f"Error reading parameters for collection '{collection}': {str(e)}"


def query_info() -> str:
    """
    Get general query information for all collections.
    
    Returns:
        String with general query instructions and formatting guidelines
    """
    return """BV-BRC Query Tool Instructions

            BASIC QUERY PARAMETERS:
            - filter_str: Solr query expression for filtering results
            Examples: 
                - field_name:value - exact match
                - field_name:"value" - exact match with quotes for strings
                - field_name:[value1 TO value2] - range query
                - field_name:*value* - wildcard search
                - field_name:value1 OR field_name:value2 - OR logic
                - field_name:value1 AND field_name:value2 - AND logic
                - -field_name:value - NOT logic (exclude)

            - select: Comma-separated list of fields to return
            Example: "genome_id,genome_name,species,strain"
            Note: Use field names exactly as they appear in the schema

            - sort: Field name to sort by (optional)
            Example: "genome_name" or "date_inserted"
            For descending order, use negative: "-genome_name"

            QUERY EXAMPLES:
            1. Find genomes by species: species:"Escherichia coli"
            2. Find genomes with specific strain: strain:*K-12*
            3. Get recent entries: date_inserted:[2023-01-01 TO *]
            4. Multiple conditions: species:"Escherichia coli" AND strain:*K-12*
            5. Select specific fields: select="genome_id,genome_name,species,strain"
            6. Sort results: sort="genome_name"
            7. Exclude specific values: -species:"test"

            FIELD TYPES:
            - string: Text values (use quotes for exact matches)
            - integer: Numeric values (no quotes)
            - date: Date values in YYYY-MM-DD format
            - boolean: true/false (no quotes)
            - wildcard: Use * for partial matches

            TIPS:
            - Use * for wildcard searches
            - Use quotes for exact string matches
            - Use [value1 TO value2] for range queries
            - Combine conditions with AND/OR
            - Check available fields in collection-specific parameters
            - Use select to return only needed fields for better performance"""


def list_solr_collections() -> str:
    """
    List all available Solr collections.
    
    Returns:
        String with the available collections and their descriptions
    """
    return """Available Solr Collections:
        1. **genome** - Complete bacterial and viral genome assemblies with metadata including taxonomy, quality metrics, geographic location, and antimicrobial resistance data.
        2. **genome_feature** - Individual genes, proteins, and functional elements within genomes, including annotations, functional classifications, and sequence information. Does not include special properties like virulence or resistance factors.
        3. **genome_sequence** - Raw DNA/RNA sequence data for genomes and individual sequences with accession numbers and sequence metadata.
        4. **antibiotics** - Comprehensive database of antimicrobial compounds with chemical properties, mechanisms of action, and pharmacological classifications.
        5. **bioset_result** - Experimental results from gene expression, proteomics, and other high-throughput studies with statistical measures and experimental conditions.
        6. **bioset** - Experimental datasets and study designs including treatment conditions, sample information, and analysis protocols.
        7. **strain** - Viral strain information with genetic segments, host data, and epidemiological metadata.
        8. **surveillance** - Clinical surveillance data including patient demographics, disease status, and treatment outcomes.
        9. **experiment** - Experimental metadata including study design, protocols, and experimental conditions.
        10. **taxonomy** - Taxonomic classification data for organisms with hierarchical relationships and nomenclature.
        11. **pathway** - Biological pathway information including metabolic and signaling pathways.
        12. **protein_structure** - Protein structural data including 3D coordinates and structural classifications.
        13. **epitope** - Antigenic epitope data for vaccine and immunology research.
        14. **serology** - Serological test results and antibody response data.
        15. **genome_amr** - Antimicrobial resistance data linked to specific genomes and resistance mechanisms.
        16. **sequence_feature** - Sequence variants and mutations with functional annotations.
        17. **protein_feature** - Protein domain and functional feature annotations.
        18. **subsystem** - Features that participate in subsystems, along with the functional roles they perform.
        19. **ppi** - Protein-protein interaction data.
        20. **spike_variant** - SARS-CoV-2 spike protein variant information.
        21. **spike_lineage** - SARS-CoV-2 lineage and variant classifications.
        22. **structured_assertion** - Curated functional assertions and annotations.
        23. **misc_niaid_sgc** - Miscellaneous NIAID Single Cell Genomics data.
        24. **enzyme_class_ref** - Enzyme classification reference data.
        25. **epitope_assay** - Epitope binding assay results.
        26. **gene_ontology_ref** - Gene Ontology reference classifications.
        27. **id_ref** - Identifier reference mappings.
        28. **pathway_ref** - Pathway reference data.
        29. **protein_family_ref** - Protein family reference classifications.
        30. **sp_gene_ref** - Special gene property reference.
        31. **sp_gene** - Genes with special properties (Antibiotic Resistance, Virulence Factor, Transporter)
        32. **subsystem_ref** - Reference of names and classifications of subsystems used to group commonly-associated functional roles.
        33. **sequence_feature_vt** - Sequence feature variant type data."""

