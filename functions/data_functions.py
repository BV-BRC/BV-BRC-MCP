"""
BV-BRC Data Functions

This module provides data query functions for the BV-BRC Solr API.
Combines mvp_functions and common_functions from the data-mcp-server.
"""

import os
import re
import json
from typing import Any, Dict, List, Tuple, Optional, Set
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
                cursorId: str | None = None, countOnly: bool = False,
                batch_size: Optional[int] = None) -> Dict[str, Any]:
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
        batch_size: Number of rows to return per page (optional, defaults to CURSOR_BATCH_SIZE=100)
        
    Returns:
        Dict with keys depending on countOnly:
        - If countOnly is True: { "count": <total_count> }
        - Else: { "results": [ ...batch... ], "count": <batch_count>, "nextCursorId": <str|None> }
        
    Note:
        Batch size defaults to 100 entries per page. Use nextCursorId from the response
        to fetch the next batch by passing it as cursorId in a subsequent call.
    """
    client = create_bvbrc_client(base_url, headers)
    options = options or {}
    
    # Use provided batch_size or fall back to default constant
    rows_per_page = batch_size if batch_size is not None else CURSOR_BATCH_SIZE
    
    # Build context_overrides with timeout and any provided base_url/headers
    context_overrides = {"timeout": SOLR_QUERY_TIMEOUT}
    if base_url:
        context_overrides["base_url"] = base_url
    if headers:
        context_overrides["headers"] = headers
    
    # Prepare a configured CursorPager via the client (ensures correct unique_key/sort per collection)
    pager = getattr(client, core).stream_all_solr(
        rows=rows_per_page,
        sort=options.get("sort"),
        fields=options.get("select"),
        q_expr=filter_str if filter_str else "*:*",
        start_cursor=cursorId or "*",
        context_overrides=context_overrides
    )

    # Helper to execute a single Solr select call based on the pager's current state
    def _single_page(cursor_mark: str) -> Tuple[List[Dict[str, Any]], str | None, Optional[int]]:
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
        num_found = response.get("numFound")
        return docs, next_cursor, num_found

    if countOnly:
        # Return Solr-reported total without iterating pages
        docs, _next_cursor, num_found = _single_page(pager.cursor)
        return {"numFound": num_found if num_found is not None else len(docs)}

    # Fetch a single batch/page and return nextCursorId for optional continuation
    docs, next_cursor, num_found = _single_page(pager.cursor)
    return {
        "results": docs,
        "count": len(docs),
        "numFound": num_found if num_found is not None else len(docs),
        "nextCursorId": next_cursor,
    }


# Helper utilities shared by MCP tools
def normalize_select(sel: Any) -> Optional[List[str]]:
    if sel is None:
        return None
    if isinstance(sel, str):
        return [part.strip() for part in sel.split(",") if part.strip()]
    if isinstance(sel, list):
        return [str(item).strip() for item in sel if str(item).strip()]
    raise ValueError("select must be a comma-separated string or list of fields")


def normalize_sort(srt: Any) -> Optional[str]:
    if srt is None:
        return None
    if isinstance(srt, str):
        return srt
    if isinstance(srt, list):
        parts: List[str] = []
        for item in srt:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "field" in item:
                direction = item.get("dir", "asc").lower()
                dir_token = "desc" if direction == "desc" else "asc"
                parts.append(f"{item['field']} {dir_token}")
        return ", ".join(parts) if parts else None
    raise ValueError("sort must be a string or list of sort entries")


def quote_value(val: Any, allow_wildcards: bool = False) -> str:
    if val is None:
        return '""'
    if isinstance(val, bool):
        return str(val).lower()
    if isinstance(val, (int, float)):
        return str(val)
    s = str(val)
    s = s.replace('"', '\\"')
    if allow_wildcards:
        # Preserve * characters; escape spaces to avoid breaking the pattern.
        s = s.replace(" ", r"\ ")
        return s
    s = s.replace("*", "\\*")
    if re.search(r"\s|[:()]", s):
        return f'"{s}"'
    return s


def build_leaf(field: str, op: str, value: Any) -> str:
    op = op.lower()
    if op == "eq":
        return f"{field}:{quote_value(value)}"
    if op == "neq":
        return f"-{field}:{quote_value(value)}"
    if op == "lt":
        return f"{field}:{{* TO {quote_value(value)}}}"
    if op == "lte":
        return f"{field}:[* TO {quote_value(value)}]"
    if op == "gt":
        return f"{field}:{{{quote_value(value)} TO *}}"
    if op == "gte":
        return f"{field}:[{quote_value(value)} TO *]"
    if op == "between":
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("between op requires [min, max]")
        return f"{field}:[{quote_value(value[0])} TO {quote_value(value[1])}]"
    if op == "in":
        if not isinstance(value, (list, tuple)):
            raise ValueError("in op requires a list of values")
        terms = [f"{field}:{quote_value(v)}" for v in value]
        return "(" + " OR ".join(terms) + ")" if terms else "*:*"
    if op == "contains":
        return f"{field}:*{quote_value(value, allow_wildcards=True)}*"
    if op == "startswith":
        return f"{field}:{quote_value(value, allow_wildcards=True)}*"
    if op == "endswith":
        return f"{field}:*{quote_value(value, allow_wildcards=True)}"
    if op == "exists":
        return f"{field}:*"
    if op == "missing":
        return f"-{field}:*"
    if op in ("wildcard", "matches"):
        return f"{field}:{quote_value(value, allow_wildcards=True)}"
    raise ValueError(f"Unsupported operator: {op}")


def build_filter(expr: Any) -> str:
    if not expr:
        return "*:*"
    if isinstance(expr, dict) and "filters" in expr:
        logic = expr.get("logic", "and").lower()
        joiner = " AND " if logic == "and" else " OR "
        built = [build_filter(item) for item in expr.get("filters", []) if item is not None]
        if not built:
            return "*:*"
        if len(built) == 1:
            return built[0]
        return "(" + joiner.join(built) + ")"
    if isinstance(expr, dict):
        field = expr.get("field")
        op = expr.get("op", "eq")
        value = expr.get("value")
        if not field:
            raise ValueError("filter leaf requires a 'field'")
        return build_leaf(field, op, value)
    raise ValueError("filters must be a dict with filters or leaf conditions")


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


def get_collection_fields(collection: str) -> List[str]:
    """
    Return the list of allowed field names for a collection based on its prompt
    definition. Falls back to an empty list if the prompt file is missing.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_dir = os.path.join(current_dir, "..", "prompts")
    prompt_file = os.path.join(prompts_dir, f"{collection}.txt")

    fields: Set[str] = set()
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.lower().startswith("primary key:"):
                    primary = stripped.split(":", 1)[1].strip()
                    if primary:
                        fields.add(primary)
                    continue
                match = re.match(r"([A-Za-z0-9_]+)", stripped)
                if match:
                    fields.add(match.group(1))
    except FileNotFoundError:
        return []
    except Exception:
        # Do not block queries if prompt parsing fails; skip validation instead.
        return []

    return sorted(fields)


def validate_filter_fields(expr: Any, allowed_fields: Set[str]) -> List[str]:
    """
    Walk a structured filter expression and return any field names that are not
    present in the allowed_fields list. If allowed_fields is empty, returns an
    empty list (no validation performed).
    """
    if not allowed_fields:
        return []

    invalid: Set[str] = set()

    def _walk(node: Any) -> None:
        if not node:
            return
        if isinstance(node, dict) and "filters" in node:
            for child in node.get("filters", []):
                _walk(child)
            return
        if isinstance(node, dict):
            field = node.get("field")
            if field and field not in allowed_fields:
                invalid.add(str(field))

    _walk(expr)
    return sorted(invalid)


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

            STRUCTURED FILTERS (no Solr syntax needed):
            filters = {
              "logic": "and" | "or",           # optional, defaults to "and"
              "filters": [
                { "field": "genome_name", "op": "eq", "value": "Escherichia coli" },
                { "field": "antibiotic", "op": "eq", "value": "ampicillin" },
                { "logic": "or", "filters": [
                    { "field": "resistant_phenotype", "op": "eq", "value": "Resistant" },
                    { "field": "resistant_phenotype", "op": "eq", "value": "Intermediate" }
                  ]
                }
              ]
            }

            SUPPORTED OPERATORS:
            - eq / neq: exact match / not equal
            - lt / lte / gt / gte: range bounds
            - between: inclusive range [min, max] (value is [min, max])
            - in: list membership (value is list)
            - contains / startswith / endswith: wildcard text matching
            - exists / missing: field presence/absence
            - wildcard / matches: raw wildcard pattern

            SELECT:
            - Provide a comma string or list of fields.

            SORT:
            - Provide a single string (e.g., "genome_name asc") or
              a list of { "field": "...", "dir": "asc|desc" }.

            TIPS:
            - Use solr_collection_parameters to discover valid fields per collection.
            - Keep filters minimal when countOnly is True to reduce payload.
            - genome_feature queries automatically constrain to patric_id:*."""


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

