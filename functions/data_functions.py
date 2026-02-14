"""
BV-BRC Data Functions

This module provides data query functions for the BV-BRC Solr API.
Combines mvp_functions and common_functions from the data-mcp-server.
"""

import os
import re
import json
import time
import asyncio
from typing import Any, Dict, List, Tuple, Optional, Set
from bvbrc_solr_api import create_client, query
from bvbrc_solr_api.core.solr_http_client import async_select as solr_select
from common.llm_client import LLMClient

# Configuration constants
CURSOR_BATCH_SIZE = 1000
# Timeout for Solr queries in seconds (5 minutes for large single-page queries)
SOLR_QUERY_TIMEOUT = 300.0
# Maximum retry attempts for failed queries
MAX_RETRIES = 3
# Base seconds for exponential backoff during retries
RETRY_BACKOFF_BASE = 1.0



def _load_prompt_file(filename: str) -> str:
    """Load a prompt file from the prompts directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(current_dir, "..", "prompts", filename)
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def _strip_markdown_code_fence(text: str) -> str:
    """Remove top-level markdown code fences from an LLM response if present."""
    response_text = text.strip()
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        response_text = "\n".join(lines).strip()
    return response_text


def _available_collections() -> List[str]:
    """List available collection names based on prompt files."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_dir = os.path.join(current_dir, "..", "prompts", "solr_collections")
    try:
        return sorted(
            f[:-4]
            for f in os.listdir(prompts_dir)
            if f.endswith(".txt")
        )
    except Exception:
        # Fallback to empty list if filesystem lookup fails.
        return []


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


async def query_direct(core: str, filter_str: str = "", options: Dict[str, Any] = None,
                base_url: str = None, headers: Dict[str, str] = None,
                cursorId: str | None = None, countOnly: bool = False,
                batch_size: Optional[int] = None) -> Dict[str, Any]:
    """
    Query BV-BRC data directly using core name and filter string with cursor-based pagination.
    
    This async function performs a single page query and returns immediately with results
    and a cursor for fetching subsequent pages. Multiple concurrent queries from different
    users are fully supported without blocking.
    
    Args:
        core: The core/collection name (e.g., "genome", "genome_feature")
        filter_str: Solr query expression (e.g., "genome_id:123.45" or "species:\"Escherichia coli\"")
        options: Optional query options (e.g., {"select": ["field1", "field2"], "sort": "field_name"})
        base_url: Optional base URL override
        headers: Optional headers override
        cursorId: Cursor ID for pagination (optional, use "*" or None for first page)
        countOnly: If True, return only the total count without fetching documents
        batch_size: Number of rows to return per page (optional, defaults to CURSOR_BATCH_SIZE=1000)
        
    Returns:
        Dict with keys depending on countOnly:
        - If countOnly is True: { "numFound": <total_count> }
        - Otherwise: { 
            "results": [...],           # Array of result documents
            "count": <batch_count>,     # Number of results in this page  
            "numFound": <total>,        # Total results matching query
            "nextCursorId": <str|None>  # Cursor for next page (None = last page)
          }
        
    Note:
        Batch size defaults to 1000 entries per page. Use nextCursorId from the response
        to fetch the next batch by passing it as cursorId in a subsequent call.
        
    Example:
        # Fetch all results across multiple pages
        cursor = "*"
        all_results = []
        while cursor:
            response = await query_direct("genome", "genus:Salmonella", cursorId=cursor)
            all_results.extend(response["results"])
            cursor = response.get("nextCursorId")
            if not cursor:
                break
    """
    # Build context_overrides with timeout and any provided base_url/headers
    context_overrides = {"timeout": SOLR_QUERY_TIMEOUT}
    if base_url:
        context_overrides["base_url"] = base_url
    if headers:
        context_overrides["headers"] = headers
    
    options = options or {}
    
    # Use provided batch_size or fall back to default constant
    rows_per_page = batch_size if batch_size is not None else CURSOR_BATCH_SIZE
    
    print(f"[query_direct] Query params: core='{core}', filter_str='{filter_str}', rows={rows_per_page}, cursorId={cursorId or '*'}, countOnly={countOnly}")
    if options:
        print(f"[query_direct] Options: {options}")
    
    # Execute single-page query with strict async client lifecycle.
    async with create_client(context_overrides) as client:
        # Prepare a configured CursorPager via the client
        pager = getattr(client, core).stream_all_solr(
            rows=rows_per_page,
            sort=options.get("sort"),
            fields=options.get("select"),
            q_expr=filter_str if filter_str else "*:*",
            start_cursor=cursorId or "*",
            context_overrides=context_overrides
        )
        
        # Helper to execute a single Solr select call with retry logic
        async def _single_page_with_retry(cursor_mark: str) -> Tuple[List[Dict[str, Any]], str | None, Optional[int]]:
            """Fetch a single page with exponential backoff retry on failure."""
            last_exception = None
            
            for attempt in range(MAX_RETRIES):
                try:
                    params = dict(pager.base_params)
                    params["rows"] = pager.rows
                    params["sort"] = pager.sort
                    params["cursorMark"] = cursor_mark
                    
                    solr_url = f"{pager.base_url}/{pager.collection}/"
                    print(f"[query_direct] Solr HTTP request: POST {solr_url}")
                    print(f"[query_direct] Solr params: {params}")
                    if pager.headers:
                        print(f"[query_direct] Solr headers: {list(pager.headers.keys())}")
                    
                    result = await solr_select(
                        pager.collection,
                        params,
                        client=client._http_client,
                        base_url=pager.base_url,
                        headers=pager.headers,
                        auth=pager.auth,
                        timeout=pager.timeout,
                    )
                    
                    response = result.get("response", {})
                    docs: List[Dict[str, Any]] = response.get("docs", [])
                    next_cursor = result.get("nextCursorMark")
                    num_found = response.get("numFound")
                    print(f"[query_direct] Solr response: numFound={num_found}, docs_returned={len(docs)}, nextCursorMark={next_cursor}")
                    return docs, next_cursor, num_found
                    
                except Exception as e:
                    last_exception = e
                    if attempt < MAX_RETRIES - 1:
                        wait_time = RETRY_BACKOFF_BASE * (2 ** attempt)
                        print(f"[query_direct] Solr request failed (attempt {attempt + 1}/{MAX_RETRIES}): {str(e)}, retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        print(f"[query_direct] Solr request failed after {MAX_RETRIES} attempts: {str(e)}")
                        raise last_exception
            
            if last_exception:
                raise last_exception
            raise RuntimeError("Unexpected error in _single_page_with_retry")

        if countOnly:
            # Return Solr-reported total without fetching documents
            docs, _next_cursor, num_found = await _single_page_with_retry(pager.cursor)
            return {"numFound": num_found if num_found is not None else len(docs)}

        # Fetch a single batch/page and return nextCursorId for pagination
        docs, next_cursor, num_found = await _single_page_with_retry(pager.cursor)
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
    prompt_file = os.path.join(prompts_dir, "solr_collections", f"{collection}.txt")

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


def _sanitize_query_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize and validate planner output into query_collection-compatible args."""
    if not isinstance(plan, dict):
        raise ValueError("Plan must be a JSON object")

    collection = str(plan.get("collection", "")).strip()
    if not collection:
        raise ValueError("Plan must include a non-empty 'collection'")

    filters = plan.get("filters")

    # Normalize select/sort using existing helpers for consistent behavior.
    select = normalize_select(plan.get("select"))
    sort = normalize_sort(plan.get("sort"))

    count_only = bool(plan.get("countOnly", False))
    batch_size = plan.get("batchSize")
    if batch_size is not None:
        if isinstance(batch_size, bool):
            raise ValueError("batchSize must be an integer")
        try:
            batch_size = int(batch_size)
        except Exception:
            raise ValueError("batchSize must be an integer")
        if batch_size < 1 or batch_size > 10000:
            raise ValueError("batchSize must be between 1 and 10000")

    num_results = plan.get("num_results")
    if num_results is not None:
        if isinstance(num_results, bool):
            raise ValueError("num_results must be an integer")
        try:
            num_results = int(num_results)
        except Exception:
            raise ValueError("num_results must be an integer")
        if num_results < 1:
            raise ValueError("num_results must be >= 1")

    output_format = str(plan.get("format", "tsv")).strip().lower()
    if output_format not in ("json", "tsv"):
        raise ValueError("format must be 'json' or 'tsv'")

    # Validate fields in structured filters against the selected collection.
    allowed_fields = set(get_collection_fields(collection))
    invalid_fields = validate_filter_fields(filters, allowed_fields) if filters else []
    if invalid_fields:
        raise ValueError(
            f"Invalid field(s) for collection '{collection}': {', '.join(invalid_fields)}"
        )

    sanitized: Dict[str, Any] = {
        "collection": collection,
        "filters": filters,
        "countOnly": count_only,
        "format": output_format,
    }
    if select:
        sanitized["select"] = select
    if sort:
        sanitized["sort"] = sort
    if batch_size is not None:
        sanitized["batchSize"] = batch_size
    if num_results is not None:
        sanitized["num_results"] = num_results
    return sanitized


def select_collection_for_query(user_query: str, llm_client: LLMClient) -> Dict[str, Any]:
    """
    STEP 1: Select the best collection for a natural-language query.
    Returns a JSON object containing at least 'collection'.
    """
    system_prompt = _load_prompt_file("data_query_collection_selection.txt")
    user_prompt = f"""Select the best BV-BRC collection for this query.

USER QUERY: {user_query}

AVAILABLE COLLECTIONS:
{list_solr_collections()}

Return ONLY a JSON object with keys:
- collection (single best collection)
- reasoning (brief string)
- confidence (0..1)
- alternatives (optional list of collection names)"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = llm_client.chat_completion(messages)
    response_text = _strip_markdown_code_fence(response)
    selection = json.loads(response_text)

    if not isinstance(selection, dict):
        raise ValueError("Collection selection response must be a JSON object")
    selected_collection = str(selection.get("collection", "")).strip()
    if not selected_collection:
        raise ValueError("Collection selection did not include 'collection'")

    available = set(_available_collections())
    if available and selected_collection not in available:
        raise ValueError(
            f"Unknown collection selected: {selected_collection}. "
            f"Available collections: {', '.join(sorted(available))}"
        )

    return selection


def generate_query_plan_for_collection(
    user_query: str,
    collection: str,
    llm_client: LLMClient,
    validation_error: Optional[str] = None,
    previous_plan: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    STEP 2: Generate query arguments for bvbrc_query_collection from user query.
    """
    system_prompt = _load_prompt_file("data_query_parameter_generation.txt")
    collection_parameters = lookup_parameters(collection)

    user_prompt = f"""Generate query arguments for bvbrc_query_collection.

USER QUERY: {user_query}
SELECTED COLLECTION: {collection}

COLLECTION FIELDS AND TYPES:
{collection_parameters}

FILTER AND QUERY RULES:
{query_info()}

Return ONLY a JSON object with:
- collection
- filters
- select
- sort
- batchSize
- num_results (optional integer: total limit on results across all pages)
- countOnly
- format
- assumptions (optional list)
- questions_for_user (optional list)

Constraints:
- Use only this collection: {collection}
- No multi-step or cross-collection planning
- Use structured filters, never raw Solr syntax
- Keep select concise and relevant
- If user wants a specific number of results, set num_results to that limit"""

    if validation_error:
        user_prompt += f"""

VALIDATION ERROR (fix this):
{validation_error}
"""
    if previous_plan:
        user_prompt += f"""

PREVIOUS PLAN (fix only what is needed):
{json.dumps(previous_plan, indent=2)}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = llm_client.chat_completion(messages)
    response_text = _strip_markdown_code_fence(response)
    raw_plan = json.loads(response_text)
    return raw_plan


def create_query_plan_internal(
    user_query: str,
    llm_client: LLMClient
) -> Dict[str, Any]:
    """
    Two-step planner for data queries:
      1) select collection
      2) generate collection-specific query arguments
    """
    if not user_query or not str(user_query).strip():
        return {"error": "user_query is required"}

    selection = select_collection_for_query(user_query, llm_client)
    collection = str(selection.get("collection", "")).strip()

    attempt = 0
    max_attempts = 2
    validation_error: Optional[str] = None
    previous_plan: Optional[Dict[str, Any]] = None

    while attempt < max_attempts:
        raw_plan = generate_query_plan_for_collection(
            user_query=user_query,
            collection=collection,
            llm_client=llm_client,
            validation_error=validation_error,
            previous_plan=previous_plan
        )
        # Force selected collection from step 1 even if model drifts.
        if isinstance(raw_plan, dict):
            raw_plan["collection"] = collection

        try:
            sanitized = _sanitize_query_plan(raw_plan)
            return {
                "plan": sanitized,
                "selection": selection,
                "rawPlan": raw_plan
            }
        except Exception as e:
            validation_error = str(e)
            previous_plan = raw_plan if isinstance(raw_plan, dict) else None
            attempt += 1
            if attempt >= max_attempts:
                return {
                    "error": "Failed to build a valid query plan",
                    "selection": selection,
                    "validationError": validation_error,
                    "rawPlan": raw_plan
                }

    return {
        "error": "Failed to build query plan for an unknown reason",
        "selection": selection
    }


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
    prompt_file = os.path.join(prompts_dir, "solr_collections", f"{collection}.txt")
    
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


async def get_feature_sequence_by_id(
    patric_ids: List[str],
    sequence_type: str,
    base_url: str = None,
    headers: Dict[str, str] = None
) -> Dict[str, Any]:
    """
    Get the nucleotide or amino acid sequences for features by their PATRIC IDs.
    
    This function performs a two-step batch query:
    1. Query genome_feature collection to get the aa_sequence_md5 or na_sequence_md5 for all IDs
    2. Query feature_sequence collection with those md5s to get the actual sequences
    
    Args:
        patric_ids: List of PATRIC IDs (e.g., ["fig|91750.131.peg.1283", "fig|91750.131.peg.1284"])
        sequence_type: Type of sequence - "na" for nucleotide or "aa" for amino acid
        base_url: Optional base URL override
        headers: Optional headers override
        
    Returns:
        Dict with either:
        - Success: {
            "results": [
              {"sequence": <str>, "md5": <str>, "sequence_type": <str>, "patric_id": <str>, "length": <int>},
              ...
            ],
            "count": <int>,
            "requested": <int>,
            "source": "bvbrc-mcp-data"
          }
        - Error: {"error": <error_message>, "source": "bvbrc-mcp-data"}
    """
    # Validate sequence_type parameter
    sequence_type = sequence_type.lower()
    if sequence_type not in ["na", "aa"]:
        return {
            "error": f"Invalid sequence_type: {sequence_type}. Must be 'na' or 'aa'.",
            "source": "bvbrc-mcp-data"
        }
    
    if not patric_ids or not isinstance(patric_ids, list):
        return {
            "error": "patric_ids must be a non-empty list",
            "source": "bvbrc-mcp-data"
        }
    
    # Remove duplicates and empty strings
    patric_ids = [pid for pid in patric_ids if pid and str(pid).strip()]
    patric_ids = list(dict.fromkeys(patric_ids))  # Remove duplicates while preserving order
    
    if not patric_ids:
        return {
            "error": "patric_ids list is empty after filtering",
            "source": "bvbrc-mcp-data"
        }
    
    md5_field = "aa_sequence_md5" if sequence_type == "aa" else "na_sequence_md5"
    
    print(f"Step 1: Querying genome_feature for {len(patric_ids)} patric_id(s) to get {md5_field}")
    
    # Step 1: Query genome_feature to get the md5 hashes for all IDs
    try:
        # Build filter using structured filters for the "in" operator
        filters = {
            "logic": "and",
            "filters": [
                {"field": "patric_id", "op": "in", "value": patric_ids}
            ]
        }
        
        filter_str = build_filter(filters)
        # Add the patric_id:* constraint for genome_feature collection
        filter_str = f"({filter_str}) AND patric_id:*"
        
        feature_result = await query_direct(
            core="genome_feature",
            filter_str=filter_str,
            options={"select": [md5_field, "patric_id"]},
            base_url=base_url,
            headers=headers,
            batch_size=min(len(patric_ids) * 2, 10000)  # Reasonable batch size
        )
        
        feature_results = feature_result.get("results", [])
        if not feature_results:
            return {
                "error": f"No features found with the provided patric_ids",
                "requested_ids": patric_ids,
                "source": "bvbrc-mcp-data"
            }
        
        print(f"Step 1: Found {len(feature_results)} feature(s)")
        
        # Build mapping of md5 -> patric_id(s) and collect all md5 hashes
        md5_to_patric_ids: Dict[str, List[str]] = {}
        patric_id_to_md5: Dict[str, str] = {}
        missing_md5_ids: List[str] = []
        
        for feature in feature_results:
            pid = feature.get("patric_id")
            md5_hash = feature.get(md5_field)
            
            if md5_hash:
                patric_id_to_md5[pid] = md5_hash
                if md5_hash not in md5_to_patric_ids:
                    md5_to_patric_ids[md5_hash] = []
                md5_to_patric_ids[md5_hash].append(pid)
            else:
                missing_md5_ids.append(pid)
        
        if not patric_id_to_md5:
            return {
                "error": f"None of the features have {md5_field} values",
                "hint": f"These features may not have {sequence_type.upper()} sequence data available",
                "requested_ids": patric_ids,
                "found_ids": [f.get("patric_id") for f in feature_results],
                "source": "bvbrc-mcp-data"
            }
        
        unique_md5s = list(md5_to_patric_ids.keys())
        print(f"Step 2: Found {len(unique_md5s)} unique MD5 hash(es), querying feature_sequence collection")
        
        # Step 2: Query feature_sequence with all the md5 hashes
        sequence_filters = {
            "field": "md5",
            "op": "in",
            "value": unique_md5s
        }
        
        sequence_filter_str = build_filter(sequence_filters)
        
        sequence_result = await query_direct(
            core="feature_sequence",
            filter_str=sequence_filter_str,
            options={"select": ["sequence", "md5", "sequence_type"]},
            base_url=base_url,
            headers=headers,
            batch_size=min(len(unique_md5s) * 2, 10000)
        )
        
        sequence_results = sequence_result.get("results", [])
        if not sequence_results:
            return {
                "error": f"No sequences found for the MD5 hashes",
                "hint": "The sequences may not be available in the feature_sequence collection",
                "md5_hashes": unique_md5s,
                "source": "bvbrc-mcp-data"
            }
        
        print(f"Step 2: Found {len(sequence_results)} sequence(s)")
        
        # Build mapping of md5 -> sequence
        md5_to_sequence: Dict[str, str] = {}
        for seq_data in sequence_results:
            md5 = seq_data.get("md5")
            sequence = seq_data.get("sequence", "")
            if md5:
                md5_to_sequence[md5] = sequence
        
        # Build final results list
        results_list: List[Dict[str, Any]] = []
        found_ids: Set[str] = set()
        
        for pid in patric_ids:
            if pid in patric_id_to_md5:
                md5 = patric_id_to_md5[pid]
                if md5 in md5_to_sequence:
                    sequence = md5_to_sequence[md5]
                    results_list.append({
                        "patric_id": pid,
                        "sequence": sequence,
                        "md5": md5,
                        "sequence_type": sequence_type,
                        "length": len(sequence)
                    })
                    found_ids.add(pid)
        
        # Collect IDs that weren't found or didn't have sequences
        not_found = [pid for pid in patric_ids if pid not in found_ids]
        
        print(f"Successfully retrieved {len(results_list)} {sequence_type.upper()} sequence(s)")
        
        result = {
            "results": results_list,
            "count": len(results_list),
            "requested": len(patric_ids),
            "source": "bvbrc-mcp-data"
        }
        
        if not_found:
            result["not_found"] = not_found
            result["warnings"] = [
                f"{len(not_found)} patric_id(s) not found or missing sequence data"
            ]
        
        if missing_md5_ids:
            if "warnings" not in result:
                result["warnings"] = []
            result["warnings"].append(
                f"{len(missing_md5_ids)} feature(s) have no {md5_field} value"
            )
            result["missing_md5_ids"] = missing_md5_ids
        
        return result
        
    except Exception as e:
        return {
            "error": f"Error retrieving sequences: {str(e)}",
            "source": "bvbrc-mcp-data"
        }

