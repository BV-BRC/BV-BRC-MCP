#!/usr/bin/env python3
"""
BV-BRC MVP Tools

This module contains MCP tools for querying MVP (Minimum Viable Product) data from BV-BRC.
"""

import json
import subprocess
import os
import re
from typing import Optional, Dict, Any, List

from fastmcp import FastMCP

from functions.data_functions import (
    query_direct,
    lookup_parameters,
    query_info,
    list_solr_collections,
    normalize_select,
    normalize_sort,
    build_filter,
    get_collection_fields,
    validate_filter_fields,
    create_query_plan_internal,
    select_collection_for_query,
    get_feature_sequence_by_id,
    CURSOR_BATCH_SIZE
)
from common.llm_client import create_llm_client_from_config

# Global variables to store configuration
_base_url = None
_token_provider = None
_llm_client = None


def _get_llm_client():
    """Create and cache the internal LLM client used for query planning."""
    global _llm_client
    if _llm_client is not None:
        return _llm_client

    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
        "config.json"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    _llm_client = create_llm_client_from_config(config)
    return _llm_client


def _contains_solr_syntax(query: str) -> bool:
    """Detect common raw Solr syntax patterns to enforce natural-language input."""
    if not query:
        return False

    # Fielded search (e.g., species:"E. coli"), boolean operators, range/grouping operators.
    solr_patterns = [
        r"\b[A-Za-z_][A-Za-z0-9_]*\s*:",  # field:
        r"\b(AND|OR|NOT)\b",
        r"[\(\)\[\]\{\}]",
        r"\bTO\b",
        r"[*?~^]",
    ]
    return any(re.search(pattern, query) for pattern in solr_patterns)


def _tokenize_keywords(text: str) -> List[str]:
    """
    Convert a natural-language query into search terms for keyword mode.
    Supports comma-separated phrases and fallback token splitting.
    """
    if not text:
        return []

    # Prefer comma-separated terms if present (allows multi-word keywords naturally).
    if "," in text:
        parts = [part.strip() for part in text.split(",")]
        return [part for part in parts if part]

    # Otherwise split on whitespace and strip punctuation.
    parts = re.split(r"\s+", text.strip())
    terms: List[str] = []
    for part in parts:
        cleaned = re.sub(r"^[^\w]+|[^\w]+$", "", part)
        if cleaned:
            terms.append(cleaned)
    return terms


def _quote_solr_term(term: str) -> str:
    """Quote and minimally escape a term for safe Solr q= usage."""
    escaped = str(term).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _build_global_search_q_expr(user_query: str, search_mode: str) -> Dict[str, Any]:
    """
    Build a Solr q expression from natural language for phrase/and/or modes.
    Returns dict containing q_expr and parsed keywords for transparency.
    """
    normalized_mode = (search_mode or "phrase").strip().lower()
    if normalized_mode not in {"phrase", "and", "or"}:
        raise ValueError("search_mode must be one of: phrase, and, or")

    if normalized_mode == "phrase":
        return {
            "q_expr": _quote_solr_term(user_query.strip()),
            "keywords": [user_query.strip()],
            "searchMode": "phrase"
        }

    keywords = _tokenize_keywords(user_query)
    if not keywords:
        raise ValueError("Could not parse keywords from user_query")
    
    # If only one keyword, treat it as a phrase search (single-term search)
    if len(keywords) == 1:
        return {
            "q_expr": _quote_solr_term(keywords[0]),
            "keywords": keywords,
            "searchMode": normalized_mode
        }

    joiner = " AND " if normalized_mode == "and" else " OR "
    q_expr = "(" + joiner.join(_quote_solr_term(term) for term in keywords) + ")"
    return {
        "q_expr": q_expr,
        "keywords": keywords,
        "searchMode": normalized_mode
    }


def convert_json_to_tsv(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert a list of JSON objects to TSV format using jq.
    
    Args:
        results: List of dictionaries to convert
        
    Returns:
        Dict with either:
        - {"tsv": <tsv_string>} on success
        - {"error": <error_message>} on failure
    """
    if not results:
        # Empty results - return empty TSV
        print("  TSV conversion: Empty results, returning empty TSV")
        return {"tsv": ""}
    
    print(f"  TSV conversion: Starting conversion of {len(results)} results to TSV format...")
    
    try:
        # Prepare jq command to convert JSON array to TSV
        # This extracts all unique keys from the first object as headers,
        # then converts each object to a TSV row
        jq_command = [
            "jq",
            "-r",
            "(.[0] | keys_unsorted) as $keys | $keys, (.[] | [.[$keys[]] | tostring]) | @tsv"
        ]
        
        # Convert results to JSON string
        json_input = json.dumps(results)
        print(f"  TSV conversion: Running jq command to convert JSON to TSV...")
        
        # Run jq command
        result = subprocess.run(
            jq_command,
            input=json_input,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            print(f"  TSV conversion: FAILED - jq returned error code {result.returncode}")
            print(f"  TSV conversion error: {result.stderr}")
            return {
                "error": f"jq conversion failed: {result.stderr}",
                "hint": "Ensure jq is installed on your system"
            }
        
        tsv_lines = result.stdout.count('\n')
        print(f"  TSV conversion: SUCCESS - Converted to TSV with {tsv_lines} lines (including header)")
        return {"tsv": result.stdout}
        
    except FileNotFoundError:
        print("  TSV conversion: FAILED - jq command not found")
        return {
            "error": "jq command not found",
            "hint": "Please install jq to use TSV format. See https://jqlang.github.io/jq/download/"
        }
    except subprocess.TimeoutExpired:
        print("  TSV conversion: FAILED - Conversion timed out after 30 seconds")
        return {
            "error": "TSV conversion timed out (>30s)",
            "hint": "Try reducing the batch size"
        }
    except Exception as e:
        print(f"  TSV conversion: FAILED - Unexpected error: {str(e)}")
        return {
            "error": f"TSV conversion error: {str(e)}"
        }


def register_data_tools(mcp: FastMCP, base_url: str, token_provider=None):
    """
    Register all MVP-related MCP tools with the FastMCP server.
    
    Args:
        mcp: FastMCP server instance
        base_url: Base URL for BV-BRC API
        token_provider: TokenProvider instance for handling authentication tokens (optional)
    """
    global _base_url, _token_provider
    _base_url = base_url
    _token_provider = token_provider

    # New, clearer tool names
    @mcp.tool(annotations={"readOnlyHint": True})
    async def bvbrc_query_collection(collection: str,
                               filters: Optional[Dict[str, Any]] = None,
                               select: Optional[Any] = None,
                               sort: Optional[Any] = None,
                               cursorId: Optional[str] = None,
                               countOnly: bool = False,
                               batchSize: Optional[int] = None,
                               num_results: Optional[int] = None,
                               format: Optional[str] = "tsv",
                               token: Optional[str] = None) -> Dict[str, Any]:
        """
        Query BV-BRC data with structured filters; Solr syntax is handled for you.
        
        ⚠️ WORKFLOW: For natural language queries, ALWAYS call bvbrc_plan_query_collection 
        FIRST to generate parameters. ONLY call this tool directly when you already have 
        structured parameters from the planner or explicitly provided by the user
        
        Returns a single page of results with cursor for pagination. For large result sets,
        make multiple calls using the returned nextCursorId until it becomes null.
        
        Args:
            collection: Collection name (e.g., "genome", "genome_feature").
            filters: Structured filter object describing conditions and grouping. Example:
                {
                  "logic": "and",
                  "filters": [
                    { "field": "genome_name", "op": "eq", "value": "Escherichia coli" },
                    { "logic": "or", "filters": [
                        { "field": "resistant_phenotype", "op": "eq", "value": "Resistant" },
                        { "field": "resistant_phenotype", "op": "eq", "value": "Intermediate" }
                      ]
                    }
                  ]
                }
            select: List of fields or comma-separated string (optional).
            sort: Sort string or list of field/direction dicts (optional).
            cursorId: Cursor ID for pagination. Use "*" or omit for first page.
                     Pass the returned nextCursorId to fetch subsequent pages.
                     When nextCursorId is null, you've reached the end.
            countOnly: If True, only return the total count without data.
            batchSize: Number of rows per page (defaults to 1000, range: 1-10000).
                      Use smaller values for faster initial response, larger for fewer round trips.
            num_results: Optional total limit on number of results to return across all pages.
                        If provided, the tool will fetch pages until this limit is reached (or results are exhausted).
                        Final page will be truncated if needed. If not provided, returns single page only.
            format: Output format - "tsv" (default) or "json". 
                   TSV format requires jq to be installed on the system.
                   When format="tsv", results are converted to tab-separated values with headers.
            token: Authentication token (optional, auto-detected if token_provider is configured).
            
        Returns:
            Dict with structure:
            - If countOnly=True: {"numFound": <int>, "source": "bvbrc-mcp-data"}
            - returns: {
                "results": [...] | "tsv": <tsv_string>,           # Array of result objects or TSV string
                "count": <int>,             # Number of results in this page
                "numFound": <int>,          # Total results matching query
                "nextCursorId": <str|null>, # Pass this to get next page (null = no more pages)
                "source": "bvbrc-mcp-data"
              }
        """
        # Validate format parameter
        if format not in ["json", "tsv"]:
            return {
                "error": f"Invalid format: {format}. Must be 'json' or 'tsv'.",
                "source": "bvbrc-mcp-data"
            }
        
        mode_str = "count-only" if countOnly else "paginated-query"
        print(f"Querying collection: {collection}, mode: {mode_str}, format: {format}, cursorId: {cursorId or '*'}")
        
        options: Dict[str, Any] = {}
        select_fields = normalize_select(select)
        sort_expr = normalize_sort(sort)
        if select_fields:
            options["select"] = select_fields
        if sort_expr:
            options["sort"] = sort_expr
        
        # Validate filter fields against the collection's allowed fields
        allowed_fields = set(get_collection_fields(collection))
        invalid_fields = validate_filter_fields(filters, allowed_fields) if filters else []
        if invalid_fields:
            sample_fields = sorted(list(allowed_fields))[:25] if allowed_fields else []
            return {
                "error": f"Invalid field(s) for collection '{collection}': {', '.join(invalid_fields)}",
                "hint": "Call bvbrc_collection_fields_and_parameters to see valid fields.",
                "allowedFieldsSample": sample_fields,
                "source": "bvbrc-mcp-data"
            }

        # Build Solr query from structured filters
        filter_str = build_filter(filters)

        # Apply collection-specific defaults
        if collection == "genome_feature":
            auto = "patric_id:*"
            if filter_str and filter_str != "*:*":
                filter_str = f"({filter_str}) AND {auto}"
            else:
                filter_str = auto

        # Validate batchSize if provided
        if batchSize is not None:
            if batchSize < 1 or batchSize > 10000:
                return {
                    "error": f"Invalid batchSize: {batchSize}. Must be between 1 and 10000.",
                    "source": "bvbrc-mcp-data"
                }

        # Validate num_results if provided
        if num_results is not None:
            if num_results < 1:
                return {
                    "error": f"Invalid num_results: {num_results}. Must be >= 1.",
                    "source": "bvbrc-mcp-data"
                }
            # If num_results is provided, we'll fetch multiple pages
            # Adjust batchSize to not exceed num_results if needed
            if batchSize is None:
                batchSize = min(CURSOR_BATCH_SIZE, num_results)
            elif batchSize > num_results:
                batchSize = num_results

        # Authentication headers
        headers: Optional[Dict[str, str]] = None
        if _token_provider:
            auth_token = _token_provider.get_token(token)
            if auth_token:
                headers = {"Authorization": auth_token}
        elif token:
            headers = {"Authorization": token}
        
        # Collection-specific headers
        if collection == "genome_sequence":
            if headers is None:
                headers = {}
            headers["http_accept"] = "application/dna+fasta"
        
        print(f"Filter is {filter_str}")
        
        try:
            # If num_results is provided and we're not in countOnly mode, fetch pages until limit
            if num_results is not None and not countOnly and (cursorId is None or cursorId == "*"):
                # Fetch multiple pages until we reach num_results
                all_results = []
                current_cursor = cursorId or "*"
                total_fetched = 0
                last_page_result = None
                
                while total_fetched < num_results:
                    # Calculate how many we need from this page
                    remaining = num_results - total_fetched
                    page_batch_size = min(batchSize or CURSOR_BATCH_SIZE, remaining)
                    
                    page_result = await query_direct(
                        collection, filter_str, options, _base_url,
                        headers=headers, cursorId=current_cursor, countOnly=False,
                        batch_size=page_batch_size
                    )
                    last_page_result = page_result
                    
                    page_results = page_result.get("results", [])
                    if not page_results:
                        # No more results available
                        break
                    
                    # Add results up to the limit
                    needed = num_results - total_fetched
                    all_results.extend(page_results[:needed])
                    total_fetched += len(page_results[:needed])
                    
                    # Check if we've reached the limit or there are no more pages
                    next_cursor = page_result.get("nextCursorId")
                    if total_fetched >= num_results or not next_cursor:
                        break
                    
                    current_cursor = next_cursor
                
                # Build final result
                num_found = last_page_result.get("numFound", len(all_results)) if last_page_result else len(all_results)
                next_cursor_id = last_page_result.get("nextCursorId") if (last_page_result and total_fetched < num_results) else None
                result = {
                    "results": all_results,
                    "count": len(all_results),
                    "numFound": num_found,
                    "nextCursorId": next_cursor_id,
                    "limit": num_results,
                    "limitReached": total_fetched >= num_results
                }
            else:
                # Single page query (original behavior)
                result = await query_direct(
                    collection, filter_str, options, _base_url, 
                    headers=headers, cursorId=cursorId, countOnly=countOnly,
                    batch_size=batchSize
                )
            
            # Prefer count for the returned page; fall back to numFound if needed
            observed_count = result.get("count", result.get("numFound"))
            if countOnly:
                print(f"Query found {observed_count} total results.")
            else:
                if num_results is not None:
                    print(f"Query returned {observed_count} results (limit: {num_results}).")
                else:
                    print(f"Query returned {observed_count} results for this page.")
                if result.get("nextCursorId"):
                    print(f"  More results available. Use nextCursorId to fetch next page.")
            
            # Convert to TSV if requested
            if format == "tsv" and not countOnly:
                results_to_convert = result.get("results", [])
                print(f"  Converting {len(results_to_convert)} results from JSON to TSV format...")
                conversion_result = convert_json_to_tsv(results_to_convert)
                
                # Check if conversion was successful
                if "error" in conversion_result:
                    # Return error with original JSON as fallback
                    print(f"  TSV conversion failed, falling back to JSON format")
                    return {
                        "error": conversion_result["error"],
                        "hint": conversion_result.get("hint", ""),
                        "results": result.get("results", []),  # Fallback to JSON
                        "count": result.get("count"),
                        "numFound": result.get("numFound"),
                        "nextCursorId": result.get("nextCursorId"),
                        "source": "bvbrc-mcp-data"
                    }
                
                # Replace results with TSV string
                result["tsv"] = conversion_result["tsv"]
                del result["results"]  # Remove JSON results
                print(f"  TSV conversion complete. Results replaced with TSV string.")
            
            # Add 'source' field to the top-level response
            result['source'] = 'bvbrc-mcp-data'
            
            return result
            
        except Exception as e:
            return {
                "error": f"Error querying {collection}: {str(e)}",
                "source": "bvbrc-mcp-data"
            }

    @mcp.tool(annotations={"readOnlyHint": True})
    def bvbrc_collection_fields_and_parameters(collection: str) -> str:
        """
        Get fields and query parameters for a given BV-BRC collection.
        
        Args:
            collection: The collection name (e.g., "genome")
        
        Returns:
            String with the parameters for the given collection
        """
        return lookup_parameters(collection)

    @mcp.tool(annotations={"readOnlyHint": True})
    def bvbrc_query_examples_and_rules() -> str:
        """
        Get general query instructions and examples for all collections.
        
        Returns:
            String with general query instructions and formatting guidelines
        """
        print("Fetching general query instructions.")
        return query_info()

    @mcp.tool(annotations={"readOnlyHint": True})
    def bvbrc_list_collections() -> str:
        """
        List all available BV-BRC collections.
        
        Returns:
            String with the available collections
        """
        print("Fetching available collections.")
        return list_solr_collections()

    @mcp.tool(annotations={"readOnlyHint": True})
    async def bvbrc_plan_query_collection(user_query: str) -> Dict[str, Any]:
        """
        Plan a single-collection query from natural language using a two-step internal LLM procedure:
        1) select collection
        2) generate validated query arguments

        ⚠️ ALWAYS call this tool FIRST for natural language queries (e.g., "find E. coli genomes", 
        "show resistant strains"). NEVER manually construct query parameters from natural language.
        
        The returned plan contains all parameters needed to call bvbrc_query_collection.

        Use the returned plan as arguments to bvbrc_query_collection.

        Args:
            user_query: Natural language description of the desired BV-BRC data query.
                This planner tool accepts only this parameter. Do not pass query args
                such as format/select/sort/filters here; those belong in the returned
                plan for bvbrc_query_collection.

        Returns:
            Dict with:
            - plan: Validated query arguments compatible with bvbrc_query_collection
            - selection: Collection selection details from planner step 1
            - nextToolCall: Suggested direct call to bvbrc_query_collection
            - source: "bvbrc-mcp-data"
            Or error details if planning fails.
        """
        if not user_query or not str(user_query).strip():
            return {
                "error": "user_query parameter is required",
                "errorType": "INVALID_PARAMETERS",
                "hint": "Provide a natural language query to plan",
                "source": "bvbrc-mcp-data"
            }

        try:
            llm_client = _get_llm_client()
            planning_result = create_query_plan_internal(user_query, llm_client)

            if "error" in planning_result:
                return {
                    "error": planning_result.get("error"),
                    "errorType": "PLANNING_FAILED",
                    "selection": planning_result.get("selection"),
                    "validationError": planning_result.get("validationError"),
                    "rawPlan": planning_result.get("rawPlan"),
                    "source": "bvbrc-mcp-data"
                }

            plan = planning_result.get("plan", {})
            return {
                "plan": plan,
                "selection": planning_result.get("selection", {}),
                "nextToolCall": {
                    "tool": "bvbrc_query_collection",
                    "arguments": plan
                },
                "source": "bvbrc-mcp-data"
            }
        except FileNotFoundError as e:
            return {
                "error": f"Configuration or prompt file not found: {str(e)}",
                "errorType": "CONFIGURATION_ERROR",
                "hint": "Ensure config/config.json and planning prompt files exist",
                "source": "bvbrc-mcp-data"
            }
        except Exception as e:
            return {
                "error": f"Query planning failed: {str(e)}",
                "errorType": "PLANNING_FAILED",
                "source": "bvbrc-mcp-data"
            }

    @mcp.tool(annotations={"readOnlyHint": True})
    async def bvbrc_global_data_search(
        user_query: str,
        search_mode: str = "phrase",
        limit: Optional[int] = None,
        cursorId: Optional[str] = None,
        format: str = "json",
        token: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform global natural-language data search across BV-BRC by:
        1) selecting the best collection with an internal LLM
        2) executing a Solr q= query against that collection

        IMPORTANT:
        - user_query must be plain natural language or plain keywords.
        - Do NOT provide raw Solr syntax (no field:value, no boolean operators, no ranges).

        Search modes:
        - phrase: Exact phrase search of the full query text.
                  Example: "pseudomonas aeruginosa"
        - and: Multi-keyword search requiring all keywords.
               Prefer comma-separated keywords for best results.
               Example: "pseudomonas, aeruginosa, resistant"
        - or: Multi-keyword search requiring any keyword.
              Prefer comma-separated keywords for best results.
              Example: "pseudomonas, aeruginosa, staphylococcus"

        Args:
            user_query: Natural language query text or keyword list (not Solr syntax).
            search_mode: One of "phrase", "and", "or".
            limit: Optional limit on number of results to return (1-10000).
                   If not provided, callers can fetch ALL matching results
            cursorId: Optional cursor for pagination. Use "*" or omit for first page.
            format: Output format - "json" (default) or "tsv".
            token: Authentication token (optional, auto-detected if token_provider is configured).

        Returns:
            Dict including selected collection, generated q expression, and query results.
        """
        if not user_query or not str(user_query).strip():
            return {
                "error": "user_query parameter is required",
                "errorType": "INVALID_PARAMETERS",
                "hint": "Provide a natural-language query or keyword list",
                "source": "bvbrc-mcp-data"
            }

        if format not in ["json", "tsv"]:
            return {
                "error": f"Invalid format: {format}. Must be 'json' or 'tsv'.",
                "errorType": "INVALID_PARAMETERS",
                "source": "bvbrc-mcp-data"
            }

        if limit is not None and (limit < 1 or limit > 10000):
            return {
                "error": f"Invalid limit: {limit}. Must be between 1 and 10000.",
                "errorType": "INVALID_PARAMETERS",
                "source": "bvbrc-mcp-data"
            }

        query_text = str(user_query).strip()
        limit_str = str(limit) if limit is not None else "ALL (no limit)"
        print(f"[bvbrc_global_data_search] Input query: '{query_text}', search_mode: '{search_mode}', limit: {limit_str}, format: '{format}'")
        
        if _contains_solr_syntax(query_text):
            return {
                "error": "user_query appears to contain Solr syntax, which is not allowed for this tool.",
                "errorType": "INVALID_PARAMETERS",
                "hint": "Provide natural language or plain keywords only. Use search_mode for phrase/and/or behavior.",
                "source": "bvbrc-mcp-data"
            }

        try:
            search_info = _build_global_search_q_expr(query_text, search_mode)
            print(f"[bvbrc_global_data_search] Parsed search info: mode={search_info.get('searchMode')}, keywords={search_info.get('keywords')}, q_expr={search_info.get('q_expr')}")
        except Exception as e:
            print(f"[bvbrc_global_data_search] Failed to build search expression: {str(e)}")
            return {
                "error": str(e),
                "errorType": "INVALID_PARAMETERS",
                "source": "bvbrc-mcp-data"
            }

        try:
            llm_client = _get_llm_client()
            print(f"[bvbrc_global_data_search] Selecting collection for query...")
            selection = select_collection_for_query(query_text, llm_client)
            collection = str(selection.get("collection", "")).strip()
            print(f"[bvbrc_global_data_search] Selected collection: '{collection}', reasoning: {selection.get('reasoning', 'N/A')}")
            
            if not collection:
                return {
                    "error": "Collection selection failed to produce a collection.",
                    "errorType": "PLANNING_FAILED",
                    "selection": selection,
                    "source": "bvbrc-mcp-data"
                }

            q_expr = str(search_info.get("q_expr", "")).strip() or "*:*"
            if collection == "genome_feature":
                q_expr = f"({q_expr}) AND patric_id:*"

            limit_str = str(limit) if limit is not None else "ALL"
            print(f"[bvbrc_global_data_search] Final Solr query: collection='{collection}', q='{q_expr}', limit={limit_str}, cursorId={cursorId or '*'}")

            # Authentication headers
            headers: Optional[Dict[str, str]] = None
            if _token_provider:
                auth_token = _token_provider.get_token(token)
                if auth_token:
                    headers = {"Authorization": auth_token}
            elif token:
                headers = {"Authorization": token}

            page_size = min(limit, CURSOR_BATCH_SIZE) if limit is not None else CURSOR_BATCH_SIZE
            print(
                f"[bvbrc_global_data_search] Executing single-page query_direct: "
                f"core={collection}, filter_str={q_expr}, batch_size={page_size}, cursorId={cursorId or '*'}"
            )
            result = await query_direct(
                core=collection,
                filter_str=q_expr,
                options={},
                base_url=_base_url,
                headers=headers,
                cursorId=cursorId,
                countOnly=False,
                batch_size=page_size
            )
            if limit is not None:
                result["limit"] = limit
                result["limitReached"] = bool(result.get("count", 0) >= limit)
            print(
                f"[bvbrc_global_data_search] Query completed (single page): "
                f"count={result.get('count')}, numFound={result.get('numFound')}, "
                f"hasNextCursor={bool(result.get('nextCursorId'))}"
            )

            # Convert to TSV if requested
            if format == "tsv":
                results_to_convert = result.get("results", [])
                conversion_result = convert_json_to_tsv(results_to_convert)
                if "error" in conversion_result:
                    return {
                        "error": conversion_result["error"],
                        "hint": conversion_result.get("hint", ""),
                        "collection": collection,
                        "selection": selection,
                        "searchMode": search_info.get("searchMode"),
                        "keywords": search_info.get("keywords", []),
                        "q": q_expr,
                        "results": result.get("results", []),
                        "count": result.get("count"),
                        "numFound": result.get("numFound"),
                        "nextCursorId": result.get("nextCursorId"),
                        "source": "bvbrc-mcp-data"
                    }

                del result["results"]
                result["tsv"] = conversion_result["tsv"]

            result["collection"] = collection
            result["selection"] = selection
            result["searchMode"] = search_info.get("searchMode")
            result["keywords"] = search_info.get("keywords", [])
            result["q"] = q_expr
            result["source"] = "bvbrc-mcp-data"
            return result
        except Exception as e:
            return {
                "error": f"Global data search failed: {str(e)}",
                "errorType": "SEARCH_FAILED",
                "source": "bvbrc-mcp-data"
            }

    @mcp.tool(annotations={"readOnlyHint": True})
    async def bvbrc_get_feature_sequence_by_id(patric_ids: List[str], 
                                              type: str,
                                              token: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the nucleotide or amino acid sequences for genomic features by their PATRIC IDs.
        
        This tool performs a two-step batch query for efficient retrieval of multiple sequences:
        1. Queries genome_feature collection to get the sequence MD5 hashes for all IDs
        2. Queries feature_sequence collection with those MD5s to retrieve the actual sequences
        
        Args:
            patric_ids: List of PATRIC feature IDs (e.g., ["fig|91750.131.peg.1283", "fig|91750.131.peg.1284"])
                       Can be a single ID in a list or multiple IDs for batch retrieval
            type: Type of sequence to retrieve - "na" for nucleotide or "aa" for amino acid
            token: Authentication token (optional, auto-detected if token_provider is configured)
            
        Returns:
            Dict with either:
            - Success: {
                "results": [                      # Array of sequence results
                  {
                    "patric_id": <str>,          # PATRIC feature ID
                    "sequence": <str>,           # The actual DNA/RNA or protein sequence
                    "md5": <str>,                # MD5 hash of the sequence
                    "sequence_type": <str>,      # "na" or "aa"
                    "length": <int>              # Length of sequence
                  },
                  ...
                ],
                "count": <int>,                  # Number of sequences successfully retrieved
                "requested": <int>,              # Number of IDs requested
                "not_found": [<str>, ...],       # Optional: IDs that weren't found
                "warnings": [<str>, ...],        # Optional: Warning messages
                "source": "bvbrc-mcp-data"
              }
            - Error: {
                "error": <error_message>,
                "source": "bvbrc-mcp-data"
              }
        
        Example:
            # Get nucleotide sequence for a single feature
            result = bvbrc_get_feature_sequence_by_id(["fig|91750.131.peg.1283"], "na")
            
            # Get amino acid sequences for multiple features (batch query)
            result = bvbrc_get_feature_sequence_by_id([
                "fig|91750.131.peg.1283",
                "fig|91750.131.peg.1284",
                "fig|91750.131.peg.1285"
            ], "aa")
        """
        print(f"Getting {type.upper()} sequence(s) for {len(patric_ids)} feature(s)")
        
        # Authentication headers
        headers: Optional[Dict[str, str]] = None
        if _token_provider:
            auth_token = _token_provider.get_token(token)
            if auth_token:
                headers = {"Authorization": auth_token}
        elif token:
            headers = {"Authorization": token}
        
        try:
            result = await get_feature_sequence_by_id(
                patric_ids=patric_ids,
                sequence_type=type,
                base_url=_base_url,
                headers=headers
            )
            return result
        except Exception as e:
            return {
                "error": f"Error retrieving sequence: {str(e)}",
                "source": "bvbrc-mcp-data"
            }
