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
    list_solr_collections,
    normalize_select,
    normalize_sort,
    build_filter,
    get_collection_fields,
    validate_filter_fields,
    create_query_plan_internal,
    select_collection_for_query,
    get_feature_sequence_by_id,
    get_genome_sequence_by_id,
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


# Common stopwords to exclude from keyword queries
STOPWORDS = {
    "a", "an", "the", "from", "to", "in", "on", "at", "by", "for", "with", 
    "of", "and", "or", "but", "is", "are", "was", "were", "be", "been", 
    "being", "have", "has", "had", "do", "does", "did", "will", "would", 
    "should", "could", "may", "might", "must", "can", "this", "that", 
    "these", "those", "i", "you", "he", "she", "it", "we", "they", "what",
    "which", "who", "when", "where", "why", "how"
}

# Custom domain-specific stopwords to exclude
CUSTOM_STOPWORDS = {
    "genomes", "subtype", "year"
}


def _tokenize_keywords(text: str) -> List[str]:
    """
    Convert a natural-language query into search terms for keyword mode.
    Supports comma-separated phrases and fallback token splitting.
    Filters out common stopwords and custom stopwords.
    """
    if not text:
        return []

    # Combine all stopwords
    all_stopwords = STOPWORDS | CUSTOM_STOPWORDS

    # Prefer comma-separated terms if present (allows multi-word keywords naturally).
    if "," in text:
        parts = [part.strip() for part in text.split(",")]
        return [part for part in parts if part]

    # Otherwise split on whitespace and strip punctuation.
    parts = re.split(r"\s+", text.strip())
    terms: List[str] = []
    for part in parts:
        cleaned = re.sub(r"^[^\w]+|[^\w]+$", "", part)
        # Filter out stopwords (case-insensitive)
        if cleaned and cleaned.lower() not in all_stopwords:
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

    def _build_auth_headers(token: Optional[str]) -> Optional[Dict[str, str]]:
        """Build auth headers using token provider when configured."""
        headers: Optional[Dict[str, str]] = None
        if _token_provider:
            auth_token = _token_provider.get_token(token)
            if auth_token:
                headers = {"Authorization": auth_token}
        elif token:
            headers = {"Authorization": token}
        return headers

    async def _execute_structured_query(
        collection: str,
        filters: Optional[Dict[str, Any]],
        select: Optional[Any],
        sort: Optional[Any],
        cursorId: Optional[str],
        countOnly: bool,
        batchSize: Optional[int],
        num_results: Optional[int],
        result_format: str,
        token: Optional[str]
    ) -> Dict[str, Any]:
        """Execute a structured single-collection query."""
        options: Dict[str, Any] = {}
        select_fields = normalize_select(select)
        sort_expr = normalize_sort(sort)
        if select_fields:
            options["select"] = select_fields
        if sort_expr:
            options["sort"] = sort_expr

        # Validate filter fields against the collection's allowed fields.
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

        filter_str = build_filter(filters)
        if collection == "genome_feature":
            auto = "patric_id:*"
            if filter_str and filter_str != "*:*":
                filter_str = f"({filter_str}) AND {auto}"
            else:
                filter_str = auto

        if batchSize is not None and (batchSize < 1 or batchSize > 10000):
            return {
                "error": f"Invalid batchSize: {batchSize}. Must be between 1 and 10000.",
                "source": "bvbrc-mcp-data"
            }

        if num_results is not None:
            if num_results < 1:
                return {
                    "error": f"Invalid num_results: {num_results}. Must be >= 1.",
                    "source": "bvbrc-mcp-data"
                }
            if batchSize is None:
                batchSize = min(CURSOR_BATCH_SIZE, num_results)
            elif batchSize > num_results:
                batchSize = num_results

        headers = _build_auth_headers(token)
        if collection == "genome_sequence":
            if headers is None:
                headers = {}
            headers["http_accept"] = "application/dna+fasta"

        try:
            if num_results is not None and not countOnly and (cursorId is None or cursorId == "*"):
                all_results = []
                current_cursor = cursorId or "*"
                total_fetched = 0
                last_page_result = None

                while total_fetched < num_results:
                    remaining = num_results - total_fetched
                    page_batch_size = min(batchSize or CURSOR_BATCH_SIZE, remaining)
                    page_result = await query_direct(
                        collection,
                        filter_str,
                        options,
                        _base_url,
                        headers=headers,
                        cursorId=current_cursor,
                        countOnly=False,
                        batch_size=page_batch_size
                    )
                    last_page_result = page_result
                    page_results = page_result.get("results", [])
                    if not page_results:
                        break

                    needed = num_results - total_fetched
                    all_results.extend(page_results[:needed])
                    total_fetched += len(page_results[:needed])

                    next_cursor = page_result.get("nextCursorId")
                    if total_fetched >= num_results or not next_cursor:
                        break
                    current_cursor = next_cursor

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
                result = await query_direct(
                    collection,
                    filter_str,
                    options,
                    _base_url,
                    headers=headers,
                    cursorId=cursorId,
                    countOnly=countOnly,
                    batch_size=batchSize
                )

            if result_format == "tsv" and not countOnly:
                conversion_result = convert_json_to_tsv(result.get("results", []))
                if "error" in conversion_result:
                    return {
                        "error": conversion_result["error"],
                        "hint": conversion_result.get("hint", ""),
                        "results": result.get("results", []),
                        "count": result.get("count"),
                        "numFound": result.get("numFound"),
                        "nextCursorId": result.get("nextCursorId"),
                        "source": "bvbrc-mcp-data"
                    }
                result["tsv"] = conversion_result["tsv"]
                del result["results"]

            result["source"] = "bvbrc-mcp-data"
            return result
        except Exception as e:
            return {
                "error": f"Error querying {collection}: {str(e)}",
                "source": "bvbrc-mcp-data"
            }

    # @mcp.tool(annotations={"readOnlyHint": True})
    def bvbrc_collection_fields_and_parameters(collection: str) -> str:
        """
        Get fields and query parameters for a given BV-BRC collection.
        
        Args:
            collection: The collection name (e.g., "genome")
        
        Returns:
            String with the parameters for the given collection
        """
        return lookup_parameters(collection)

    # @mcp.tool(annotations={"readOnlyHint": True})
    def bvbrc_list_collections() -> str:
        """
        List all available BV-BRC collections.
        
        Returns:
            String with the available collections
        """
        print("Fetching available collections.")
        return list_solr_collections()

    @mcp.tool(annotations={"readOnlyHint": True})
    async def bvbrc_search_data(
        user_query: str,
        advanced: bool = False,
        search_mode: str = "phrase",
        format: str = "tsv",
        limit: Optional[int] = None,
        cursorId: Optional[str] = None,
        countOnly: bool = False,
        batchSize: Optional[int] = None,
        num_results: Optional[int] = None,
        token: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Unified natural-language data search tool for BV-BRC.

        Default behavior (`advanced=False`) is exploratory global search:
        - Select a likely collection with internal LLM routing
        - Execute a q= search using phrase/and/or semantics

        Advanced behavior (`advanced=True`) is targeted retrieval:
        - Plan a structured query from user_query
        - Execute the resulting single-collection query with validated fields
        - Supports countOnly, batchSize, and num_results controls

        Args:
            user_query: Natural language query text or keyword list.
            advanced: False (default) for global discovery; True for targeted structured query execution.
            search_mode: Global mode only. One of "phrase", "and", "or".
            format: Output format, "tsv" (default) or "json".
            limit: Optional max results. In global mode limits one page; in advanced mode maps to num_results when num_results is not provided.
            cursorId: Optional cursor for pagination. Use "*" or omit for first page.
            countOnly: Advanced mode only. Return only total count.
            batchSize: Advanced mode only. Rows per page (1-10000).
            num_results: Advanced mode only. Total results target across cursor pages.
            token: Authentication token (optional, auto-detected if token_provider is configured).
        """
        # FOR TESTING: Force advanced to False
        advanced = False
        # FOR TESTING: Force search_mode to 'and'
        search_mode = 'and'
        # FOR TESTING: Force TSV output
        format = "tsv"
        
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
        if not advanced:
            if countOnly or batchSize is not None or num_results is not None:
                return {
                    "error": "countOnly, batchSize, and num_results are only supported when advanced=true.",
                    "errorType": "INVALID_PARAMETERS",
                    "source": "bvbrc-mcp-data"
                }

            if _contains_solr_syntax(query_text):
                return {
                    "error": "user_query appears to contain Solr syntax, which is not allowed for this tool.",
                    "errorType": "INVALID_PARAMETERS",
                    "hint": "Provide natural language or plain keywords only. Use search_mode for phrase/and/or behavior.",
                    "source": "bvbrc-mcp-data"
                }

            try:
                search_info = _build_global_search_q_expr(query_text, search_mode)
                llm_client = _get_llm_client()
                selection = select_collection_for_query(query_text, llm_client)
                collection = str(selection.get("collection", "")).strip()
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

                headers = _build_auth_headers(token)
                page_size = min(limit, CURSOR_BATCH_SIZE) if limit is not None else CURSOR_BATCH_SIZE
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

                if format == "tsv":
                    conversion_result = convert_json_to_tsv(result.get("results", []))
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
                result["mode"] = "global"
                result["source"] = "bvbrc-mcp-data"
                return result
            except Exception as e:
                return {
                    "error": f"Global data search failed: {str(e)}",
                    "errorType": "SEARCH_FAILED",
                    "source": "bvbrc-mcp-data"
                }

        # advanced=True: plan + execute targeted query
        try:
            llm_client = _get_llm_client()
            planning_result = create_query_plan_internal(query_text, llm_client)
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
            collection = plan.get("collection")
            if not collection:
                return {
                    "error": "Planner did not return a target collection.",
                    "errorType": "PLANNING_FAILED",
                    "selection": planning_result.get("selection"),
                    "rawPlan": plan,
                    "source": "bvbrc-mcp-data"
                }

            effective_num_results = num_results if num_results is not None else limit
            result = await _execute_structured_query(
                collection=collection,
                filters=plan.get("filters"),
                select=plan.get("select"),
                sort=plan.get("sort"),
                cursorId=cursorId if cursorId is not None else plan.get("cursorId"),
                countOnly=countOnly if countOnly else bool(plan.get("countOnly", False)),
                batchSize=batchSize if batchSize is not None else plan.get("batchSize"),
                num_results=effective_num_results if effective_num_results is not None else plan.get("num_results"),
                result_format=format if format is not None else plan.get("format", "json"),
                token=token
            )

            if "error" in result:
                return result

            result["mode"] = "advanced"
            result["selection"] = planning_result.get("selection", {})
            result["plan"] = plan
            return result
        except FileNotFoundError as e:
            return {
                "error": f"Configuration or prompt file not found: {str(e)}",
                "errorType": "CONFIGURATION_ERROR",
                "hint": "Ensure config/config.json and planning prompt files exist",
                "source": "bvbrc-mcp-data"
            }
        except Exception as e:
            return {
                "error": f"Advanced data query failed: {str(e)}",
                "errorType": "SEARCH_FAILED",
                "source": "bvbrc-mcp-data"
            }

    # @mcp.tool(annotations={"readOnlyHint": True})
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

    # @mcp.tool(annotations={"readOnlyHint": True})
    async def bvbrc_get_genome_sequence_by_id(genome_ids: List[str],
                                             token: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the nucleotide sequences for complete genomes by their genome IDs.
        
        This tool queries the genome_sequence collection to retrieve full genomic sequences
        including chromosomes and plasmids. Each genome may have multiple sequences.
        
        Args:
            genome_ids: List of genome IDs (e.g., ["208964.12", "511145.12"])
                       Can be a single ID in a list or multiple IDs for batch retrieval
            token: Authentication token (optional, auto-detected if token_provider is configured)
            
        Returns:
            Dict with either:
            - Success: {
                "fasta": <str>,                   # FASTA formatted sequences with headers
                "count": <int>,                   # Number of sequences successfully retrieved
                "requested": <int>,               # Number of IDs requested
                "not_found": [<str>, ...],        # Optional: IDs that weren't found
                "warnings": [<str>, ...],         # Optional: Warning messages
                "source": "bvbrc-mcp-data"
              }
            - Error: {
                "error": <error_message>,
                "source": "bvbrc-mcp-data"
              }
        
        Example FASTA format:
            >NC_002516
            tttaaagagaccggcgattctagtgaaatcgaacgggcaggtc...
            >plasmid_01
            atcgatcgatcgatcg...
        
        Example:
            # Get genome sequence for a single genome
            result = bvbrc_get_genome_sequence_by_id(["208964.12"])
            
            # Get genome sequences for multiple genomes (batch query)
            result = bvbrc_get_genome_sequence_by_id([
                "208964.12",
                "511145.12",
                "83332.12"
            ])
        """
        print(f"Getting genome sequence(s) for {len(genome_ids)} genome(s)")
        
        # Authentication headers
        headers: Optional[Dict[str, str]] = None
        if _token_provider:
            auth_token = _token_provider.get_token(token)
            if auth_token:
                headers = {"Authorization": auth_token}
        elif token:
            headers = {"Authorization": token}
        
        try:
            result = await get_genome_sequence_by_id(
                genome_ids=genome_ids,
                base_url=_base_url,
                headers=headers
            )
            return result
        except Exception as e:
            return {
                "error": f"Error retrieving genome sequence: {str(e)}",
                "source": "bvbrc-mcp-data"
            }
