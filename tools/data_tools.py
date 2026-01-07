#!/usr/bin/env python3
"""
BV-BRC MVP Tools

This module contains MCP tools for querying MVP (Minimum Viable Product) data from BV-BRC.
"""

import json
from typing import Optional, Dict, Any, List

from fastmcp import FastMCP

# Global variables to store configuration
_base_url = None
_token_provider = None

from functions.data_functions import (
    query_direct,
    lookup_parameters,
    query_info,
    list_solr_collections,
    normalize_select,
    normalize_sort,
    build_filter,
    get_collection_fields,
    validate_filter_fields
)


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
    def bvbrc_query_collection(collection: str,
                               filters: Optional[Dict[str, Any]] = None,
                               select: Optional[Any] = None,
                               sort: Optional[Any] = None,
                               cursorId: Optional[str] = None,
                               countOnly: bool = False,
                               batchSize: Optional[int] = None,
                               token: Optional[str] = None) -> str:
        """
        Query BV-BRC data with structured filters; Solr syntax is handled for you.
        
        Args:
            collection: Collection name.
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
            cursorId: Cursor ID for pagination ("*" or omit for first page).
            countOnly: If True, only return the total count without data.
            batchSize: Number of rows to return per page (optional, defaults to 100, valid range: 1-10000).
            token: Authentication token (optional, auto-detected if token_provider is configured).
        """
        print(f"Querying collection: {collection}, count flag = {countOnly}.")
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
            return json.dumps({
                "error": f"Invalid field(s) for collection '{collection}': {', '.join(invalid_fields)}",
                "hint": "Call bvbrc_collection_fields_and_parameters to see valid fields.",
                "allowedFieldsSample": sample_fields,
                "source": "bvbrc-mcp-data"
            }, indent=2, sort_keys=True)

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
                return json.dumps({
                    "error": f"Invalid batchSize: {batchSize}. Must be between 1 and 10000.",
                    "source": "bvbrc-mcp-data"
                }, indent=2, sort_keys=True)

        # Authentication headers
        headers: Optional[Dict[str, str]] = None
        if _token_provider:
            auth_token = _token_provider.get_token(token)
            if auth_token:
                headers = {"Authorization": auth_token}
        elif token:
            headers = {"Authorization": token}
        
        print(f"Filter is {filter_str}")
        try:
            result = query_direct(collection, filter_str, options, _base_url, 
                                 headers=headers, cursorId=cursorId, countOnly=countOnly,
                                 batch_size=batchSize)
            # Prefer count for the returned page; fall back to numFound if needed
            observed_count = result.get("count", result.get("numFound"))
            print(f"Query returned {observed_count} results.")
            
            # Add 'source' field to the top-level response
            result['source'] = 'bvbrc-mcp-data'
            
            return json.dumps(result, indent=2, sort_keys=True)
        except Exception as e:
            return json.dumps({
                "error": f"Error querying {collection}: {str(e)}"
            }, indent=2)

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
