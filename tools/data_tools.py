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
    build_filter
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

    @mcp.tool(annotations={"readOnlyHint": True})
    def query_collection(collection: str,
                          filters: Optional[Dict[str, Any]] = None,
                          select: Optional[Any] = None,
                          sort: Optional[Any] = None,
                          cursorId: Optional[str] = None,
                          countOnly: bool = False,
                          token: Optional[str] = None) -> str:
        """
        Query BV-BRC data with structured filters; Solr syntax is handled for you.
        
        Args:
            collection: Collection name (e.g., "genome", "genome_feature").
            filters: Structured filter object describing conditions and grouping.
                Format:
                {
                  "logic": "and" | "or",   # optional, defaults to "and"
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
            select: List of fields or comma-separated string (optional).
            sort: Sort string (e.g., "genome_name asc") or list of
                { "field": "...", "dir": "asc|desc" } entries (optional).
            cursorId: Cursor ID for pagination ("*" or omit for first page).
            countOnly: If True, only return the total count without data.
            token: Authentication token (optional, auto-detected if token_provider is configured).

        Notes: Use `solr_collection_parameters` to discover available fields. The tool
            automatically applies collection-specific defaults (e.g., patric-only features).

        Returns:
            JSON string:
            - If countOnly is True: {"count": <total_count>, "source": "bvbrc-mcp-data"}
            - Otherwise: {"count": <batch_count>, "results": [...], "nextCursorId": <str|None>, "source": "bvbrc-mcp-data"}
        """

        print(f"Querying collection: {collection}, count flag = {countOnly}.")
        options: Dict[str, Any] = {}
        select_fields = normalize_select(select)
        sort_expr = normalize_sort(sort)
        if select_fields:
            options["select"] = select_fields
        if sort_expr:
            options["sort"] = sort_expr
        
        # Build Solr query from structured filters
        filter_str = build_filter(filters)

        # Apply collection-specific defaults
        if collection == "genome_feature":
            auto = "patric_id:*"
            if filter_str and filter_str != "*:*":
                filter_str = f"({filter_str}) AND {auto}"
            else:
                filter_str = auto

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
                                 headers=headers, cursorId=cursorId, countOnly=countOnly)
            print(f"Query returned {result['count']} results.")
            
            # Add 'source' field to the top-level response
            result['source'] = 'bvbrc-mcp-data'
            
            return json.dumps(result, indent=2, sort_keys=True)
        except Exception as e:
            return json.dumps({
                "error": f"Error querying {collection}: {str(e)}"
            }, indent=2)
    
    @mcp.tool(annotations={"readOnlyHint": True})
    def solr_collection_parameters(collection: str) -> str:
        """
        Get parameters for a given collection.
        
        Args:
            collection: The collection name (e.g., "genome")
        
        Returns:
            String with the parameters for the given collection
        """
        return lookup_parameters(collection)

    @mcp.tool(annotations={"readOnlyHint": True})
    def solr_query_instructions() -> str:
        """
        Get general query instructions for all collections.
        
        Returns:
            String with general query instructions and formatting guidelines
        """
        print("Fetching general query instructions.")
        return query_info()

    @mcp.tool(annotations={"readOnlyHint": True})
    def solr_collections() -> str:
        """
        Get all available collections.
        
        Returns:
            String with the available collections
        """
        print("Fetching available collections.")
        return list_solr_collections()

