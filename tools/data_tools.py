#!/usr/bin/env python3
"""
BV-BRC MVP Tools

This module contains MCP tools for querying MVP (Minimum Viable Product) data from BV-BRC.
"""

import json
import re
from typing import Optional

from fastmcp import FastMCP

# Global variables to store configuration
_base_url = None

from functions.data_functions import (
    query_direct,
    lookup_parameters,
    query_info,
    list_solr_collections
)


def register_data_tools(mcp: FastMCP, base_url: str):
    """Register all MVP-related MCP tools with the FastMCP server."""
    global _base_url
    _base_url = base_url

    @mcp.tool(annotations={"readOnlyHint": True})
    def query_collection(collection: str, filter_str: str = "",
                          select: Optional[str] = None, sort: Optional[str] = None) -> str:
        """
        Query BV-BRC data directly using collection name and Solr query expression.
        
        Args:
            collection: The collection name (e.g., "genome", "genome_feature")
            filter_str: Solr query expression (e.g., "genome_id:123.45" or "species:\"Escherichia coli\"")
            select: Comma-separated list of fields to select (optional)
            sort: Field to sort by (optional)

        Notes: Information on genome resistance to antibiotics is in the genome_amr table. Information on
            special feature properties like Antibiotic Resistance, Virulence Factor, and Essential Gene is in the
            sp_gene table. To find which features are in a subsystem, use the subsystem_ref table. Use the
            genome_name field to search for an organism by name. The solr_collection_parameters tool lists
            all the field names for each collection. This tool should be checked to avoid Bad Request errors.

        Returns:
            Formatted query results
        """
        print(f"Querying collection: {collection}")
        options = {}
        if select:
            options["select"] = select.split(",")
        if sort:
            options["sort"] = sort
        # If we have a genome_feature query, we need to insure only patric features come back.
        if not filter_str:
            filter_str = "patric_id:*"
        elif collection == "genome_feature" and not re.search(r"\bpatric_id:", filter_str):
            filter_str += " AND patric_id:*"
        print(f"Filter is {filter_str}")

        try:
            result, count = query_direct(collection, filter_str, options, _base_url)
            print(f"Query returned {len(result)} of {count} results.")
            return json.dumps({
                "count": count,
                "results": result
            }, indent=2)
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

