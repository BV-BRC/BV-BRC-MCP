#!/usr/bin/env python3
"""
BV-BRC MVP Tools

This module contains MCP tools for querying MVP (Minimum Viable Product) data from BV-BRC.
"""

import json
import re
from typing import Optional, Dict

from fastmcp import FastMCP

# Global variables to store configuration
_base_url = None
_token_provider = None

from functions.data_functions import (
    query_direct,
    lookup_parameters,
    query_info,
    list_solr_collections
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
    def query_collection(collection: str, filter_str: str = "",
                          select: Optional[str] = None, sort: Optional[str] = None,
                          cursorId: Optional[str] = None, countOnly: bool = False,
                          token: Optional[str] = None) -> str:
        """
        Query BV-BRC data directly using collection name and Solr query expression.
        
        Args:
            collection: The collection name (e.g., "genome", "genome_feature")
            filter_str: Solr query expression (e.g., "genome_id:123.45" or "species:\"Escherichia coli\"")
            select: Comma-separated list of fields to select (optional)
            sort: Field to sort by (optional)
            cursorId: Cursor ID for pagination (optional, use "*" or omit for first page)
            countOnly: If True, only return the total count without data (optional, default False)
            token: Authentication token for API access (optional, will be auto-detected if token_provider is configured)

        Notes: Information on genome resistance to antibiotics is in the genome_amr table. Information on
            special feature properties like Antibiotic Resistance, Virulence Factor, and Essential Gene is in the
            sp_gene table. To find which features are in a subsystem, use the subsystem_ref table. Use the
            genome_name field to search for an organism by name. Note that antibiotic names are case-sensitive
            and stored in all lower case (e.g. "methicillin"). Thus, to ask for all Bacillus subtilis resistant
            to ampicillin, you would use the filter string

            resistant_phenotype:Resistant AND genome_name:"Bacillus subtilis" AND antibiotic:ampicillin

            In the filter string, any field value with spaces in it must be enclosed in double quotes (e.g. "field value").
            
            The solr_collection_parameters tool lists all the field names for each collection. This tool should
            be checked to avoid Bad Request errors.

            When countOnly is True, use the minimum number of fields in the select parameter to reduce the number of fields returned.

            To find a DNA or protein sequence for a genome feature, you need to call this method twice, first using the
            genome_feature table to get the aa_sequence_md5 or na_sequence_md5 for the feature. Then you can use the
            appropriate MD5 value to query the feature_sequence table using the md5 field as the key, and extract the
            sequence from the sequence field.

            In the genome_feature table, whenever possible, the patric_id should be used instead of the feature_id, as
            only the patric_id is made visible to the end user. Patric IDs always begin with "fig|".

        Returns:
            JSON string with query results:
            - If countOnly is True: {"count": <total_count>}
            - Otherwise: {"count": <batch_count>, "results": [...], "nextCursorId": <str|None>}
        """
        print(f"Querying collection: {collection}, count flag = {countOnly}.")
        options = {}
        if select:
            options["select"] = select.split(",")
        if sort:
            options["sort"] = sort
        
        # Get authentication token and build headers
        headers: Optional[Dict[str, str]] = None
        if _token_provider:
            auth_token = _token_provider.get_token(token)
            if auth_token:
                headers = {"Authorization": auth_token}
        elif token:
            # Fallback: if token is provided directly and no token_provider, use it
            headers = {"Authorization": token}

        # If we have a genome_feature query, we need to insure only patric features come back.
        if collection == "genome_feature" and not filter_str:
            filter_str = "patric_id:*"
        elif collection == "genome_feature" and not re.search(r"\bpatric_id:", filter_str):
            filter_str += " AND patric_id:*"
        # For all queries, we have to make sure the field values with spaces are quoted.
        filter_list = re.split(r'\s+AND\s+', filter_str)
        for i, f in enumerate(filter_list):
            match = re.match(r'(\S+):["(](.*)[)"]', f)
            if not match:
                match = re.match(r'(\S+):(.+)', f)
                if match:
                    field, value = match.groups()
                    if ' ' in value:
                        filter_list[i] = f'{field}:"{value}"'
        filter_str = ' AND '.join(filter_list)
        print(f"Filter is {filter_str}")
        
        try:
            result = query_direct(collection, filter_str, options, _base_url, 
                                 headers=headers, cursorId=cursorId, countOnly=countOnly)
            print(f"Query returned {result['count']} results.")
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

