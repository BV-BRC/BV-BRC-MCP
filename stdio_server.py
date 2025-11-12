#!/usr/bin/env python3
"""
BVBRC Consolidated MCP Server - STDIO Mode

This server consolidates the data, service, and workspace MCP servers into a single unified server
running in STDIO mode for use with MCP clients like Claude Desktop.
"""

from fastmcp import FastMCP
from common.json_rpc import JsonRpcCaller
from tools.data_tools import register_data_tools
from tools.service_tools import register_service_tools
from tools.workspace_tools import register_workspace_tools
from common.token_provider import TokenProvider
import json
import sys
import os

# Load configuration
try:
    with open("config/config.json", "r") as f:
        config = json.load(f)
except FileNotFoundError:
    print("Warning: config/config.json not found, using defaults", file=sys.stderr)
    config = {}

# Get configuration values
base_url = config.get("base_url", "https://www.bv-brc.org/api-bulk")
workspace_api_url = config.get("workspace_url", "https://p3.theseed.org/services/Workspace")
service_api_url = config.get("service_api_url", "https://p3.theseed.org/services/app_service")
similar_genome_finder_api_url = config.get("similar_genome_finder_api_url", service_api_url)

# Initialize token provider for STDIO mode
# In STDIO mode, token comes from KB_AUTH_TOKEN environment variable
token_provider = TokenProvider(mode="stdio")

# Initialize the JSON-RPC callers
workspace_api = JsonRpcCaller(workspace_api_url)
service_api = JsonRpcCaller(service_api_url)
similar_genome_finder_api = JsonRpcCaller(similar_genome_finder_api_url)

# Create FastMCP server
mcp = FastMCP("BVBRC Consolidated MCP Server")

# Register all tools from the three modules
print("Registering data tools...", file=sys.stderr)
register_data_tools(mcp, base_url, token_provider)

print("Registering service tools...", file=sys.stderr)
register_service_tools(mcp, service_api, similar_genome_finder_api, token_provider)

print("Registering workspace tools...", file=sys.stderr)
register_workspace_tools(mcp, workspace_api, token_provider)

# Add health check tool
@mcp.tool()
def health_check() -> str:
    """Health check endpoint"""
    return '{"status": "healthy", "service": "bvbrc-consolidated-mcp", "mode": "stdio"}'

def main() -> int:
    """Main entry point for the BVBRC Consolidated MCP Server in STDIO mode."""
    print("Starting BVBRC Consolidated MCP Server in STDIO mode...", file=sys.stderr)
    print(f"  - Data tools: {base_url}", file=sys.stderr)
    print(f"  - Service tools: {service_api_url}", file=sys.stderr)
    print(f"  - Workspace tools: {workspace_api_url}", file=sys.stderr)
    print(f"  - Authentication: KB_AUTH_TOKEN environment variable", file=sys.stderr)
    
    # Check if KB_AUTH_TOKEN is set
    if not os.getenv("KB_AUTH_TOKEN"):
        print("Warning: KB_AUTH_TOKEN environment variable is not set", file=sys.stderr)
        print("  Service and workspace tools will require a token parameter", file=sys.stderr)
    else:
        print("  KB_AUTH_TOKEN is set ✓", file=sys.stderr)
    
    try:
        # Run in stdio mode
        mcp.run(transport="stdio")
    except KeyboardInterrupt:
        print("Server stopped.", file=sys.stderr)
    except Exception as e:
        print(f"Server error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

