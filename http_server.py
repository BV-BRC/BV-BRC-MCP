#!/usr/bin/env python3
"""
BVBRC Consolidated MCP Server

This server consolidates the data, service, and workspace MCP servers into a single unified server.
"""

from fastmcp import FastMCP
from common.json_rpc import JsonRpcCaller
from tools.data_tools import register_data_tools
from tools.service_tools import register_service_tools
from tools.workspace_tools import register_workspace_tools
from common.token_provider import TokenProvider
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse
import json
import sys
import os
from common.auth import BvbrcOAuthProvider

# Load configuration
with open("config/config.json", "r") as f:
    config = json.load(f)

# Get configuration values
base_url = config.get("base_url", "https://www.bv-brc.org/api-bulk")
workspace_api_url = config.get("workspace_url", "https://p3.theseed.org/services/Workspace")
service_api_url = config.get("service_api_url", "https://p3.theseed.org/services/app_service")
similar_genome_finder_api_url = config.get("similar_genome_finder_api_url", service_api_url)
authentication_url = config.get("authentication_url", "https://user.patricbrc.org/authenticate")
openid_config_url = config.get("openid_config_url", "https://dev-7.bv-brc.org")
port = int(os.environ.get("PORT", config.get("port", 12010)))
mcp_url = config.get("mcp_url", "127.0.0.1")

# Initialize token provider for HTTP mode
token_provider = TokenProvider(mode="http")

# Initialize the JSON-RPC callers
workspace_api = JsonRpcCaller(workspace_api_url)
service_api = JsonRpcCaller(service_api_url)
similar_genome_finder_api = JsonRpcCaller(similar_genome_finder_api_url)

# Publicly reachable server URL for discovery and metadata
# Use PUBLIC_BASE_URL env var if set, otherwise construct from openid_config_url
# The server URL is where this MCP server is accessible, not the data API URL
server_url = os.environ.get("PUBLIC_BASE_URL") or openid_config_url

# Initialize OAuth provider (class-based) with server URL (not data API base_url)
oauth = BvbrcOAuthProvider(
    base_url=server_url,  # This is the MCP server URL, not the data API URL
    openid_config_url=openid_config_url,
    authentication_url=authentication_url,
)

# Create FastMCP server with auth provider so /mcp is protected by FastMCP
mcp = FastMCP("BVBRC Consolidated MCP Server", auth=oauth)

# Register all tools from the three modules
print("Registering data tools...", file=sys.stderr)
register_data_tools(mcp, base_url, token_provider)

print("Registering service tools...", file=sys.stderr)
register_service_tools(mcp, service_api, similar_genome_finder_api, token_provider)

print("Registering workspace tools...", file=sys.stderr)
register_workspace_tools(mcp, workspace_api, token_provider)

# Add health check tool
# @mcp.tool()
def health_check() -> str:
    """Health check endpoint"""
    return '{"status": "healthy", "service": "bvbrc-consolidated-mcp"}'

# Add OAuth2 endpoints
@mcp.custom_route("/mcp/.well-known/openid-configuration", methods=["GET"])
async def openid_configuration_route(request) -> JSONResponse:
    """
    Serves the OIDC discovery document that ChatGPT expects.
    """
    return await oauth.openid_configuration(request)

# OAuth Authorization Server metadata (well-known)
# Per RFC 8414 section 3, if issuer is "https://example.com/issuer1",
# metadata is at "https://example.com/.well-known/oauth-authorization-server/issuer1"
# So for issuer {server_url}/mcp, metadata should be at /.well-known/oauth-authorization-server/mcp
@mcp.custom_route("/.well-known/oauth-authorization-server/mcp", methods=["GET"])
async def oauth_as_metadata(request) -> JSONResponse:
    issuer = f"{server_url}/mcp"
    return JSONResponse({
        "issuer": issuer,
        "authorization_endpoint": f"{issuer}/oauth2/authorize",
        "token_endpoint": f"{issuer}/oauth2/token",
        "registration_endpoint": f"{issuer}/oauth2/register",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code"],
        "token_endpoint_auth_methods_supported": ["none", "client_secret_post"],
        "code_challenge_methods_supported": ["S256"],
        "scopes_supported": ["profile", "token"],
    })

@mcp.custom_route("/mcp/oauth2/register", methods=["POST"])
async def oauth2_register_route(request) -> JSONResponse:
    """
    Registers a new client with the OAuth2 server.
    Implements RFC 7591 OAuth 2.0 Dynamic Client Registration.
    """
    return await oauth.oauth2_register(request)

@mcp.custom_route("/mcp/oauth2/authorize", methods=["GET"])
async def oauth2_authorize_route(request):
    """
    Authorization endpoint - displays login page for user authentication.
    This is where ChatGPT redirects the user to log in.
    """
    return await oauth.oauth2_authorize(request)

@mcp.custom_route("/mcp/oauth2/login", methods=["POST"])
async def oauth2_login_route(request):
    """
    Handles the login form submission.
    Authenticates the user and generates an authorization code.
    Redirects back to ChatGPT's callback URL with the code.
    """
    return await oauth.oauth2_login(request)

@mcp.custom_route("/mcp/oauth2/token", methods=["POST"])
async def oauth2_token_route(request):
    """
    Handles the token request.
    Exchanges an authorization code for an access token.
    Retrieves the stored user token using the authorization code.
    """
    return await oauth.oauth2_token(request)

def main() -> int:
    print(f"Starting BVBRC Consolidated MCP Server on port {port}...", file=sys.stderr)
    print(f"  - Data tools: {base_url}", file=sys.stderr)
    print(f"  - Service tools: {service_api_url}", file=sys.stderr)
    print(f"  - Workspace tools: {workspace_api_url}", file=sys.stderr)
    
    try:
        mcp.run(transport="http", host=mcp_url, port=port)
    except KeyboardInterrupt:
        print("Server stopped.", file=sys.stderr)
    except Exception as e:
        print(f"Server error: {e}", file=sys.stderr)
        return 1
    
    return 0

if __name__ == "__main__":
    main()

