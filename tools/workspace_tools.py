
from fastmcp import FastMCP
from functions.workspace_functions import (
    workspace_ls, workspace_get_file_metadata, workspace_download_file,
    workspace_upload, workspace_search, workspace_create_genome_group,
    workspace_create_feature_group, workspace_get_genome_group_ids, workspace_get_feature_group_ids
)
from common.json_rpc import JsonRpcCaller
from common.token_provider import TokenProvider
import json
from typing import List, Optional

def extract_userid_from_token(token: str = None) -> str:
    """
    Extract user ID from JWT token.
    Returns a default user ID if token is None or invalid.
    """
    if not token:
        return None

    try:
        user_id = token.split('|')[0].replace('un=','')
        return user_id

    except Exception as e:
        print(f"Error extracting user ID from token: {e}")
        return None

def get_user_home_path(user_id: str) -> str:
    """
    Get the user's home path in the workspace.

    Args:
        user_id: User ID extracted from token

    Returns:
        User's home path in format /{user_id}/home
    """
    if not user_id:
        return "/"
    return f"/{user_id}/home"

def resolve_relative_paths(paths: List[str], user_id: str) -> List[str]:
    """
    Convert relative paths to absolute paths by prepending user's home directory.

    Args:
        paths: List of relative paths
        user_id: User ID extracted from token

    Returns:
        List of absolute paths
    """
    if not paths:
        return [get_user_home_path(user_id)]

    home_path = get_user_home_path(user_id)
    resolved_paths = []

    for path in paths:
        # If path already starts with /, treat as absolute
        if path.startswith('/'):
            resolved_paths.append(path)
        else:
            # Treat as relative to home directory
            resolved_paths.append(f"{home_path}/{path}")

    return resolved_paths

def resolve_relative_path(path: str, user_id: str) -> str:
    """
    Convert a relative path to an absolute path by prepending user's home directory.

    Args:
        path: Relative path
        user_id: User ID extracted from token

    Returns:
        Absolute path
    """
    if not path:
        return get_user_home_path(user_id)

    home_path = get_user_home_path(user_id)

    # If path already starts with /, treat as absolute
    if path.startswith('/'):
        return path
    else:
        # Treat as relative to home directory
        return f"{home_path}/{path}"

def register_workspace_tools(mcp: FastMCP, api: JsonRpcCaller, token_provider: TokenProvider):
    """Register workspace tools with the FastMCP server"""
    
    @mcp.tool()
    def workspace_ls_tool(token: Optional[str] = None, paths: List[str] = None) -> str:
        """List the contents of the workspace.

        Args:
            token: Authentication token (optional - will use default if not provided)
            paths: Optional list of paths to list (relative to user's home directory). If empty or None, lists user home directory.

        Returns:
            String representation of workspace contents.
        """
        # Get the appropriate token (automatically checks Authorization header in HTTP mode)
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return "Error: No authentication token available"

        # Extract user_id from token for path resolution
        user_id = extract_userid_from_token(auth_token)
        paths = resolve_relative_paths(paths or [], user_id)

        print(f"Listing paths: {paths}, user_id: {user_id}")
        result = workspace_ls(api, paths, auth_token)
        return str(result)

    @mcp.tool()
    def workspace_search_tool(token: Optional[str] = None, search_term: str = None, paths: List[str] = None) -> str:
        """Search the workspace for a given term.

        Args:
            token: Authentication token (optional - will use default if not provided)
            search_term: Term to search the workspace for.
            paths: Optional list of paths to search (relative to user's home directory). If empty or None, searches user home directory.
        """
        if not search_term:
            return "Error: search_term parameter is required"

        # Get the appropriate token (automatically checks Authorization header in HTTP mode)
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return "Error: No authentication token available"

        # Extract user_id from token for path resolution
        user_id = extract_userid_from_token(auth_token)
        paths = resolve_relative_paths(paths or [], user_id)

        print(f"Searching in paths: {paths}, user_id: {user_id}, term: {search_term}")
        result = workspace_search(api, paths, search_term, auth_token)
        return str(result)

    @mcp.tool()
    def workspace_get_file_metadata_tool(token: Optional[str] = None, path: str = None) -> str:
        """Get the metadata of a file from the workspace.

        Args:
            token: Authentication token (optional - will use default if not provided)
            path: Path to the file to get (relative to user's home directory).
        """
        # Get the appropriate token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return "Error: No authentication token available"

        # Extract user_id from token for path resolution and logging
        user_id = extract_userid_from_token(auth_token)
        resolved_path = resolve_relative_path(path, user_id)

        print(f"Getting metadata for path: {resolved_path}, user_id: {user_id}")

        result = workspace_get_file_metadata(api, resolved_path, auth_token)
        return str(result)

    @mcp.tool()
    def workspace_download_file_tool(token: Optional[str] = None, path: str = None, output_file: str = None) -> str:
        """Download a file from the workspace.

        Args:
            token: Authentication token (optional - will use default if not provided)
            path: Path to the file to download (relative to user's home directory).
            output_file: Name and path of the file to save the downloaded content to.
        """
        # Get the appropriate token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return "Error: No authentication token available"

        # Extract user_id from token for path resolution and logging
        user_id = extract_userid_from_token(auth_token)
        resolved_path = resolve_relative_path(path, user_id)

        print(f"Downloading file from path: {resolved_path}, user_id: {user_id}")

        result = workspace_download_file(api, resolved_path, auth_token, output_file)
        return str(result)

    @mcp.tool()
    def workspace_upload(token: Optional[str] = None, filename: str = None, upload_dir: str = None) -> str:
        """Create an upload URL for a file in the workspace.

        Args:
            token: Authentication token (optional - will use default if not provided)
            filename: Name of the file to create upload URL for.
            upload_dir: Directory to upload the file to (relative to user's home directory, defaults to user's home directory).
        """
        if not filename:
            return "Error: filename parameter is required"

        # Get the appropriate token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return "Error: No authentication token available"

        # Extract user_id from token for path resolution and logging
        user_id = extract_userid_from_token(auth_token)
        if not upload_dir:
            upload_dir = get_user_home_path(user_id)
        else:
            # If upload_dir is provided and doesn't start with /, treat as relative to home
            if not upload_dir.startswith('/') and user_id:
                upload_dir = f"{get_user_home_path(user_id)}/{upload_dir}"

        print(f"Uploading file: {filename}, user_id: {user_id}, upload_dir: {upload_dir}")

        result = workspace_upload(api, filename, upload_dir, auth_token)
        return str(result)

    @mcp.tool()
    def create_genome_group(token: Optional[str] = None, genome_group_name: str = None, genome_id_list: str = None, genome_group_path: str = None) -> str:
        """Create a genome group in the workspace.

        Args:
            token: Authentication token (optional - will use default if not provided)
            genome_group_name: Name of the genome group to create (used if genome_group_path not provided).
            genome_id_list: List of genome IDs to add to the genome group. Accepts multiple genome ids as a string with comma separation. Example: genome_id1,genome_id2,genome_id3,...
            genome_group_path: Full path for the genome group. If not provided, defaults to /<user_id>/home/<genome_group_name>.
        """
        if not genome_group_name:
            return "Error: genome_group_name parameter is required"

        if not genome_id_list:
            return "Error: genome_id_list parameter is required"

        # Get the appropriate token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return "Error: No authentication token available"

        # Extract user_id from token for path resolution
        user_id = extract_userid_from_token(auth_token)
        if not genome_group_path:
            # Create path from name - treat as relative to home directory
            genome_group_path = f"{get_user_home_path(user_id)}/Genome Groups/{genome_group_name}"
        else:
            # If genome_group_path is provided and doesn't start with /, treat as relative to home
            if not genome_group_path.startswith('/') and user_id:
                genome_group_path = f"{get_user_home_path(user_id)}/{genome_group_path}"

        print(f"Creating genome group: {genome_group_name}, user_id: {user_id}, path: {genome_group_path}")

        result = workspace_create_genome_group(api, genome_group_path, genome_id_list, auth_token)
        return str(result)

    @mcp.tool()
    def create_feature_group(token: Optional[str] = None, feature_group_name: str = None, feature_id_list: str = None, feature_group_path: str = None) -> str:
        """Create a feature group in the workspace.

        Args:
            token: Authentication token (optional - will use default if not provided)
            feature_group_name: Name of the feature group to create (used if feature_group_path not provided).
            feature_id_list: List of feature IDs as a string with comma separation to add to the feature group. Example: feature_id1,feature_id2,feature_id3,...
            feature_group_path: Full path for the feature group. If not provided, defaults to /<user_id>/home/<feature_group_name>.
        """
        if not feature_group_name and not feature_group_path:
            return "Error: feature_group_name or feature_group_path parameter is required"

        if feature_group_name and feature_group_path:
            return "Error: only one of feature_group_name or feature_group_path parameter can be provided"

        if not feature_id_list:
            return "Error: feature_id_list parameter is required"

        # The LLM is consistently forgetting the final '.' in the feature IDs
        feature_id_list = feature_id_list.split(',')
        processed_feature_id_list = []
        for feature_id in feature_id_list:
            feature_id = feature_id.strip()
            if len(feature_id) >= 4 and feature_id[-4] != '.':
                # Insert '.' at fourth-to-last position
                feature_id = feature_id[:-3] + '.' + feature_id[-3:]
            processed_feature_id_list.append(feature_id)
        feature_id_list = processed_feature_id_list

        # TODO: include a feature id verification step to ensure the feature IDs are valid

        # Get the appropriate token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return "Error: No authentication token available"

        # Extract user_id from token for path resolution
        user_id = extract_userid_from_token(auth_token)
        if not feature_group_path:
            # Create path from name - treat as relative to home directory
            feature_group_path = f"{get_user_home_path(user_id)}/Feature Groups/{feature_group_name}"
        else:
            # If genome_group_path is provided and doesn't start with /, treat as relative to home
            if not feature_group_path.startswith('/') and user_id:
                feature_group_path = f"{get_user_home_path(user_id)}/{feature_group_path}"

        print(f"Creating feature group: {feature_group_name}, user_id: {user_id}, path: {feature_group_path}")

        result = workspace_create_feature_group(api, feature_group_path, feature_id_list, auth_token)
        return json.dumps(result)

    @mcp.tool()
    def get_genome_group_ids(token: Optional[str] = None, genome_group_name: str = None, genome_group_path: str = None) -> List[str]:
        """Get the IDs of the genomes in a genome group.

        Args:
            token: Authentication token (optional - will use default if not provided)
            genome_group_name: Name of the genome group to get the IDs of.
            genome_group_path: Full path for the genome group. If not provided, defaults to /<user_id>/home/Genome Groups/<genome_group_name>.

            Only one of genome_group_name or genome_group_path parameter can be provided.

        Returns:
            List of genome IDs in the genome group.
        """
        if not genome_group_name and not genome_group_path:
            return "Error: genome_group_name or genome_group_path parameter is required"
        
        if genome_group_name and genome_group_path:
            return "Error: only one of genome_group_name or genome_group_path parameter can be provided"

        # Get the appropriate token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return "Error: No authentication token available"

        # Extract user_id from token for path resolution
        user_id = extract_userid_from_token(auth_token)
        if not genome_group_path:
            genome_group_path = f"{get_user_home_path(user_id)}/Genome Groups/{genome_group_name}"
        else:
            # If genome_group_path is provided and doesn't start with /, treat as relative to home
            if not genome_group_path.startswith('/') and user_id:
                genome_group_path = f"{get_user_home_path(user_id)}/{genome_group_path}"

        print(f"Getting genome group IDs: {genome_group_name}, user_id: {user_id}, path: {genome_group_path}")

        result = workspace_get_genome_group_ids(api, genome_group_path, auth_token)
        return result

    @mcp.tool()
    def get_feature_group_ids(token: Optional[str] = None, feature_group_name: str = None, feature_group_path: str = None) -> List[str]:
        """Get the IDs of the features in a feature group.

        Args:
            token: Authentication token (optional - will use default if not provided)
            feature_group_name: Name of the feature group to get the IDs of.
            feature_group_path: Full path for the feature group. If not provided, defaults to /<user_id>/home/Feature Groups/<feature_group_name>.
        """
        if not feature_group_name and not feature_group_path:
            return "Error: feature_group_name or feature_group_path parameter is required"

        if feature_group_name and feature_group_path:
            return "Error: only one of feature_group_name or feature_group_path parameter can be provided"

        # Get the appropriate token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return "Error: No authentication token available"

        # Extract user_id from token for path resolution
        user_id = extract_userid_from_token(auth_token)
        if not feature_group_path:
            feature_group_path = f"{get_user_home_path(user_id)}/Feature Groups/{feature_group_name}"
        else:
            # If feature_group_path is provided and doesn't start with /, treat as relative to home
            if not feature_group_path.startswith('/') and user_id:
                feature_group_path = f"{get_user_home_path(user_id)}/{feature_group_path}"

        print(f"Getting feature group IDs: {feature_group_name}, user_id: {user_id}, path: {feature_group_path}")

        result = workspace_get_feature_group_ids(api, feature_group_path, auth_token)
        return result