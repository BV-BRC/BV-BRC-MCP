
from fastmcp import FastMCP
from functions.workspace_functions import (
    workspace_get_file_metadata, workspace_download_file,
    workspace_upload as workspace_upload_func, workspace_create_genome_group,
    workspace_create_feature_group, workspace_get_genome_group_ids, workspace_get_feature_group_ids,
    workspace_preview_file, workspace_read_range, workspace_browse
)
from common.json_rpc import JsonRpcCaller
from common.token_provider import TokenProvider
import json
from typing import List, Optional
import sys
import os
import csv
import base64
from pymongo import MongoClient
from pymongo.errors import PyMongoError

_file_registry_client: Optional[MongoClient] = None

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
        # Strip /workspace prefix if present (orchestrator may add this)
        if path.startswith('/workspace/'):
            path = path[len('/workspace'):]
        
        # If path already starts with /, treat as absolute
        if path.startswith('/'):
            resolved_paths.append(path)
        elif path == 'home':
            # If path is just "home", return home_path directly to avoid /home/home
            resolved_paths.append(home_path)
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
    if not path or path == '/':
        return get_user_home_path(user_id)

    # Strip /workspace prefix if present (orchestrator may add this)
    if path.startswith('/workspace/'):
        path = path[len('/workspace'):]

    home_path = get_user_home_path(user_id)

    # If path already starts with /, treat as absolute
    if path.startswith('/'):
        return path
    elif path == 'home':
        # If path is just "home", return home_path directly to avoid /home/home
        return home_path
    else:
        # Treat as relative to home directory
        return f"{home_path}/{path}"

def _get_file_registry_client(file_utilities_config: dict) -> Optional[MongoClient]:
    """Create or reuse a MongoDB client for session file lookups."""
    global _file_registry_client
    mongo_url = (file_utilities_config or {}).get("mongo_url")
    if not mongo_url:
        return None
    if _file_registry_client is None:
        _file_registry_client = MongoClient(mongo_url, serverSelectionTimeoutMS=5000)
    return _file_registry_client

def _get_registered_file_path(session_id: str, file_id: str, file_utilities_config: dict) -> Optional[str]:
    """Resolve filePath from configured session_files registry when available."""
    client = _get_file_registry_client(file_utilities_config)
    if client is None:
        return None

    db_name = file_utilities_config.get("mongo_database", "copilot")
    collection_name = file_utilities_config.get("mongo_collection", "session_files")
    collection = client[db_name][collection_name]
    queries = [
        {"session_id": session_id, "fileId": file_id},
        {"sessionId": session_id, "fileId": file_id},
        {"session_id": session_id, "file_id": file_id},
    ]

    try:
        for query in queries:
            record = collection.find_one(query)
            if record and record.get("filePath"):
                return record["filePath"]
    except PyMongoError as exc:
        print(f"Mongo file lookup failed: {exc}", file=sys.stderr)
    return None

def _resolve_local_file_path(session_id: str, file_id: str, file_utilities_config: Optional[dict]) -> str:
    """Resolve local session file path using configured file_utilities settings."""
    config = file_utilities_config or {}
    base_path = config.get("session_base_path")
    if not base_path:
        raise ValueError("file_utilities.session_base_path must be configured")

    registered_path = _get_registered_file_path(session_id, file_id, config)
    if registered_path and os.path.exists(registered_path):
        return registered_path

    downloads_path = os.path.join(base_path, session_id, "downloads")

    extensions = ["", ".json", ".csv", ".tsv", ".txt"]
    for ext in extensions:
        candidate = os.path.join(downloads_path, f"{file_id}{ext}")
        if os.path.exists(candidate):
            return candidate

    return os.path.join(downloads_path, file_id)

def _detect_local_file_type(file_path: str) -> str:
    """Detect local file type for line-oriented reads."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".json":
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return "json_array"
            if isinstance(data, dict):
                return "json_object"
        except Exception:
            pass
    if ext == ".csv":
        return "csv"
    if ext == ".tsv":
        return "tsv"
    return "text"

def _read_local_file_lines(file_path: str, start: int, end: Optional[int], limit: int) -> dict:
    """Read local file lines using Copilot tool semantics."""
    limit = min(limit, 10000)
    file_type = _detect_local_file_type(file_path)

    if file_type in ("json_array", "json_object"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if file_type == "json_object":
            data = [{"key": k, "value": v} for k, v in data.items()]
        total_lines = len(data)
        start_idx = max(0, start - 1)
        end_idx = min(end if end else total_lines, total_lines)
        end_idx = min(start_idx + limit, end_idx)
        return {
            "lines": data[start_idx:end_idx],
            "startLine": start_idx + 1,
            "endLine": end_idx,
            "totalLines": total_lines,
            "hasMore": end_idx < total_lines,
            "source": "bvbrc-workspace"
        }

    if file_type in ("csv", "tsv"):
        delimiter = "\t" if file_type == "tsv" else ","
        with open(file_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter=delimiter))
        total_lines = len(rows)
        start_idx = max(0, start - 1)
        end_idx = min(end if end else total_lines, total_lines)
        end_idx = min(start_idx + limit, end_idx)
        return {
            "lines": rows[start_idx:end_idx],
            "startLine": start_idx + 1,
            "endLine": end_idx,
            "totalLines": total_lines,
            "hasMore": end_idx < total_lines,
            "source": "bvbrc-workspace"
        }

    with open(file_path, "r", encoding="utf-8") as f:
        text_lines = f.readlines()
    total_lines = len(text_lines)
    start_idx = max(0, start - 1)
    end_idx = min(end if end else total_lines, total_lines)
    end_idx = min(start_idx + limit, end_idx)
    return {
        "lines": [line.rstrip("\n") for line in text_lines[start_idx:end_idx]],
        "startLine": start_idx + 1,
        "endLine": end_idx,
        "totalLines": total_lines,
        "hasMore": end_idx < total_lines,
        "source": "bvbrc-workspace"
    }

def register_workspace_tools(
    mcp: FastMCP,
    api: JsonRpcCaller,
    token_provider: TokenProvider,
    file_utilities_config: Optional[dict] = None
):
    """Register workspace tools with the FastMCP server"""
    
    @mcp.tool()
    async def workspace_browse_tool(
        token: Optional[str] = None,
        path: Optional[str] = None,
        search: bool = False,
        filename_search_terms: Optional[List[str]] = None,
        file_extension: Optional[List[str]] = None,
        file_types: Optional[List[str]] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
        num_results: int = 50
    ) -> dict:
        """Browse the workspace using a single unified interface.

        Args:
            token: Authentication token (optional - will use default if not provided)
            path: Path to inspect/search. IMPORTANT PATH FORMAT:
                - Absolute paths MUST start with /{user_id}/home (e.g., /user1@patricbrc.org/home or /user1@patricbrc.org/home/subfolder)
                - Relative paths (e.g., "subfolder") are resolved relative to /{user_id}/home
                - DO NOT include /workspace prefix - paths should be /{user_id}/home or relative paths
                - If path is not provided or empty, defaults to user's home directory
                - Examples: "/user1@patricbrc.org/home", "subfolder", "/user1@patricbrc.org/home/Genome Groups"
            search: CRITICAL - Controls search behavior:
                - search=True: RECURSIVE search through all subdirectories. Use this when:
                    * Finding files across the entire workspace (e.g., "find all fastq files", "10 most recent files")
                    * Searching by filename, extension, or type across multiple folders
                    * You need to look beyond just the direct contents of one folder
                - search=False: NON-RECURSIVE inspection of a single path. Use this when:
                    * Listing direct contents of a specific folder only (one level)
                    * Getting metadata of a single file or folder
                    * You only want to see what's immediately in the target directory
                EXAMPLES:
                    * "Find all fastq files in my workspace" → search=True
                    * "Show me the 10 most recent files" → search=True
                    * "List contents of my home directory" → search=False
                    * "What's in the Genome Groups folder?" → search=False
            filename_search_terms: Words/terms that must appear IN the filename itself (AND logic). 
                                   Use this ONLY to filter by text within the actual filename.
            file_extension: File extensions to match (OR logic). Example: ["fastq", "fq"] finds .fastq OR .fq files.
            file_types: Workspace object types to match (OR logic). Valid types include: "reads", "contigs", "genome_group", 
                       "feature_group", "folder", "unspecified". Example: ["reads", "contigs"] finds reads OR contigs objects.
                       Note: This is the BVBRC workspace metadata type, NOT filename text or extensions.
            sort_by: Optional sort field. Valid options: creation_time, name, size, type.
            sort_order: Optional sort direction. Valid options: asc, desc.
            num_results: Maximum number of results to return. Defaults to 50.
        """
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return {
                "error": "No authentication token available",
                "errorType": "AUTHENTICATION_ERROR",
                "source": "bvbrc-workspace"
            }

        user_id = extract_userid_from_token(auth_token)
        resolved_path = resolve_relative_path(path, user_id)

        print(
            f"Browsing workspace path: {resolved_path}, user_id: {user_id}, "
            f"search: {search}, filename_search_terms: {filename_search_terms}, extension: {file_extension}, "
            f"file_types: {file_types}, sort_by: {sort_by}, sort_order: {sort_order}, num_results: {num_results}",
            file=sys.stderr
        )
        return await workspace_browse(
            api=api,
            token=auth_token,
            path=resolved_path,
            search=search,
            filename_search_terms=filename_search_terms,
            file_extension=file_extension,
            file_types=file_types,
            sort_by=sort_by,
            sort_order=sort_order,
            num_results=num_results,
            tool_name="workspace_browse_tool"
        )

    @mcp.tool()
    async def workspace_get_file_metadata_tool(token: Optional[str] = None, path: str = None) -> dict:
        """Get the metadata of a file from the workspace.

        Args:
            token: Authentication token (optional - will use default if not provided)
            path: Path to the file to get (relative to user's home directory).
        """
        # Get the appropriate token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return {
                "error": "No authentication token available",
                "errorType": "AUTHENTICATION_ERROR",
                "source": "bvbrc-workspace"
            }

        # Extract user_id from token for path resolution and logging
        user_id = extract_userid_from_token(auth_token)
        resolved_path = resolve_relative_path(path, user_id)

        print(f"Getting metadata for path: {resolved_path}, user_id: {user_id}")

        result = await workspace_get_file_metadata(api, resolved_path, auth_token)
        return result

    @mcp.tool()
    async def workspace_download_file_tool(token: Optional[str] = None, path: str = None, output_file: Optional[str] = None, return_data: bool = False) -> dict:
        """Download a file from the workspace.

        Args:
            token: Authentication token (optional - will use default if not provided)
            path: Path to the file to download (relative to user's home directory).
            output_file: Optional name and path of the file to save the downloaded content to.
                        If not provided and return_data is False, file data will be returned directly.
            return_data: If True, return the file data directly (base64 encoded for binary files, text for text files).
                        If False and output_file is provided, only write to file. If False and output_file is None,
                        returns file data (default behavior).
        """
        # Get the appropriate token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return {
                "error": "No authentication token available",
                "errorType": "AUTHENTICATION_ERROR",
                "source": "bvbrc-workspace"
            }

        # Extract user_id from token for path resolution and logging
        user_id = extract_userid_from_token(auth_token)
        resolved_path = resolve_relative_path(path, user_id)

        print(f"Downloading file from path: {resolved_path}, user_id: {user_id}, output_file: {output_file}, return_data: {return_data}")

        result = await workspace_download_file(api, resolved_path, auth_token, output_file, return_data)
        return result

    @mcp.tool(annotations={"readOnlyHint": True})
    async def workspace_preview_file_tool(token: Optional[str] = None, path: str = None) -> dict:
        """Preview a file from the workspace by downloading only the first portion.
        
        This tool downloads a preview of the file (first portion) without downloading the entire file.
        Useful for quickly viewing the beginning of large files.

        Args:
            token: Authentication token (optional - will use default if not provided)
            path: Path to the file to preview (relative to user's home directory).
        """
        # Get the appropriate token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return {
                "error": "No authentication token available",
                "errorType": "AUTHENTICATION_ERROR",
                "source": "bvbrc-workspace"
            }

        # Extract user_id from token for path resolution and logging
        user_id = extract_userid_from_token(auth_token)
        resolved_path = resolve_relative_path(path, user_id)

        print(f"Previewing file from path: {resolved_path}, user_id: {user_id}")

        result = await workspace_preview_file(api, resolved_path, auth_token)
        return result

    @mcp.tool(annotations={"readOnlyHint": True})
    async def workspace_read_range_tool(
        token: Optional[str] = None,
        path: str = None,
        start_byte: int = 0,
        max_bytes: int = 8192
    ) -> dict:
        """Read a byte range from a workspace file.

        Use this tool to page through large files safely by changing start_byte.

        Args:
            token: Authentication token (optional - will use default if not provided)
            path: Path to the file to read (relative to user's home directory).
            start_byte: Zero-based starting byte offset (default: 0).
            max_bytes: Maximum bytes to read (default: 8192, max: 1048576).
        """
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return {
                "error": "No authentication token available",
                "errorType": "AUTHENTICATION_ERROR",
                "source": "bvbrc-workspace"
            }

        user_id = extract_userid_from_token(auth_token)
        resolved_path = resolve_relative_path(path, user_id)

        print(
            f"Reading workspace file range from path: {resolved_path}, user_id: {user_id}, "
            f"start_byte: {start_byte}, max_bytes: {max_bytes}"
        )

        return await workspace_read_range(
            api=api,
            path=resolved_path,
            token=auth_token,
            start_byte=start_byte,
            max_bytes=max_bytes
        )

    @mcp.tool(annotations={"readOnlyHint": True})
    def preview_file(
        session_id: str,
        file_id: str,
        start_byte: int = 0,
        max_bytes: int = 8192
    ) -> dict:
        """Preview a byte range from a local Copilot session file."""
        print(f"Getting preview of local file {file_id} in session {session_id}...", file=sys.stderr)
        try:
            if not session_id or not file_id:
                return {
                    "error": True,
                    "errorType": "INVALID_PARAMETERS",
                    "message": "Missing required parameters: session_id and file_id",
                    "source": "bvbrc-workspace"
                }

            if start_byte < 0:
                return {
                    "error": True,
                    "errorType": "INVALID_PARAMETERS",
                    "message": "start_byte must be >= 0",
                    "source": "bvbrc-workspace"
                }

            if max_bytes <= 0:
                return {
                    "error": True,
                    "errorType": "INVALID_PARAMETERS",
                    "message": "max_bytes must be > 0",
                    "source": "bvbrc-workspace"
                }

            max_bytes = min(max_bytes, 1024 * 1024)
            file_path = _resolve_local_file_path(session_id, file_id, file_utilities_config)

            with open(file_path, "rb") as f:
                f.seek(start_byte)
                chunk = f.read(max_bytes)

            try:
                return {"content": chunk.decode("utf-8"), "source": "bvbrc-workspace"}
            except UnicodeDecodeError:
                base64_content = base64.b64encode(chunk).decode("utf-8")
                return {
                    "content": f"<base64_encoded_data>{base64_content}</base64_encoded_data>",
                    "source": "bvbrc-workspace"
                }
        except Exception as e:
            return {
                "error": True,
                "errorType": "PROCESSING_ERROR",
                "message": f"Error getting file preview: {str(e)}",
                "source": "bvbrc-workspace"
            }

    @mcp.tool(annotations={"readOnlyHint": True})
    def read_file_lines(
        session_id: str,
        file_id: str,
        start: int = 1,
        end: Optional[int] = None,
        limit: int = 1000
    ) -> dict:
        """Read line ranges from a local Copilot session file."""
        print(
            f"Reading lines from local file {file_id} in session {session_id}: start={start}, end={end}, limit={limit}",
            file=sys.stderr
        )
        try:
            if not session_id or not file_id:
                return {
                    "error": True,
                    "errorType": "INVALID_PARAMETERS",
                    "message": "Missing required parameters: session_id and file_id",
                    "source": "bvbrc-workspace"
                }

            file_path = _resolve_local_file_path(session_id, file_id, file_utilities_config)
            if not os.path.exists(file_path):
                return {
                    "error": True,
                    "errorType": "FILE_NOT_FOUND",
                    "message": f"File {file_id} not found",
                    "details": {"fileId": file_id, "session_id": session_id},
                    "source": "bvbrc-workspace"
                }

            return _read_local_file_lines(file_path, start, end, limit)
        except Exception as e:
            return {
                "error": True,
                "errorType": "PROCESSING_ERROR",
                "message": f"Error reading file lines: {str(e)}",
                "source": "bvbrc-workspace"
            }

    @mcp.tool()
    async def workspace_upload(token: Optional[str] = None, filename: str = None, upload_dir: str = None) -> dict:
        """Create an upload URL for a file in the workspace.

        Args:
            token: Authentication token (optional - will use default if not provided)
            filename: Name of the file to create upload URL for.
            upload_dir: Directory to upload the file to (relative to user's home directory, defaults to user's home directory).
        """
        if not filename:
            return {
                "error": "filename parameter is required",
                "errorType": "INVALID_PARAMETERS",
                "source": "bvbrc-workspace"
            }

        # Get the appropriate token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return {
                "error": "No authentication token available",
                "errorType": "AUTHENTICATION_ERROR",
                "source": "bvbrc-workspace"
            }

        # Extract user_id from token for path resolution and logging
        user_id = extract_userid_from_token(auth_token)
        if not upload_dir:
            upload_dir = get_user_home_path(user_id)
        else:
            # If upload_dir is provided and doesn't start with /, treat as relative to home
            if not upload_dir.startswith('/') and user_id:
                upload_dir = f"{get_user_home_path(user_id)}/{upload_dir}"

        print(f"Uploading file: {filename}, user_id: {user_id}, upload_dir: {upload_dir}")

        result = await workspace_upload_func(api, filename, upload_dir, auth_token)
        return result

    @mcp.tool()
    async def create_genome_group(token: Optional[str] = None, genome_group_name: str = None, genome_id_list: str = None, genome_group_path: str = None) -> dict:
        """Create a genome group in the workspace.

        Args:
            token: Authentication token (optional - will use default if not provided)
            genome_group_name: Name of the genome group to create (used if genome_group_path not provided).
            genome_id_list: List of genome IDs to add to the genome group. Accepts multiple genome ids as a string with comma separation. Example: genome_id1,genome_id2,genome_id3,...
            genome_group_path: Full path for the genome group. If not provided, defaults to /<user_id>/home/<genome_group_name>.
        """
        if not genome_group_name:
            return {
                "error": "genome_group_name parameter is required",
                "errorType": "INVALID_PARAMETERS",
                "source": "bvbrc-workspace"
            }

        if not genome_id_list:
            return {
                "error": "genome_id_list parameter is required",
                "errorType": "INVALID_PARAMETERS",
                "source": "bvbrc-workspace"
            }

        # Get the appropriate token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return {
                "error": "No authentication token available",
                "errorType": "AUTHENTICATION_ERROR",
                "source": "bvbrc-workspace"
            }

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
        print("genome_id_list (raw)", repr(genome_id_list), file=sys.stderr)
        print("genome_id_list type", type(genome_id_list), file=sys.stderr)

        # Convert comma-separated string to list
        if isinstance(genome_id_list, str):
            genome_id_list_parsed = [gid.strip() for gid in genome_id_list.split(',') if gid.strip()]
        elif isinstance(genome_id_list, list):
            genome_id_list_parsed = [str(gid).strip() for gid in genome_id_list if gid]
        else:
            return {
                "error": f"genome_id_list must be a string or list, got {type(genome_id_list)}",
                "errorType": "INVALID_PARAMETERS",
                "source": "bvbrc-workspace"
            }
        
        print("genome_id_list_parsed", genome_id_list_parsed, file=sys.stderr)
        print("genome_id_list_parsed length", len(genome_id_list_parsed), file=sys.stderr)
        result = await workspace_create_genome_group(api, genome_group_path, genome_id_list_parsed, auth_token)
        return result

    @mcp.tool()
    async def create_feature_group(token: Optional[str] = None, feature_group_name: str = None, feature_id_list: str = None, feature_group_path: str = None) -> dict:
        """Create a feature group in the workspace.

        Args:
            token: Authentication token (optional - will use default if not provided)
            feature_group_name: Name of the feature group to create (used if feature_group_path not provided).
            feature_id_list: List of feature IDs as a string with comma separation to add to the feature group. Example: feature_id1,feature_id2,feature_id3,...
            feature_group_path: Full path for the feature group. If not provided, defaults to /<user_id>/home/<feature_group_name>.
        """
        if not feature_group_name and not feature_group_path:
            return {
                "error": "feature_group_name or feature_group_path parameter is required",
                "errorType": "INVALID_PARAMETERS",
                "source": "bvbrc-workspace"
            }

        if feature_group_name and feature_group_path:
            return {
                "error": "only one of feature_group_name or feature_group_path parameter can be provided",
                "errorType": "INVALID_PARAMETERS",
                "source": "bvbrc-workspace"
            }

        if not feature_id_list:
            return {
                "error": "feature_id_list parameter is required",
                "errorType": "INVALID_PARAMETERS",
                "source": "bvbrc-workspace"
            }

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
            return {
                "error": "No authentication token available",
                "errorType": "AUTHENTICATION_ERROR",
                "source": "bvbrc-workspace"
            }

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

        result = await workspace_create_feature_group(api, feature_group_path, feature_id_list, auth_token)
        return result

    @mcp.tool()
    async def get_genome_group_ids(token: Optional[str] = None, genome_group_name: str = None, genome_group_path: str = None) -> dict:
        """Get genome_ids from a specific genome group. Requires either genome_group_name or genome_group_path to identify the group.

        Args:
            token: Authentication token (optional - will use default if not provided)
            genome_group_name: Name of the genome group to get genome_ids from.
            genome_group_path: Full path for the genome group. If not provided, defaults to /<user_id>/home/Genome Groups/<genome_group_name>.

            Only one of genome_group_name or genome_group_path parameter can be provided.

        Returns:
            List of genome_ids in the specified genome group.
        """
        if not genome_group_name and not genome_group_path:
            return {
                "error": "genome_group_name or genome_group_path parameter is required",
                "errorType": "INVALID_PARAMETERS",
                "source": "bvbrc-workspace"
            }
        
        if genome_group_name and genome_group_path:
            return {
                "error": "only one of genome_group_name or genome_group_path parameter can be provided",
                "errorType": "INVALID_PARAMETERS",
                "source": "bvbrc-workspace"
            }

        # Get the appropriate token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return {
                "error": "No authentication token available",
                "errorType": "AUTHENTICATION_ERROR",
                "source": "bvbrc-workspace"
            }

        # Extract user_id from token for path resolution
        user_id = extract_userid_from_token(auth_token)
        if not genome_group_path:
            genome_group_path = f"{get_user_home_path(user_id)}/Genome Groups/{genome_group_name}"
        else:
            # If genome_group_path is provided and doesn't start with /, treat as relative to home
            if not genome_group_path.startswith('/') and user_id:
                genome_group_path = f"{get_user_home_path(user_id)}/{genome_group_path}"

        print(f"Getting genome group IDs: {genome_group_name}, user_id: {user_id}, path: {genome_group_path}")

        result = await workspace_get_genome_group_ids(api, genome_group_path, auth_token)
        return result

    @mcp.tool()
    async def get_feature_group_ids(token: Optional[str] = None, feature_group_name: str = None, feature_group_path: str = None) -> dict:
        """Get feature_ids from a specific feature group. Requires either feature_group_name or feature_group_path to identify the group.

        Args:
            token: Authentication token (optional - will use default if not provided)
            feature_group_name: Name of the feature group to get feature_ids from.
            feature_group_path: Full path for the feature group. If not provided, defaults to /<user_id>/home/Feature Groups/<feature_group_name>.
            
            Only one of feature_group_name or feature_group_path parameter can be provided.

        Returns:
            List of feature_ids in the specified feature group.
        """
        if not feature_group_name and not feature_group_path:
            return {
                "error": "feature_group_name or feature_group_path parameter is required",
                "errorType": "INVALID_PARAMETERS",
                "source": "bvbrc-workspace"
            }

        if feature_group_name and feature_group_path:
            return {
                "error": "only one of feature_group_name or feature_group_path parameter can be provided",
                "errorType": "INVALID_PARAMETERS",
                "source": "bvbrc-workspace"
            }

        # Get the appropriate token
        auth_token = token_provider.get_token(token)
        if not auth_token:
            return {
                "error": "No authentication token available",
                "errorType": "AUTHENTICATION_ERROR",
                "source": "bvbrc-workspace"
            }

        # Extract user_id from token for path resolution
        user_id = extract_userid_from_token(auth_token)
        if not feature_group_path:
            feature_group_path = f"{get_user_home_path(user_id)}/Feature Groups/{feature_group_name}"
        else:
            # If feature_group_path is provided and doesn't start with /, treat as relative to home
            if not feature_group_path.startswith('/') and user_id:
                feature_group_path = f"{get_user_home_path(user_id)}/{feature_group_path}"

        print(f"Getting feature group IDs: {feature_group_name}, user_id: {user_id}, path: {feature_group_path}")

        result = await workspace_get_feature_group_ids(api, feature_group_path, auth_token)
        return result