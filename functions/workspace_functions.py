from common.json_rpc import JsonRpcCaller
from typing import List, Any
import asyncio
import httpx
import requests
import os
import json
import sys
import base64

def _build_grid_payload(
    entity_type: str,
    items: list = None,
    result_type: str = "list_result",
    pagination: dict = None,
    sort: dict = None,
    source: str = "bvbrc-workspace",
    columns: list = None,
    selectable: bool = True,
    multi_select: bool = True,
    sortable: bool = True
) -> dict:
    return {
        "schema_version": "1.0",
        "entity_type": entity_type,
        "source": source,
        "result_type": result_type,
        "capabilities": {
            "selectable": selectable,
            "multi_select": multi_select,
            "sortable": sortable
        },
        "pagination": pagination,
        "sort": sort,
        "columns": columns or [],
        "items": items or []
    }

def _fix_duplicated_user_id_in_path(path: str, user_id: str) -> str:
    """
    Detect and fix paths with duplicated user_id (e.g., /user@domain.com/user@domain.com/home -> /user@domain.com/home).
    
    Args:
        path: Path that may have duplicated user_id
        user_id: User ID to check for duplication
    
    Returns:
        Path with duplicated user_id removed
    """
    if not path or not user_id:
        return path
    
    # Check if path starts with /{user_id}/{user_id}/ (duplicated user_id)
    duplicated_prefix = f"/{user_id}/{user_id}/"
    if path.startswith(duplicated_prefix):
        # Remove the duplicate user_id segment
        fixed_path = f"/{user_id}/" + path[len(duplicated_prefix):]
        print(f"Fixed duplicated user_id in path: {path} -> {fixed_path}", file=sys.stderr)
        return fixed_path
    
    return path

def _fix_duplicated_user_id_in_paths(paths: List[str], user_id: str) -> List[str]:
    """
    Fix duplicated user_id in a list of paths.
    
    Args:
        paths: List of paths that may have duplicated user_id
        user_id: User ID to check for duplication
    
    Returns:
        List of paths with duplicated user_id removed
    """
    if not paths or not user_id:
        return paths
    
    return [_fix_duplicated_user_id_in_path(path, user_id) for path in paths]

async def workspace_ls(
    api: JsonRpcCaller,
    paths: List[str],
    token: str,
    file_types: List[str] = None,
    sort_by: str = None,
    sort_order: str = None,
    limit: int = None
) -> dict:
    """
    List the contents of a specific workspace directory using the JSON-RPC API.
    This is NOT a generic search function, use `workspace_search` for search functionality.

    Args:
        api: JsonRpcCaller instance configured with workspace URL and token
        paths: List of paths to list
        token: Authentication token for API calls
        file_types: Optional list of file types to filter by (e.g., ['contigs', 'folder', 'unspecified']).
                    If provided, only files/objects with these types will be returned.
        sort_by: Optional sort field. Valid options: creation_time, name, size, type.
        sort_order: Optional sort direction. Valid options: asc, desc.
        limit: Optional maximum number of results to return (server-side limiting).
    Returns:
        List of workspace items
    """
    try:
        # Fix any duplicated user_id in paths (defensive measure)
        user_id = _get_user_id_from_token(token)
        if user_id:
            paths = _fix_duplicated_user_id_in_paths(paths, user_id)
        
        # Build API call parameters
        # Enable recursive search when file_types is provided to search subdirectories
        if file_types:
            # file_types is already a list
            file_types_list = file_types if file_types else ['unspecified']

            # Use recursive search format matching workspace_search when filtering by type
            # Pass type as array directly (API expects array format for type filtering)
            api_params = {
                "recursive": True,
                "includeSubDirs": True,
                "excludeDirectories": False,
                "excludeObjects": False,
                "paths": paths,
                "query": {
                    "type": file_types_list
                }
            }
        else:
            # Non-recursive listing when no file_type filter
            api_params = {
                "Recursive": False,
                "includeSubDirs": False,
                "paths": paths
            }
        # Pass backend-supported sort options through directly when provided
        if sort_by:
            api_params["sort_by"] = sort_by
        if sort_order:
            api_params["sort_order"] = sort_order
        if limit is not None:
            api_params["limit"] = limit

        # Only include file_types_list in print if it was defined
        print_msg = f"workspace_ls file_types: {file_types}, Query params: {json.dumps(api_params, indent=2)}"
        if file_types:
            print_msg = f"workspace_ls file_types: {file_types}, file_types_list: {file_types_list}, Query params: {json.dumps(api_params, indent=2)}"
        print(print_msg, file=sys.stderr)
        result = await api.acall("Workspace.ls", api_params, 1, token)

        # Standardize response structure
        result_list = result if isinstance(result, list) else [result]
        return {
            "items": result_list,
            "count": len(result_list),
            "path": paths[0] if paths else "/",
            "source": "bvbrc-workspace",
            "ui_grid": _build_grid_payload(
                entity_type="workspace_item",
                items=result_list,
                result_type="list_result",
                source="bvbrc-workspace",
                sort={"sort_by": sort_by, "sort_order": sort_order},
                pagination={"limit": limit, "offset": 0, "has_more": None},
                columns=[
                    {"key": "name", "label": "Name", "sortable": True},
                    {"key": "type", "label": "Type", "sortable": True},
                    {"key": "creation_time", "label": "Created", "sortable": True},
                    {"key": "size", "label": "Size", "sortable": True}
                ],
                selectable=True,
                multi_select=True,
                sortable=True
            )
        }
    except Exception as e:
        return {
            "error": f"Error listing workspace: {str(e)}",
            "errorType": "API_ERROR",
            "source": "bvbrc-workspace"
        }

async def workspace_search(
    api: JsonRpcCaller,
    paths: List[str] = None,
    filename_search_terms: List[str] = None,
    file_extension: List[str] = None,
    file_types: List[str] = None,
    token: str = None,
    sort_by: str = None,
    sort_order: str = None,
    limit: int = None
) -> dict:
    """
    Search the entire workspace for a given term and/or file extension and/or file type.

    Args:
        api: JsonRpcCaller instance configured with workspace URL and token
        paths: List of paths to search
        filename_search_terms: Optional list of terms to search for within file/object names. All terms must appear in the name (AND logic).
                               Example: ["genome", "bacteria"] will match files containing both words in their name.
        file_extension: Optional list of file extensions to filter by (e.g., ['py', 'txt', 'json']).
                       Can include or exclude the leading dot. Multiple extensions use OR logic.
        file_types: Optional list of file types to filter by (e.g., ['contigs', 'folder', 'unspecified']).
                   If provided, only files/objects with these types will be returned. This filters by the workspace object type,
                   not by file extension, so it can match files with different extensions that share the same type.
        token: Authentication token for API calls
        sort_by: Optional sort field. Valid options: creation_time, name, size, type.
        sort_order: Optional sort direction. Valid options: asc, desc.
        limit: Optional maximum number of results to return (server-side limiting).
    Returns:
        List of matching workspace items
    """
    if not paths:
        user_id = _get_user_id_from_token(token)
        if not user_id:
            return {
                "error": "Unable to derive user id from token",
                "errorType": "INVALID_PARAMETERS",
                "source": "bvbrc-workspace"
            }
        paths = [f"/{user_id}/home"]
    else:
        # Fix any duplicated user_id in paths (defensive measure)
        user_id = _get_user_id_from_token(token)
        if user_id:
            paths = _fix_duplicated_user_id_in_paths(paths, user_id)

    # Build query conditions based on what's provided
    query_conditions = {}
    name_conditions = []
    type_conditions = []

    # Add filename search term condition(s) if provided
    if filename_search_terms:
        # filename_search_terms is already a list
        name_conditions.extend(filename_search_terms)

    # Add file extension filter(s) if provided
    if file_extension:
        # file_extension is already a list
        for ext in file_extension:
            # Normalize extension: remove leading dot if present, add it back for regex
            ext = ext.lstrip('.')
            # Create regex pattern that matches files ending with the extension
            # This ensures we match the extension at the end of the filename
            ext_pattern = f"\\.{ext}$"
            name_conditions.append(ext_pattern)

    # Add file type filter if provided
    if file_types:
        # file_types is already a list
        # Add type filter condition
        if len(file_types) == 1:
            type_conditions.append({
                "type": file_types[0]
            })
        else:
            type_conditions.append({
                "type": {"$in": file_types}
            })

    # Build final query conditions
    # NOTE: The workspace API does not support $and operator despite using MongoDB syntax.
    # Instead, we combine multiple name-based regex patterns into a single regex using lookahead.
    if len(name_conditions) == 0 and len(type_conditions) == 0:
        # No explicit filters: perform recursive search under provided paths
        query_conditions = None
    elif len(name_conditions) > 0 and len(type_conditions) == 0:
        # Only name-based filters
        if len(name_conditions) == 1:
            # Single name condition
            query_conditions = {
                "name": {
                    "$regex": name_conditions[0],
                    "$options": "i"
                }
            }
        else:
            # Multiple name conditions: combine using regex lookahead
            # Pattern: (?=.*pattern1)(?=.*pattern2)....*
            # For extension patterns ending with $, we need special handling
            combined_pattern = ""
            end_patterns = []

            for pattern in name_conditions:
                if pattern.endswith("$"):
                    # Extension pattern - save for end (OR logic for multiple extensions)
                    end_patterns.append(pattern)
                else:
                    # Search term - add as lookahead (AND logic for multiple terms)
                    combined_pattern += f"(?=.*{pattern})"

            if end_patterns:
                # Combine lookaheads with the end-anchored pattern(s)
                if len(end_patterns) > 1:
                    # Multiple extensions: OR them together
                    ext_or_pattern = "(" + "|".join(end_patterns) + ")"
                    combined_pattern += ".*" + ext_or_pattern
                else:
                    # Single extension
                    combined_pattern += ".*" + end_patterns[0]
            else:
                # No end anchor, just match anything after lookaheads
                combined_pattern += ".*"

            query_conditions = {
                "name": {
                    "$regex": combined_pattern,
                    "$options": "i"
                }
            }
    elif len(name_conditions) == 0 and len(type_conditions) > 0:
        # Only type-based filters
        query_conditions = type_conditions[0]
    else:
        # Both name and type conditions
        # Since we can't use $and, we need to handle this differently
        # We'll apply name filter in the query and filter by type client-side after
        # Actually, let's try putting both in a dict - MongoDB should support multiple fields
        if len(name_conditions) == 1:
            name_regex = name_conditions[0]
        else:
            # Combine name conditions
            combined_pattern = ""
            end_patterns = []

            for pattern in name_conditions:
                if pattern.endswith("$"):
                    # Extension pattern - save for end (OR logic for multiple extensions)
                    end_patterns.append(pattern)
                else:
                    # Search term - add as lookahead (AND logic for multiple terms)
                    combined_pattern += f"(?=.*{pattern})"

            if end_patterns:
                # Combine lookaheads with the end-anchored pattern(s)
                if len(end_patterns) > 1:
                    # Multiple extensions: OR them together
                    ext_or_pattern = "(" + "|".join(end_patterns) + ")"
                    combined_pattern += ".*" + ext_or_pattern
                else:
                    # Single extension
                    combined_pattern += ".*" + end_patterns[0]
            else:
                # No end anchor, just match anything after lookaheads
                combined_pattern += ".*"

            name_regex = combined_pattern

        # Try combining name and type in a single dict (implicit AND in MongoDB)
        query_conditions = {
            "name": {
                "$regex": name_regex,
                "$options": "i"
            }
        }
        # Add type condition to the same dict
        query_conditions.update(type_conditions[0])

    try:
        api_params = {
            "recursive": True,
            "excludeDirectories": False,
            "excludeObjects": False,
            "includeSubDirs": True,
            "paths": paths
        }
        if query_conditions is not None:
            api_params["query"] = query_conditions
        if sort_by:
            api_params["sort_by"] = sort_by
        if sort_order:
            api_params["sort_order"] = sort_order
        if limit is not None:
            api_params["limit"] = limit

        result = await api.acall("Workspace.ls", api_params, 1, token)

        # Standardize response structure
        result_list = result if isinstance(result, list) else [result]
        return {
            "items": result_list,
            "count": len(result_list),
            "filename_search_terms": filename_search_terms,
            "paths": paths,
            "source": "bvbrc-workspace",
            "ui_grid": _build_grid_payload(
                entity_type="workspace_item",
                items=result_list,
                result_type="search_result",
                source="bvbrc-workspace",
                sort={"sort_by": sort_by, "sort_order": sort_order},
                pagination={"limit": limit, "offset": 0, "has_more": None},
                columns=[
                    {"key": "name", "label": "Name", "sortable": True},
                    {"key": "type", "label": "Type", "sortable": True},
                    {"key": "creation_time", "label": "Created", "sortable": True},
                    {"key": "size", "label": "Size", "sortable": True}
                ],
                selectable=True,
                multi_select=True,
                sortable=True
            )
        }
    except Exception as e:
        return {
            "error": f"Error searching workspace: {str(e)}",
            "errorType": "API_ERROR",
            "source": "bvbrc-workspace"
        }

async def workspace_browse(
    api: JsonRpcCaller,
    token: str,
    path: str = None,
    search: bool = False,
    filename_search_terms: List[str] = None,
    file_extension: List[str] = None,
    file_types: List[str] = None,
    sort_by: str = None,
    sort_order: str = None,
    num_results: int = 50,
    tool_name: str = "workspace_browse_tool"
) -> dict:
    """
    Unified workspace browser entrypoint.

    - If search is True: perform recursive search under path (or user home by default).
    - If search is False: inspect path and return folder listing or file/object metadata.

    Args:
        api: JsonRpcCaller instance configured with workspace URL and token
        token: Authentication token for API calls
        path: Path to inspect/search. Defaults to user home if not provided.
        search: If True, perform recursive search. If False, inspect path and return listing or metadata.
        filename_search_terms: Optional list of terms to search within file/object names (used when search=True).
                               All terms must appear in the filename (AND logic).
        file_extension: Optional list of extension filters (used when search=True). Multiple extensions use OR logic.
        file_types: Optional list of workspace type filters (used in search and folder listing modes).
        sort_by: Optional sort field. Valid options: creation_time, name, size, type.
        sort_order: Optional sort direction. Valid options: asc, desc.
        num_results: Maximum number of results to return. Defaults to 50.
        tool_name: Name of the calling tool (for response envelope).

    Returns a consistent response envelope with all data nested under "result":
      {
        "result": {
          "items": <array> | "metadata": <dict>,  // items for search/list, metadata for file/object
          "tool_name": <tool name>,
          "result_type": "search_result" | "list_result" | "metadata_result",
          "count": <number>,  // present for search_result and list_result
          "path": <workspace path>,
          "source": "bvbrc-workspace"
        }
      }
    """
    if not token:
        return {
            "error": "Authentication token not provided",
            "errorType": "INVALID_PARAMETERS",
            "source": "bvbrc-workspace"
        }

    if not path or path == '/':
        user_id = _get_user_id_from_token(token)
        if not user_id:
            return {
                "error": "Unable to derive user id from token",
                "errorType": "INVALID_PARAMETERS",
                "source": "bvbrc-workspace"
            }
        path = f"/{user_id}/home"

    if search:
        search_result = await workspace_search(
            api=api,
            paths=[path],
            filename_search_terms=filename_search_terms,
            file_extension=file_extension,
            file_types=file_types,
            token=token,
            sort_by=sort_by,
            sort_order=sort_order,
            limit=num_results  # Pass limit to server-side
        )
        if "error" in search_result:
            return search_result

        items = search_result.get("items", [])
        # Server-side limiting via limit parameter, no client-side slicing needed
        return {
            "result": {
                "items": items,
                "tool_name": tool_name,
                "result_type": "search_result",
                "count": len(items),
                "path": path,
                "source": "bvbrc-workspace",
                "ui_grid": _build_grid_payload(
                    entity_type="workspace_item",
                    items=items,
                    result_type="search_result",
                    source="bvbrc-workspace",
                    sort={"sort_by": sort_by, "sort_order": sort_order},
                    pagination={"limit": num_results, "offset": 0, "has_more": None},
                    columns=[
                        {"key": "name", "label": "Name", "sortable": True},
                        {"key": "type", "label": "Type", "sortable": True},
                        {"key": "creation_time", "label": "Created", "sortable": True},
                        {"key": "size", "label": "Size", "sortable": True}
                    ]
                )
            },
            "call": {
                "tool": tool_name,
                "backend_method": "Workspace.ls",
                "arguments_executed": {
                    "path": path,
                    "search": True,
                    "filename_search_terms": filename_search_terms,
                    "file_extension": file_extension,
                    "file_types": file_types,
                    "sort_by": sort_by,
                    "sort_order": sort_order,
                    "num_results": num_results
                },
                "replayable": True
            }
        }

    metadata_result = await workspace_get_object(api, path, metadata_only=True, token=token)
    if "error" in metadata_result:
        return metadata_result

    metadata = metadata_result.get("metadata")
    if not metadata or not isinstance(metadata, dict):
        return {
            "error": "Invalid metadata response",
            "errorType": "INVALID_RESPONSE",
            "source": "bvbrc-workspace"
        }

    auto_meta = metadata.get("autoMeta") if isinstance(metadata.get("autoMeta"), dict) else {}
    is_folder = metadata.get("type") == "folder" or auto_meta.get("is_folder") == 1

    if is_folder:
        list_result = await workspace_ls(
            api=api,
            paths=[path],
            token=token,
            file_types=file_types,
            sort_by=sort_by,
            sort_order=sort_order,
            limit=num_results  # Pass limit to server-side
        )
        if "error" in list_result:
            return list_result

        items = list_result.get("items", [])
        # Server-side limiting via limit parameter, no client-side slicing needed
        return {
            "result": {
                "items": items,
                "tool_name": tool_name,
                "result_type": "list_result",
                "count": len(items),
                "path": path,
                "source": "bvbrc-workspace",
                "ui_grid": _build_grid_payload(
                    entity_type="workspace_item",
                    items=items,
                    result_type="list_result",
                    source="bvbrc-workspace",
                    sort={"sort_by": sort_by, "sort_order": sort_order},
                    pagination={"limit": num_results, "offset": 0, "has_more": None},
                    columns=[
                        {"key": "name", "label": "Name", "sortable": True},
                        {"key": "type", "label": "Type", "sortable": True},
                        {"key": "creation_time", "label": "Created", "sortable": True},
                        {"key": "size", "label": "Size", "sortable": True}
                    ]
                )
            },
            "call": {
                "tool": tool_name,
                "backend_method": "Workspace.ls",
                "arguments_executed": {
                    "path": path,
                    "search": False,
                    "file_types": file_types,
                    "sort_by": sort_by,
                    "sort_order": sort_order,
                    "num_results": num_results
                },
                "replayable": True
            }
        }

    return {
        "result": {
            "metadata": metadata,
            "tool_name": tool_name,
            "result_type": "metadata_result",
            "path": path,
            "source": "bvbrc-workspace",
            "ui_grid": _build_grid_payload(
                entity_type="workspace_metadata",
                items=[metadata],
                result_type="metadata_result",
                source="bvbrc-workspace",
                columns=[
                    {"key": "name", "label": "Name", "sortable": False},
                    {"key": "type", "label": "Type", "sortable": False},
                    {"key": "creation_time", "label": "Created", "sortable": False},
                    {"key": "size", "label": "Size", "sortable": False}
                ],
                selectable=False,
                multi_select=False,
                sortable=False
            )
        },
        "call": {
            "tool": tool_name,
            "backend_method": "Workspace.get",
            "arguments_executed": {
                "path": path,
                "search": False
            },
            "replayable": True
        }
    }

async def workspace_get_file_metadata(api: JsonRpcCaller, path: str, token: str) -> dict:
    """
    Get the metadata of a file from the workspace using the JSON-RPC API.

    Args:
        api: JsonRpcCaller instance configured with workspace URL and token
        path: Path to the file to get the metadata of
        token: Authentication token for API calls
    Returns:
        String representation of the file metadata
    """
    try:
        result = await api.acall("Workspace.get", {
            "objects": [path],
            "metadata_only": True
        },1, token)

        # Add source field to response
        if isinstance(result, dict):
            result["source"] = "bvbrc-workspace"
            return result
        else:
            return {
                "data": result,
                "source": "bvbrc-workspace"
            }
    except Exception as e:
        return {
            "error": f"Error getting file metadata: {str(e)}",
            "errorType": "API_ERROR",
            "source": "bvbrc-workspace"
        }


async def workspace_download_file(api: JsonRpcCaller, path: str, token: str, output_file: str = None, return_data: bool = False) -> dict:
    """
    Download a file from the workspace using the JSON-RPC API.

    Args:
        api: JsonRpcCaller instance configured with workspace URL and token
        path: Path to the file to download
        token: Authentication token for API calls
        output_file: Optional name and path of the file to save the downloaded content to.
        return_data: If True, return the file data directly (base64 encoded for binary files, text for text files).
                    If False and output_file is provided, only write to file. If False and output_file is None,
                    returns file data (default behavior for backward compatibility).
    Returns:
        If return_data is True or output_file is None: Returns file data (base64 encoded for binary, text for text files).
        If output_file is provided and return_data is False: Returns success message.
        If both output_file and return_data are True: Returns file data along with success message.
    """
    try:
        download_url_obj = await _get_download_url(api, path, token)
        download_url = download_url_obj[0][0]

        headers = {
            "Authorization": token
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(download_url, headers=headers)
            response.raise_for_status()
            content = response.content

            result_parts = []

            # Write to file if output_file is provided
            if output_file:
                with open(output_file, 'wb') as file:
                    file.write(content)
                result_parts.append(f"File downloaded and saved to {output_file}")

            # Return data if return_data is True, or if output_file is None (backward compatibility)
            if return_data or output_file is None:
                # Try to decode as text first
                try:
                    text_content = content.decode('utf-8')
                    result_parts.append(text_content)
                except UnicodeDecodeError:
                    # If it's binary, encode as base64
                    base64_content = base64.b64encode(content).decode('utf-8')
                    result_parts.append(f"<base64_encoded_data>{base64_content}</base64_encoded_data>")

            # Return appropriate result
            if len(result_parts) == 1:
                return {
                    "data": result_parts[0],
                    "source": "bvbrc-workspace"
                }
            elif len(result_parts) == 2:
                # Both file write and data return
                return {
                    "message": result_parts[0],
                    "data": result_parts[1],
                    "source": "bvbrc-workspace"
                }
            else:
                return {
                    "data": content,
                    "source": "bvbrc-workspace"
                }
    except Exception as e:
        return {
            "error": f"Error downloading file: {str(e)}",
            "errorType": "API_ERROR",
            "source": "bvbrc-workspace"
        }

async def workspace_preview_file(api: JsonRpcCaller, path: str, token: str) -> dict:
    """
    Preview a file from the workspace by downloading only the first portion using byte ranges.
    This function uses HTTP Range headers internally to download only a portion of the file.

    Args:
        api: JsonRpcCaller instance configured with workspace URL and token
        path: Path to the file to preview
        token: Authentication token for API calls
    Returns:
        Dictionary containing the preview data (text for text files, base64 for binary files)
    """
    # Internal byte range parameters - not exposed to user
    # Default to first 8KB (8192 bytes) for preview - kept below common file save thresholds
    PREVIEW_BYTE_RANGE = 8192

    try:
        download_url_obj = await _get_download_url(api, path, token)
        download_url = download_url_obj[0][0]

        headers = {
            "Authorization": token,
            "Range": f"bytes=0-{PREVIEW_BYTE_RANGE - 1}"
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(download_url, headers=headers)
            # 206 Partial Content is expected for range requests, treat as success
            if response.status_code not in (200, 206):
                response.raise_for_status()
            content = response.content

            # Try to decode as text first
            try:
                text_content = content.decode('utf-8')
                return {
                    "data": text_content,
                    "preview_size": len(content),
                    "is_preview": True,
                    "source": "bvbrc-workspace"
                }
            except UnicodeDecodeError:
                # If it's binary, encode as base64
                base64_content = base64.b64encode(content).decode('utf-8')
                return {
                    "data": f"<base64_encoded_data>{base64_content}</base64_encoded_data>",
                    "preview_size": len(content),
                    "is_preview": True,
                    "source": "bvbrc-workspace"
                }
    except Exception as e:
        return {
            "error": f"Error previewing file: {str(e)}",
            "errorType": "API_ERROR",
            "source": "bvbrc-workspace"
        }

async def workspace_read_range(api: JsonRpcCaller, path: str, token: str, start_byte: int = 0, max_bytes: int = 8192) -> dict:
    """
    Read a specific byte range from a workspace file using HTTP Range headers.

    Args:
        api: JsonRpcCaller instance configured with workspace URL and token
        path: Path to the file
        token: Authentication token for API calls
        start_byte: Zero-based byte offset to start reading from
        max_bytes: Maximum number of bytes to read (capped at 1 MiB)

    Returns:
        Dictionary containing ranged file data (text for text files, base64 for binary files)
    """
    if start_byte < 0:
        return {
            "error": "start_byte must be >= 0",
            "errorType": "INVALID_PARAMETERS",
            "source": "bvbrc-workspace"
        }

    if max_bytes <= 0:
        return {
            "error": "max_bytes must be > 0",
            "errorType": "INVALID_PARAMETERS",
            "source": "bvbrc-workspace"
        }

    max_bytes = min(max_bytes, 1024 * 1024)
    end_byte = start_byte + max_bytes - 1

    try:
        download_url_obj = await _get_download_url(api, path, token)
        download_url = download_url_obj[0][0]

        headers = {
            "Authorization": token,
            "Range": f"bytes={start_byte}-{end_byte}"
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(download_url, headers=headers)
            if response.status_code not in (200, 206):
                response.raise_for_status()
            content = response.content

            try:
                data = content.decode("utf-8")
            except UnicodeDecodeError:
                base64_content = base64.b64encode(content).decode("utf-8")
                data = f"<base64_encoded_data>{base64_content}</base64_encoded_data>"

            return {
                "data": data,
                "start_byte": start_byte,
                "bytes_read": len(content),
                "requested_max_bytes": max_bytes,
                "next_start_byte": start_byte + len(content),
                "source": "bvbrc-workspace"
            }
    except Exception as e:
        return {
            "error": f"Error reading byte range: {str(e)}",
            "errorType": "API_ERROR",
            "source": "bvbrc-workspace"
        }

async def _get_download_url(api: JsonRpcCaller, path: str, token: str) -> str:
    """
    Get the download URL of a file from the workspace using the JSON-RPC API.

    Args:
        api: JsonRpcCaller instance configured with workspace URL and token
        path: Path to the file to get the download URL of
        token: Authentication token for API calls
    Returns:
        String representation of the download URL
    """
    try:
        result = await api.acall("Workspace.get_download_url", {
            "objects": [path],
        },1, token)
        return result
    except Exception as e:
        return [f"Error getting download URL: {str(e)}"]

def _get_user_id_from_token(token: str) -> str:
    """
    Extract user ID from a BV-BRC/KBase style auth token.
    Returns None if token is None or invalid.
    """
    if not token:
        return None
    try:
        # Token format example: "un=username|..."; take first segment and strip prefix
        return token.split('|')[0].replace('un=','')
    except Exception as e:
        print(f"Error extracting user ID from token: {e}")
        return None

async def workspace_upload(api: JsonRpcCaller, filename: str, upload_dir: str = None, token: str = None) -> dict:
    """
    Create an upload URL for a file in the workspace using the JSON-RPC API.

    Args:
        api: JsonRpcCaller instance configured with workspace URL and token
        filename: Name of the file to create upload URL for
        upload_dir: Directory to upload the file to (defaults to /<user_id>/home)
        token: Authentication token for API calls (required)
    Returns:
        String representation of the upload URL response with parsed metadata
    """
    try:

        if not token:
            return {
                "error": "Authentication token not provided",
                "errorType": "INVALID_PARAMETERS",
                "source": "bvbrc-workspace"
            }

        if not upload_dir:
            user_id = _get_user_id_from_token(token)
            if not user_id:
                return {
                    "error": "Unable to derive user id from token",
                    "errorType": "INVALID_PARAMETERS",
                    "source": "bvbrc-workspace"
                }
            upload_dir = '/' + user_id + '/home'
        download_url_path = os.path.join(upload_dir,os.path.basename(filename))
        # call format: workspace file location, file type, object metadata, object content
        result = await _workspace_create(
            api,
            [[download_url_path, 'unspecified', {}, '']],
            token,
            create_upload_nodes=True,
            overwrite=None
        )

        # Parse the result if successful
        if result and len(result) > 0 and len(result[0]) > 0:
            # Extract the metadata array from result[0][0]
            meta_list = result[0][0]

            # Convert the array to a structured object
            meta_obj = {
                "id": meta_list[4],
                "path": meta_list[2] + meta_list[0],
                "name": meta_list[0],
                "type": meta_list[1],
                "creation_time": meta_list[3],
                "link_reference": meta_list[11],
                "owner_id": meta_list[5],
                "size": meta_list[6],
                "userMeta": meta_list[7],
                "autoMeta": meta_list[8],
                "user_permission": meta_list[9],
                "global_permission": meta_list[10],
                "timestamp": meta_list[3]  # Keep as string for now, could parse to timestamp if needed
            }

            upload_url = meta_obj["link_reference"]

            msg = {
                "file": os.path.basename(filename),
                "uploadDirectory": upload_dir,
                "url": upload_url,
                "source": "bvbrc-workspace"
            }

            # Upload the file to the upload URL
            print(f"Uploading file to {upload_url}")
            upload_result = _upload_file_to_url(filename, upload_url, token)
            print(f"Upload result: {upload_result}")
            if upload_result.get("success"):
                msg["upload_status"] = "success"
                msg["upload_message"] = upload_result.get("message", "File uploaded successfully")
            else:
                msg["upload_status"] = "failed"
                msg["upload_error"] = upload_result.get("error", "Upload failed")

            return msg
        else:
            return {
                "error": "No valid result returned from workspace API",
                "errorType": "API_ERROR",
                "source": "bvbrc-workspace"
            }

    except Exception as e:
        return {
            "error": f"Error creating upload URL: {str(e)}",
            "errorType": "API_ERROR",
            "source": "bvbrc-workspace"
        }

async def _workspace_create(api: JsonRpcCaller, objects: list, token: str, create_upload_nodes: bool = True, overwrite: Any = None):
    """
    Helper to invoke Workspace.create via JSON-RPC.
    """
    try:
        return await api.acall(
            "Workspace.create",
            {
                "objects": objects,
                "createUploadNodes": create_upload_nodes,
                "overwrite": overwrite
            },
            1,
            token
        )
    except Exception as e:
        return [f"Error creating workspace object: {str(e)}"]

def _upload_file_to_url(filename: str, upload_url: str, token: str) -> dict:
    """
    Upload a file to the specified Shock API URL using binary data.

    Args:
        filename: Path to the file to upload
        upload_url: The upload URL from workspace API
        token: Authentication token for API calls
    Returns:
        Dictionary with upload result status and message
    """
    try:
        # Check if file exists
        if not os.path.exists(filename):
            return {"success": False, "error": f"File {filename} does not exist"}

        # Read the file content
        with open(filename, 'rb') as file:
            file_content = file.read()

        # Set up headers for the Shock API request
        headers = {
            'Authorization': 'OAuth ' + token
        }

        # Prepare the file for multipart form data upload
        with open(filename, 'rb') as file:
            files = {
                'upload': (os.path.basename(filename), file, 'application/octet-stream')
            }

            # Make the POST request with multipart form data
            response = requests.put(upload_url, files=files, headers=headers, timeout=30)

        if response.status_code == 200:
            return {
                "success": True,
                "message": f"File {filename} uploaded successfully",
                "status_code": response.status_code
            }
        else:
            return {
                "success": False,
                "error": f"Upload failed with status code {response.status_code}: {response.text}",
                "status_code": response.status_code
            }

    except Exception as e:
        return {"success": False, "error": f"Upload failed: {str(e)}"}

async def workspace_create_genome_group(api: JsonRpcCaller, genome_group_path: str, genome_id_list: List[str], token: str) -> dict:
    """
    Create a genome group in the workspace using the JSON-RPC API.
    """
    genome_group_name = genome_group_path.split('/')[-1]
    try:
        content = {
            'id_list': {
                'genome_id': genome_id_list
            },
            'name': genome_group_name
        }
        print("content", json.dumps(content, indent=2), file=sys.stderr)
        result = await api.acall("Workspace.create", [{
            "objects": [[genome_group_path, 'genome_group', {}, content]]
        }],1, token)

        # Add source field to response
        if isinstance(result, dict):
            result["source"] = "bvbrc-workspace"
            return result
        else:
            return {
                "data": result,
                "source": "bvbrc-workspace"
            }
    except Exception as e:
        return {
            "error": f"Error creating genome group: {str(e)}",
            "errorType": "API_ERROR",
            "source": "bvbrc-workspace"
        }

async def workspace_create_feature_group(api: JsonRpcCaller, feature_group_path: str, feature_id_list: List[str], token: str) -> dict:
    """
    Create a feature group in the workspace using the JSON-RPC API.
    """
    feature_group_name = feature_group_path.split('/')[-1]
    try:
        content = {
            'id_list': {
                'feature_id': feature_id_list
            },
            'name': feature_group_name
        }
        result = await api.acall("Workspace.create", {
            "objects": [[feature_group_path, 'feature_group', {}, content]]
        },1, token)

        # Add source field to response
        return {
            "data": result[0][0],
            "source": "bvbrc-workspace"
        }
    except Exception as e:
        return {
            "error": f"Error creating feature group: {str(e)}",
            "errorType": "API_ERROR",
            "source": "bvbrc-workspace"
        }

async def workspace_get_object(api: JsonRpcCaller, path: str, metadata_only: bool = False, token: str = None) -> dict:
    """
    Get an object from the workspace using the JSON-RPC API.

    Args:
        api: JsonRpcCaller instance configured with workspace URL and token
        path: Path to the object to retrieve
        metadata_only: If True, only return metadata without the actual data
        token: Authentication token for API calls
    Returns:
        Dictionary containing metadata and optionally data
    """
    if not path:
        return {
            "error": "Invalid Path(s) to retrieve",
            "errorType": "INVALID_PARAMETERS",
            "source": "bvbrc-workspace"
        }

    try:
        # Decode URL-encoded path
        path = requests.utils.unquote(path)

        # Call Workspace.get API
        result = await api.acall("Workspace.get", {
            "objects": [path],
            "metadata_only": metadata_only
        }, 1, token)

        # Validate response structure
        if not result or not result[0] or not result[0][0] or not result[0][0][0] or not result[0][0][0][4]:
            return {
                "error": "Object not found",
                "errorType": "NOT_FOUND",
                "source": "bvbrc-workspace"
            }

        # Extract metadata from nested array structure
        meta_array = result[0][0][0]
        metadata = {
            "name": meta_array[0],
            "type": meta_array[1],
            "path": meta_array[2],
            "creation_time": meta_array[3],
            "id": meta_array[4],
            "owner_id": meta_array[5],
            "size": meta_array[6],
            "userMeta": meta_array[7],
            "autoMeta": meta_array[8],
            "user_permissions": meta_array[9],
            "global_permission": meta_array[10],
            "link_reference": meta_array[11]
        }

        # If metadata only, return just the metadata
        if metadata_only:
            return {
                "metadata": metadata,
                "source": "bvbrc-workspace"
            }

        # Get the actual data
        data = result[0][0][1]

        return {
            "metadata": metadata,
            "data": data,
            "source": "bvbrc-workspace"
        }

    except Exception as e:
        return {
            "error": f"Error getting workspace object: {str(e)}",
            "errorType": "API_ERROR",
            "source": "bvbrc-workspace"
        }

async def workspace_get_genome_group_ids(api: JsonRpcCaller, genome_group_path: str, token: str) -> dict:
    """
    Get the IDs of the genomes in a genome group using the JSON-RPC API.
    """
    try:
        # Get the genome group object using workspace_get_object
        result = await workspace_get_object(api, genome_group_path, metadata_only=False, token=token)
        # Check if there was an error
        if "error" in result:
            return {
                "error": f"Error getting genome group: {result['error']}",
                "errorType": "API_ERROR",
                "source": "bvbrc-workspace"
            }
        # Extract genome IDs from the data
        data = json.loads(result.get("data", {}))
        if not data or "id_list" not in data:
            return {
                "error": "Genome group data not found or invalid structure",
                "errorType": "INVALID_RESPONSE",
                "source": "bvbrc-workspace"
            }

        genome_ids = data['id_list']['genome_id']
        # Ensure we return a list of strings
        if isinstance(genome_ids, list):
            return {
                "genome_ids": genome_ids,
                "count": len(genome_ids),
                "source": "bvbrc-workspace"
            }
        else:
            return {
                "genome_ids": [str(genome_ids)],
                "count": 1,
                "source": "bvbrc-workspace"
            }
    except Exception as e:
        return {
            "error": f"Error getting genome group IDs: {str(e)}",
            "errorType": "API_ERROR",
            "source": "bvbrc-workspace"
        }

async def workspace_get_feature_group_ids(api: JsonRpcCaller, feature_group_path: str, token: str) -> dict:
    """
    Get the IDs of the features in a feature group using the JSON-RPC API.
    """
    try:
        # Get the feature group object using workspace_get_object
        result = await workspace_get_object(api, feature_group_path, metadata_only=False, token=token)

        # Check if there was an error
        if "error" in result:
            return {
                "error": f"Error getting feature group: {result['error']}",
                "errorType": "API_ERROR",
                "source": "bvbrc-workspace"
            }

        # Extract feature IDs from the data
        data = json.loads(result.get("data", {}))
        if not data or "id_list" not in data:
            return {
                "error": "Feature group data not found or invalid structure",
                "errorType": "INVALID_RESPONSE",
                "source": "bvbrc-workspace"
            }

        feature_ids = data['id_list']['feature_id']

        # Ensure we return a list of strings
        if isinstance(feature_ids, list):
            return {
                "feature_ids": feature_ids,
                "count": len(feature_ids),
                "source": "bvbrc-workspace"
            }
        else:
            return {
                "feature_ids": [str(feature_ids)],
                "count": 1,
                "source": "bvbrc-workspace"
            }

    except Exception as e:
        return {
            "error": f"Error getting feature group IDs: {str(e)}",
            "errorType": "API_ERROR",
            "source": "bvbrc-workspace"
        }