# BV-BRC MCP Server

A Model Context Protocol (MCP) server for the Bacterial-Viral Bioinformatics resource Center:
- **Data Tools**: Query BV-BRC Solr collections for genome, feature, and other biological data
- **Service Tools**: Submit and manage BV-BRC analysis jobs (assembly, annotation, BLAST, etc.)
- **Workspace Tools**: Manage BV-BRC workspace files, folders, and groups

<details>
<summary><h2>Features</h2></summary>

### Data Tools
- `query_collection`: Query any BV-BRC Solr collection with flexible filtering
- `solr_collection_parameters`: Get schema information for collections
- `solr_query_instructions`: Get help on query syntax
- `solr_collections`: List all available collections

### Service Tools
- `list_service_apps`: List all available BV-BRC analysis services
- `get_job_details`: Query the status of submitted jobs
- Submit jobs for various analyses:
  - Genome Assembly
  - Genome Annotation
  - Comprehensive Genome Analysis
  - BLAST
  - Primer Design
  - Variation Analysis
  - TnSeq
  - Phylogenetic Trees (Bacterial Genome Tree, Gene Tree)
  - SNP Analysis (Whole Genome, MSA)
  - Metagenomics (Taxonomic Classification, Binning, Read Mapping)
  - RNA-Seq
  - Viral Services (SARS-CoV-2 Analysis, Sequence Submission)
  - And many more...

### Workspace Tools
- `workspace_ls_tool`: List workspace contents
- `workspace_search_tool`: Search workspace for files
- `workspace_get_file_metadata_tool`: Get file metadata
- `workspace_download_file_tool`: Download workspace files
- `workspace_upload`: Upload files to workspace
- `create_genome_group`: Create genome groups
- `create_feature_group`: Create feature groups
- `get_genome_group_ids`: Get genome IDs from a group
- `get_feature_group_ids`: Get feature IDs from a group

</details>

<details>
<summary><h2>Connecting the Remote BV-BRC MCP Server to ChatGPT</h2></summary>

### Step 1: Enable Developer Mode

1. Click the **plus** next to "Ask me anything"
2. Click **"Add sources"**
3. You should now see "Sources" and "Add" below your chat box
4. Click the **down arrow** next to "Add"
5. Click **"Connect more"**
6. Scroll down to **Advanced Settings**
7. Click the toggle next to **Developer Mode** (must be "on")
8. Click **Back**

### Step 2: Create MCP Server Connection

1. In the upper right-hand corner, click **"Create"**
2. Fill in the following:
   - **Icon**: Optional
   - **Name**: BV-BRC MCP
   - **Description**: ''
   - **MCP Server URL**: https://dev-7.bv-brc.org/mcp
3. **Authentication**: 
   - Leave authentication on OAuth
4. Check the box if you **Trust this application**
5. Click **"Create"**

### Step 3: Connect to Your Server

1. You should now see 'BV-BRC MCP' under **"Enabled apps & connectors"**
2. Click the 'X' in the top left to go back to the chat screen
3. In a **New Chat**, click the '+' button and hover over **More**
  - You should see **BV-BRC MCP** as an option under **Canvas**
4. Select **BV-BRC MCP**

</details>

<details>
<summary><h2>Connecting the Remote BV-BRC MCP Server to Claude</h2></summary>

1. Click **account** in bottom left and go to **settings**

2. Click **'Connectors'**

3. Click **'Add custom connector'**

4. Fill in the following:
   - **Name**: BV-BRC MCP
   - **Remote MCP server URL**: https://dev-7.bv-brc.org/mcp

5. Click **'Add'**

6. Then click **'Connect'**

7. Log into BV-BRC

8. It's now available to use in a new chat

</details>

<details>
<summary><h2>Installing as a Claude Extension</h2></summary>

Open your terminal

0. Clone Github Repository
   ```bash
   git clone https://github.com/cucinellclark/bvbrc-mcp-server
   cd bvbrc-mcp-server
   ```

1. Install the requirements:
   ```bash
   ./install.sh
   ```

2. Activate the environment:
   ```bash
   source mcp_env/bin/activate
   ```

3. Install the MCP Builder CLI:
   ```bash
   npm install -g @anthropic-ai/mcpb
   ```

4. Generate the MCP configuration file (if you haven't already):
   ```bash
   python3 bvbrc_login_and_setup.py
   ```

5. Pack the extension:
   ```bash
   mcpb pack
   ```

6. In Claude, Go to Settings, then Extensions

7. Click Advanced settings, then Install Extension

8. Select the file 'bvbrc-mcp-server.mcpb' then click Preview

9. It should pull up a preview page, then click Install

</details>

<details>
<summary><h2>MCP Development</h2></summary>

## Note
Local server development is recommended for working on MCP tools

## Installation

Run the installation script, which will create the virtual python environment and install a data api and remaining requirements
```bash
# clone the repository and enter it
./install.sh
```

## Local server development

Generate mcp config file:
```bash
python3 bvbrc_login_and_setup.py
```
Creates mcp_config.json

Paste its contents into your chatbots mcp config file

## Remote server development
Configure the server (for remote servers):

Edit config.json to set:
- API URLs (workspace, service, data)
- Server host and port
- Authentication URL

## Configuration

The `config.json` file contains:

```json
{
    "base_url": "https://www.bv-brc.org/api-bulk",
    "workspace_url": "https://p3.theseed.org/services/Workspace",
    "service_api_url": "https://p3.theseed.org/services/app_service",
    "similar_genome_finder_api_url": "https://p3.theseed.org/services/minhash_service",
    "authentication_url": "https://user.patricbrc.org/authenticate",
    "mcp_url": "127.0.0.1",
    "port": 12010
}
```

## Running the Server

### HTTP Mode (for web clients, ChatGPT, etc.)

Start the HTTP server:
```bash
./start_http_server.sh
```

The server will start on the configured host and port (default: `127.0.0.1:12010`).

#### STDIO Mode (for Claude Desktop, etc.)

Start the STDIO server:
Set it up in your MCP client configuration (e.g., Claude Desktop):
See installation section
```json
{
  "mcpServers": {
    "bvbrc": {
      "command": "python",
      "args": ["/path/to/bvbrc-mcp/stdio_server.py"],
      "env": {
        "KB_AUTH_TOKEN": "your_token_here"
      }
    }
  }
}
```

## Instructions for MCP configuration Setup

[Claude](https://support.claude.com/en/articles/11175166-getting-started-with-custom-connectors-using-remote-mcp)

[ChatGPT](https://docs.atlan.com/product/capabilities/atlan-ai/how-tos/chatgpt-remote-mcp)

## Architecture

### Directory Structure

```
bvbrc-mcp-server/
├── __init__.py                      # Package initialization
├── __main__.py                      # Entry point for module execution
├── http_server.py                    # HTTP server (for web clients, ChatGPT)
├── install.sh                       # Installation script
├── mcp_config.json                  # Generated MCP configuration
├── mcp_example.json                 # Example MCP configuration
├── mcp.pm2.config.js                 # PM2 process manager configuration
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── start_server.sh                  # Server startup script
├── stdio_server.py                  # STDIO server (for Claude Desktop, etc.)
├── bvbrc_login_and_setup.py        # Script to login and generate MCP config file
├── common/                          # Common utility modules
│   ├── __init__.py
│   ├── auth.py                      # OAuth2 authentication endpoints (HTTP only)
│   ├── json_rpc.py                  # JSON-RPC client for API calls
│   └── token_provider.py            # Unified authentication token handling
├── config/                          # Configuration files
│   ├── config.json                  # Main configuration file
│   └── ...
├── bvbrc-python-api/                # BV-BRC Python API dependency
│   ├── bvbrc_solr_api/              # Solr API implementation
│   ├── pyproject.toml
│   └── README.md
├── functions/                       # Function implementation modules
│   ├── __init__.py
│   ├── data_functions.py            # Data query implementations
│   ├── service_functions.py         # Service job implementations
│   └── workspace_functions.py       # Workspace implementations
├── images/                          # Image assets
│   └── bvbrc_logo_base64.txt
├── prompts/                         # Collection schema documentation
│   ├── antibiotics.txt
│   ├── bacterial_genome_tree.txt
│   ├── bioset.txt
│   ├── bioset_result.txt
│   ├── blast.txt
│    ...
└── tools/                           # Tool registration modules
    ├── __init__.py
    ├── data_tools.py                # Data query tools registration
    ├── service_tools.py             # Service job tools registration
    └── workspace_tools.py           # Workspace management tools registration
```

</details>
