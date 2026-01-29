# How to Add Cognee MCP Server to Goose

This guide shows you exactly how to configure Goose to use the cognee-mcp server.

## Understanding Goose MCP Configuration

Goose has two ways to add MCP servers:

### **Option 1: MCP Servers (Recommended)**
Add to `~/.config/goose/profiles.yaml` under `mcp_servers`

### **Option 2: Extensions**
Use Goose's extension system (requires creating an extension package)

---

## 🚀 Quick Setup (2 Minutes)

### Step 1: Create/Edit Goose Profile

```bash
# Create config directory if it doesn't exist
mkdir -p ~/.config/goose

# Edit or create profiles.yaml
nano ~/.config/goose/profiles.yaml  # or use your preferred editor
```

### Step 2: Add Cognee MCP Server Configuration

Add this to your `profiles.yaml`:

```yaml
default:
  provider: ollama
  processor: llama3.2:latest
  accelerator: llama3.2:latest
  moderator: llama3.2:latest

  # Add MCP servers section
  mcp_servers:
    cognee:
      command: uv
      args:
        - run
        - python
        - /home/ashiedu/apps/cognee/cognee-mcp/src/server.py
        - --transport
        - stdio
      env:
        LLM_PROVIDER: ollama
        LLM_MODEL: llama3.2:latest
        LLM_ENDPOINT: http://localhost:11434
        EMBEDDING_PROVIDER: ollama
        EMBEDDING_MODEL: nomic-embed-text:latest
        EMBEDDING_ENDPOINT: http://localhost:11434
        VECTOR_DB_PROVIDER: lancedb
        VECTOR_DB_URL: /home/ashiedu/apps/cognee/cognee-mcp/.cognee_system/vector_db
        GRAPH_DATABASE_PROVIDER: networkx
        GRAPH_DATABASE_URL: /home/ashiedu/apps/cognee/cognee-mcp/.cognee_system/graph.pkl
```

### Step 3: Verify Configuration

```bash
# Test the configuration
goose session start

# In Goose, try:
# "List available MCP tools"
# "Use cognee to search for 'test'"
```

---

## 📋 Alternative: Complete Profile Example

If you want a complete profile configuration, create this file:

**File: `~/.config/goose/profiles.yaml`**

```yaml
# Default profile with Ollama + Cognee MCP
default:
  provider: ollama
  processor: llama3.2:latest
  accelerator: llama3.2:latest
  moderator: llama3.2:latest

  mcp_servers:
    # Cognee Knowledge Management
    cognee:
      command: uv
      args:
        - run
        - python
        - /home/ashiedu/apps/cognee/cognee-mcp/src/server.py
        - --transport
        - stdio
      env:
        # LLM Configuration
        LLM_PROVIDER: ollama
        LLM_MODEL: llama3.2:latest
        LLM_ENDPOINT: http://localhost:11434

        # Embedding Configuration
        EMBEDDING_PROVIDER: ollama
        EMBEDDING_MODEL: nomic-embed-text:latest
        EMBEDDING_ENDPOINT: http://localhost:11434
        EMBEDDING_DIMENSIONS: "768"

        # Vector Database
        VECTOR_DB_PROVIDER: lancedb
        VECTOR_DB_URL: /home/ashiedu/apps/cognee/cognee-mcp/.cognee_system/vector_db

        # Graph Database
        GRAPH_DATABASE_PROVIDER: networkx
        GRAPH_DATABASE_URL: /home/ashiedu/apps/cognee/cognee-mcp/.cognee_system/graph.pkl

        # Data Storage
        DATA_DIRECTORY: /home/ashiedu/apps/cognee/cognee-mcp/.cognee_system/data

# Alternative profile with API mode
cognee-api:
  provider: ollama
  processor: llama3.2:latest
  accelerator: llama3.2:latest
  moderator: llama3.2:latest

  mcp_servers:
    cognee:
      command: uv
      args:
        - run
        - python
        - /home/ashiedu/apps/cognee/cognee-mcp/src/server.py
        - --transport
        - stdio
        - --api-url
        - http://localhost:8000
      env:
        COGNEE_API_URL: http://localhost:8000
        # No need for LLM/embedding config in API mode
```

---

## 🧪 Testing the Integration

### 1. Start Ollama (if not running)
```bash
ollama serve
```

### 2. Pull Required Models (if not already downloaded)
```bash
ollama pull llama3.2:latest
ollama pull nomic-embed-text:latest
```

### 3. Test Cognee MCP Server Standalone
```bash
cd /home/ashiedu/apps/cognee/cognee-mcp

# Create .env file
cat > .env << 'EOF'
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2:latest
LLM_ENDPOINT=http://localhost:11434
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text:latest
EMBEDDING_ENDPOINT=http://localhost:11434
VECTOR_DB_PROVIDER=lancedb
GRAPH_DATABASE_PROVIDER=networkx
EOF

# Install dependencies
uv sync --dev --all-extras --reinstall

# Test server
uv run python src/server.py --transport stdio
# Press Ctrl+C after seeing "MCP Server started"
```

### 4. Test in Goose
```bash
goose session start

# Try these commands in Goose:
# 1. "What MCP tools are available?"
# 2. "Use cognee to cognify this text: 'Goose is an AI assistant'"
# 3. "Use cognee to search for 'goose'"
```

---

## 🔧 Troubleshooting

### Issue: "MCP server 'cognee' not found"

**Solution 1: Check YAML syntax**
```bash
# Validate YAML syntax
python3 -c "import yaml; yaml.safe_load(open('$HOME/.config/goose/profiles.yaml'))"
```

**Solution 2: Check file paths**
```bash
# Verify cognee-mcp exists
ls -la /home/ashiedu/apps/cognee/cognee-mcp/src/server.py

# Verify uv is in PATH
which uv
```

**Solution 3: Use absolute path to uv**
```yaml
mcp_servers:
  cognee:
    command: /home/ashiedu/.local/bin/uv  # Use absolute path
    args:
      - run
      - python
      # ... rest of config
```

### Issue: "Server starts but no tools available"

**Check server logs:**
```bash
# Run server manually to see output
cd /home/ashiedu/apps/cognee/cognee-mcp
uv run python src/server.py --transport stdio 2>&1 | tee server.log
```

### Issue: "Environment variables not working"

**Verify .env file:**
```bash
cd /home/ashiedu/apps/cognee/cognee-mcp
cat .env
```

**Check Ollama is running:**
```bash
curl http://localhost:11434/api/tags
```

---

## 📊 Expected Tools Available

Once configured, you should see these 11 Cognee MCP tools in Goose:

| Tool | Description |
|------|-------------|
| **cognify** | Build knowledge graphs from documents |
| **search** | Multi-mode search (GRAPH_COMPLETION, RAG, CODE, etc.) |
| **save_interaction** | Learn from conversations |
| **list_data** | List all indexed datasets |
| **delete** | Remove specific dataset |
| **prune** | Reset all cognee data |
| **cognify_status** | Check processing pipeline status |
| **codify** | Analyze code repositories |
| **codify_status** | Check code analysis status |
| **get_developer_rules** | Get learned coding rules |
| **cognee_add_developer_rules** | Ingest developer rule files |

---

## 🎯 Example Usage in Goose

```
You: "Use cognee to index this document: 'Goose is an AI assistant that helps developers.'"

Goose: [Calls cognee.cognify tool]
"I've indexed that document into cognee's knowledge graph."

You: "Now search for information about AI assistants"

Goose: [Calls cognee.search tool]
"Here's what I found: Goose is an AI assistant that helps developers..."
```

---

## 🔄 Alternative: Use Goose Extensions (Advanced)

If you prefer the extension system, you'll need to create a Goose extension package:

```bash
# This is more complex - recommended to use MCP servers approach above
goose extension create cognee-extension
```

However, the **MCP servers approach is simpler and recommended** for tools like cognee.

---

## 📚 Related Documentation

- **Goose MCP Docs**: Check `goose --help` or Goose documentation
- **Cognee Quickstart**: `/home/ashiedu/apps/cognee/cognee-mcp/OLLAMA_QUICKSTART.md`
- **Full Ollama Setup**: `/home/ashiedu/apps/cognee/cognee-mcp/OLLAMA_SETUP.md`

---

## ✅ Quick Checklist

- [ ] Ollama is running (`ollama serve`)
- [ ] Models downloaded (`ollama pull llama3.2:latest`)
- [ ] Cognee .env created (`/home/ashiedu/apps/cognee/cognee-mcp/.env`)
- [ ] Dependencies installed (`uv sync --dev --all-extras --reinstall`)
- [ ] profiles.yaml created (`~/.config/goose/profiles.yaml`)
- [ ] MCP server config added to profiles.yaml
- [ ] Tested standalone (`uv run python src/server.py --transport stdio`)
- [ ] Tested in Goose (`goose session start`)

---

**You're ready to use Cognee with Goose!** 🎉
