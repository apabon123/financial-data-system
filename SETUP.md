# Environment Setup

## Dependencies
- Python 3.9+
- UV package manager
- Required API keys:
  - TradeStation API
  - Other economic data APIs

## Installation

### Install UV
\`\`\`bash
curl -sSf https://install.pydantic.dev | python3
\`\`\`

### Setup Virtual Environment
\`\`\`bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\\Scripts\\activate
# On Unix or MacOS:
source venv/bin/activate

# Install dependencies
uv pip install duckdb pandas requests beautifulsoup4 python-dotenv typer rich
\`\`\`

## Environment Variables
Create a \`.env\` file in the project root with the following:

\`\`\`
OPENAI_API_KEY=your-api-key-here
ANTHROPIC_API_KEY=your-api-key-here
TRADESTATION_API_KEY=your-api-key-here
TRADESTATION_API_SECRET=your-api-key-here
FRED_API_KEY=your-api-key-here
\`\`\`