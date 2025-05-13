#!/bin/bash
# DB Inspector Dependency Checker
# This script checks all required and optional dependencies for the DB Inspector tool

# Determine Python executable to use (python3 preferred)
if command -v python3 &>/dev/null; then
    PYTHON="python3"
elif command -v python &>/dev/null; then
    PYTHON="python"
else
    echo "❌ ERROR: Python not found. Please install Python 3.8 or higher."
    echo "Visit https://www.python.org/downloads/"
    exit 1
fi

# Function to display a header
function display_header() {
    echo "==============================================="
    echo "$1"
    echo "==============================================="
    echo ""
}

# Check Python version
PY_VERSION=$($PYTHON -c "import sys; print('{}.{}'.format(sys.version_info.major, sys.version_info.minor))")
echo "Using Python $PY_VERSION"

if [[ "$(echo "$PY_VERSION < 3.8" | bc)" -eq 1 ]]; then
    echo "❌ ERROR: Python 3.8+ is required. Found Python $PY_VERSION"
    echo "Please upgrade your Python installation."
    exit 1
fi

display_header "DB Inspector Dependency Checker"

# Check core dependencies
echo "Checking core dependencies..."
echo ""

CORE_DEPS="pandas duckdb numpy pyyaml rich"
CORE_MISSING=""
CORE_INSTALLED=""

for dep in $CORE_DEPS; do
    if [ "$dep" = "pyyaml" ]; then
        $PYTHON -c "import yaml" 2>/dev/null
    else
        $PYTHON -c "import $dep" 2>/dev/null
    fi
    
    if [ $? -ne 0 ]; then
        CORE_MISSING="$CORE_MISSING $dep"
    else
        CORE_INSTALLED="$CORE_INSTALLED $dep"
    fi
done

if [ -z "$CORE_MISSING" ]; then
    echo "✅ All core dependencies are installed:"
    for dep in $CORE_DEPS; do
        if [ "$dep" = "pyyaml" ]; then
            VERSION=$($PYTHON -c "import yaml; print(getattr(yaml, '__version__', 'unknown'))" 2>/dev/null)
        else
            VERSION=$($PYTHON -c "import $dep; print(getattr($dep, '__version__', 'unknown'))" 2>/dev/null)
        fi
        echo "   - $dep: $VERSION"
    done
else
    echo "❌ Missing core dependencies:$CORE_MISSING"
    echo ""
    echo "These dependencies are required for DB Inspector to function."
    echo "Install them with:"
    echo ""
    echo "   pip install$CORE_MISSING"
    echo ""
    echo "Or install all dependencies with:"
    echo ""
    echo "   pip install -r requirements.txt"
fi

echo ""
echo "Checking optional dependencies..."
echo ""

# Check advanced visualization dependencies
VIZ_DEPS="matplotlib seaborn"
VIZ_MISSING=""

for dep in $VIZ_DEPS; do
    $PYTHON -c "import $dep" 2>/dev/null
    if [ $? -ne 0 ]; then
        VIZ_MISSING="$VIZ_MISSING $dep"
    fi
done

if [ -z "$VIZ_MISSING" ]; then
    echo "✅ All visualization dependencies are installed."
else
    echo "⚠️ Missing visualization dependencies:$VIZ_MISSING"
    echo "   These are used for data plots and charts."
fi

# Check schema visualization dependencies
SCHEMA_DEPS="networkx"
SCHEMA_MISSING=""

for dep in $SCHEMA_DEPS; do
    $PYTHON -c "import $dep" 2>/dev/null
    if [ $? -ne 0 ]; then
        SCHEMA_MISSING="$SCHEMA_MISSING $dep"
    fi
done

if [ -z "$SCHEMA_MISSING" ]; then
    echo "✅ All schema visualization dependencies are installed."
else
    echo "⚠️ Missing schema visualization dependencies:$SCHEMA_MISSING"
    echo "   These are used for database schema visualization."
fi

# Check SQL dependencies
SQL_DEPS="prompt_toolkit pygments"
SQL_MISSING=""

for dep in $SQL_DEPS; do
    $PYTHON -c "import $dep" 2>/dev/null
    if [ $? -ne 0 ]; then
        SQL_MISSING="$SQL_MISSING $dep"
    fi
done

if [ -z "$SQL_MISSING" ]; then
    echo "✅ All SQL interface dependencies are installed."
else
    echo "⚠️ Missing SQL interface dependencies:$SQL_MISSING"
    echo "   These are used for the interactive SQL interface."
fi

# Check other optional dependencies
OTHER_DEPS="tabulate"
OTHER_MISSING=""

for dep in $OTHER_DEPS; do
    $PYTHON -c "import $dep" 2>/dev/null
    if [ $? -ne 0 ]; then
        OTHER_MISSING="$OTHER_MISSING $dep"
    fi
done

if [ -z "$OTHER_MISSING" ]; then
    echo "✅ All other optional dependencies are installed."
else
    echo "⚠️ Missing other optional dependencies:$OTHER_MISSING"
    echo "   These are used for enhanced formatting and display."
fi

# Collect all missing optional dependencies
ALL_OPT_MISSING="$VIZ_MISSING$SCHEMA_MISSING$SQL_MISSING$OTHER_MISSING"

if [ ! -z "$ALL_OPT_MISSING" ]; then
    echo ""
    echo "To install all missing optional dependencies, run:"
    echo ""
    echo "   pip install$ALL_OPT_MISSING"
    echo ""
fi

# Summary
echo ""
echo "Summary:"
echo "========"

if [ -z "$CORE_MISSING" ] && [ -z "$ALL_OPT_MISSING" ]; then
    echo "✅ All dependencies are installed. DB Inspector should work with full functionality."
    echo "   Run ./db_inspect_enhanced to start the application."
elif [ -z "$CORE_MISSING" ]; then
    echo "⚠️ Core dependencies are installed, but some optional features will be limited."
    echo "   Run ./db_inspect_enhanced to start with limited functionality."
    echo "   Run ./db_inspect_enhanced --fallback for a more basic interface if you encounter issues."
else
    echo "❌ Core dependencies are missing. DB Inspector may not function correctly."
    echo "   Install required dependencies before running the application."
    echo "   Alternatively, run ./db_inspect_enhanced --fallback for a basic interface using only standard library."
fi

echo ""
echo "For more information, see the documentation or run:"
echo "   ./db_inspect_enhanced --help"