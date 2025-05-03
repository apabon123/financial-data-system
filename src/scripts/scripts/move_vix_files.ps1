# PowerShell script to move VIX/VX-specific files to the new directory structure

# Market data VIX-specific files
$marketDataVixFiles = @(
    "fill_vx_zero_prices.py",
    "update_vx_futures.py",
    "check_vix_data.py",
    "debug_vix_index.py",
    "update_vix_index.py",
    "fill_vx_continuous_gaps.py",
    "generate_vix_roll_calendar.py",
    "load_cboe_vix_index.py",
    "load_vx_futures.py",
    "load_cboe_vx_futures.py"
)

# Analysis VIX-specific files
$analysisVixFiles = @(
    "find_missing_vx_continuous.py",
    "show_vix_continuous_data.py",
    "view_vix_data_formatted.py",
    "view_vix_data.py",
    "verify_vx_continuous.py",
    "verify_vx_rollovers.py",
    "analyze_vx_data.py",
    "check_vx_data.py",
    "export_vix_comparison.py"
)

# Create directories if they don't exist
$marketDataVixDir = "src/scripts/market_data/vix"
$analysisVixDir = "src/scripts/analysis/vix"

# Copy market data VIX files
foreach ($file in $marketDataVixFiles) {
    $source = "src/scripts/market_data/$file"
    $destination = "$marketDataVixDir/$file"
    
    if (Test-Path $source) {
        Write-Host "Copying $file to VIX market data directory..."
        Copy-Item $source -Destination $destination
    } else {
        Write-Host "Warning: $file not found in market_data directory"
    }
}

# Copy analysis VIX files
foreach ($file in $analysisVixFiles) {
    $source = "src/scripts/analysis/$file"
    $destination = "$analysisVixDir/$file"
    
    if (Test-Path $source) {
        Write-Host "Copying $file to VIX analysis directory..."
        Copy-Item $source -Destination $destination
    } else {
        Write-Host "Warning: $file not found in analysis directory"
    }
}

Write-Host "`nComplete! VIX-specific files have been copied to their new locations."
Write-Host "The original files are still in place. Once you've verified everything works,"
Write-Host "you may want to remove the original files." 