import typer
from datetime import datetime
import pandas as pd
import duckdb
from pathlib import Path
import os
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import mplfinance as mpf

app = typer.Typer()

def load_data(symbol: str, interval: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Load data from the database for the specified parameters."""
    db_path = Path("data/financial_data.duckdb")
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}")
    
    # Parse interval into value and unit
    interval_value = 1
    interval_unit = interval.lower()
    
    conn = duckdb.connect(str(db_path))
    
    # Determine which table to query based on the symbol
    if symbol.endswith('_continuous'):
        query = f"""
        SELECT timestamp::DATE as date, open, high, low, close, volume
        FROM continuous_contracts
        WHERE symbol = '{symbol}'
        AND timestamp::DATE >= '{start_date}'::DATE
        AND timestamp::DATE <= '{end_date}'::DATE
        ORDER BY timestamp
        """
    else:
        query = f"""
        SELECT timestamp::DATE as date, open, high, low, close, volume
        FROM market_data
        WHERE symbol = '{symbol}'
        AND interval_value = {interval_value}
        AND interval_unit = '{interval_unit}'
        AND timestamp::DATE >= '{start_date}'::DATE
        AND timestamp::DATE <= '{end_date}'::DATE
        ORDER BY timestamp
        """
    
    df = conn.execute(query).df()
    conn.close()
    
    if df.empty:
        return df
        
    # Ensure data types are correct
    df['date'] = pd.to_datetime(df['date'])
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Set date as index for mplfinance
    df.set_index('date', inplace=True)
    
    return df

def create_visualization(df: pd.DataFrame, symbol: str, output_dir: str = "output") -> str:
    """Create an HTML file with chart and data table."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot candlestick chart using mplfinance
    mpf.plot(df, 
             type='candle',
             title=f'{symbol} Price',
             ylabel='Price',
             volume=False,
             figsize=(15, 8),
             savefig=dict(fname=os.path.join(output_dir, f"{symbol}_chart.png"), dpi=300, bbox_inches='tight'))
    
    # Format table data
    table_df = df.copy()
    table_df.index = table_df.index.strftime('%Y-%m-%d')
    for col in ['open', 'high', 'low', 'close']:
        table_df[col] = table_df[col].round(2)
    table_df['volume'] = table_df['volume'].astype(int)
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{symbol} Analysis</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .chart-container {{
                margin-bottom: 30px;
                text-align: center;
            }}
            .chart-container img {{
                max-width: 100%;
                height: auto;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
                font-size: 14px;
            }}
            th, td {{
                padding: 12px;
                text-align: right;
                border: 1px solid #e0e0e0;
            }}
            th {{
                background-color: #f8f9fa;
                font-weight: 600;
                text-align: center;
            }}
            tr:nth-child(even) {{
                background-color: #f8f9fa;
            }}
            .symbol-header {{
                font-size: 24px;
                font-weight: 600;
                color: #333;
                margin-bottom: 20px;
                text-align: center;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="symbol-header">{symbol}</div>
            <div class="chart-container">
                <img src="{symbol}_chart.png" alt="Price Chart">
            </div>
            <div class="table-container">
                {table_df.to_html(classes='data-table')}
            </div>
        </div>
    </body>
    </html>
    """
    
    # Save the file
    output_path = os.path.join(output_dir, f"{symbol}_analysis.html")
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    return output_path

@app.command()
def visualize(
    symbol: str = typer.Argument(..., help="Symbol to visualize"),
    interval: str = typer.Argument(..., help="Data interval (e.g., daily)"),
    start_date: str = typer.Argument(..., help="Start date (YYYY-MM-DD)"),
    end_date: str = typer.Argument(..., help="End date (YYYY-MM-DD)"),
    output_dir: str = typer.Option("output", help="Output directory")
):
    """Create an interactive visualization for the specified symbol and time period."""
    try:
        # Load data
        df = load_data(symbol, interval, start_date, end_date)
        
        if df.empty:
            typer.echo(f"No data found for {symbol} {interval} between {start_date} and {end_date}")
            return
        
        # Create visualization
        output_path = create_visualization(df, symbol, output_dir)
        typer.echo(f"Analysis saved to: {output_path}")
        
    except Exception as e:
        typer.echo(f"Error: {str(e)}", err=True)
        raise typer.Exit(1)

if __name__ == "__main__":
    app()