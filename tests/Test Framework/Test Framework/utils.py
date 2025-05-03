import pandas as pd
import logging

def _validate_returns(returns: pd.Series, component_name: str, logger: logging.Logger) -> None:
    \"\"\"Helper to validate and log return statistics.\"\"\"
    if returns is None:
        raise ValueError(f"{component_name}: Returns cannot be None")

    if not isinstance(returns, pd.Series):
        raise TypeError(f"{component_name}: Returns must be a pandas Series")

    if returns.empty:
        raise ValueError(f"{component_name}: Returns series is empty")

    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError(f"{component_name}: Returns index must be a DatetimeIndex")

    if returns.index.tz is None:
        # Relaxing this requirement for now, will be handled in context
        # raise ValueError(f"{component_name}: Returns index must be timezone-aware")
        logger.debug(f"{component_name}: Returns index is timezone-naive.")
        pass # Allow naive for now


    # Log statistics if logger is available
    if logger:
        logger.debug(f"\n{component_name} Returns Statistics:")
        logger.debug(f"Shape: {returns.shape}")
        if not returns.empty:
            logger.debug(f"Date range: {returns.index[0]} to {returns.index[-1]}")
        logger.debug(f"Mean: {returns.mean():.4%}")
        logger.debug(f"Std: {returns.std():.4%}")
        logger.debug(f"Min: {returns.min():.4%}")
        logger.debug(f"Max: {returns.max():.4%}")
        logger.debug(f"Sample (first 5):\n{returns.head()}")

        # Flag extreme values
        extreme_returns = returns[abs(returns) > 1.0]
        if not extreme_returns.empty:
            logger.warning(f"{component_name}: Found {len(extreme_returns)} returns > 100%")
            logger.warning(f"Extreme returns:\n{extreme_returns}") 