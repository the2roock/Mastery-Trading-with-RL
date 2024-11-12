import os


class BinanceConfig:
    """Configuration class for Binance API credetials using environment variables."""
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    
    @staticmethod
    def validate() -> None:
        """Validate the presence of all required environment variables."""
        missing_vars = []
        for var in ["api_key", "api_secret"]:
            if getattr(BinanceConfig, var) is None:
                missing_vars.append(var)

        if missing_vars:
            raise EnvironmentError(f"Missing environment variables: {', '.join(missing_vars)}")

# Validate configuration on import
BinanceConfig.validate()
