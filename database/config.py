import os


class DatabaseConfig:
    """Configuration class for database connection settings using environment variables."""
    host = os.getenv("DATABASE_HOST")
    user = os.getenv("DATABASE_USER")
    port = os.getenv("DATABASE_PORT")
    password = os.getenv("DATABASE_PASSWORD")
    name = os.getenv("DATABASE_NAME")

    @staticmethod
    def validate() -> None:
        """Validate the presence of all required environment variables."""
        missing_vars = []
        for var in ["host", "user", "port", "password", "name"]:
            if getattr(DatabaseConfig, var) is None:
                missing_vars.append(var)

        if missing_vars:
            raise EnvironmentError(f"Missing environment variables: {', '.join(missing_vars)}")
        

# Validate configuration on import
DatabaseConfig.validate()
