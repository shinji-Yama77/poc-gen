"""
LLM Factory Module.
Centralized management of Language Model instances for the MCP server.
Handles configuration for both Azure OpenAI and direct OpenAI APIs.
"""
import logging
import os
from typing import Optional
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.language_models import BaseChatModel

load_dotenv()

logger = logging.getLogger("llm_factory")

class LLMFactory:
    """
    Factory class for creating and managing Language Model instances.
    Provides centralized configuration and initialization for all AI API connections.
    """

    @staticmethod
    def create_llm(config: Optional[dict] = None) -> BaseChatModel:
        """
        Create an LLM instance based on configuration and environment variables.

        Args:
            config: Optional configuration dictionary to override environment variables

        Returns:
            BaseChatModel: Configured LLM instance (ChatOpenAI or AzureChatOpenAI)

        Raises:
            ValueError: If required configuration is missing
        """
        config = config or {}

        ai_provider = config.get("ai_provider")

        if ai_provider == "azure":
            return LLMFactory._create_azure_openai_llm(config)

        raise ValueError(f"Unsupported AI provider: {ai_provider}")

    @staticmethod
    def _create_azure_openai_llm(config: dict) -> AzureChatOpenAI:
        """
        Create an Azure OpenAI LLM instance.

        Args:
            config: Configuration dictionary

        Returns:
            AzureChatOpenAI: Configured Azure OpenAI instance

        Raises:
            ValueError: If required Azure configuration is missing
        """
        # Get Azure OpenAI configuration
        api_key = config.get("azure_api_key")
        api_endpoint = config.get("azure_endpoint")
        deployment_name = config.get("azure_deployment")
        api_version = config.get("azure_api_version", "2024-08-01-preview")
        temperature = config.get("temperature", 0.3)

        # Validate required configuration
        missing_config = []
        if not api_key:
            missing_config.append("AZURE_OPENAI_API_KEY")
        if not api_endpoint:
            missing_config.append("AZURE_OPENAI_ENDPOINT")
        if not deployment_name:
            missing_config.append("AZURE_OPENAI_DEPLOYMENT_NAME")

        if missing_config:
            raise ValueError(
                f"Missing required Azure AI Foundry configuration: {', '.join(missing_config)}. "
                "Please configure these in Azure AI Foundry and set the corresponding environment variables. "
                "Visit https://ai.azure.com/ to set up your Azure AI Foundry project and deployments."
            )

        logger.info("Creating Azure AI Foundry LLM with deployment: %s", deployment_name)

        return AzureChatOpenAI(
            azure_deployment=deployment_name,
            openai_api_version=api_version,
            azure_endpoint=api_endpoint,
            api_key=api_key,
            temperature=temperature,
        )

    @staticmethod
    def get_provider_info(config: Optional[dict] = None) -> dict:
        """
        Get information about the configured AI provider.

        Args:
            config: Optional configuration dictionary

        Returns:
            dict: Information about the provider (name, model, etc.)
        """
        config = config or {}

        return {
            "provider": "azure",
            "deployment": config.get("azure_deployment"),
            "endpoint": config.get("azure_endpoint"),
            "api_version": config.get("azure_api_version"),
            "model": config.get("model"),
        }

class LLMConfig:
    """
    Configuration class for LLM settings.
    Provides a structured way to configure LLM instances.
    """

    def __init__(
        self,
        ai_provider: str = "azure",
        temperature: float = 0.3,
        azure_api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        azure_api_version: str = "2024-08-01-preview",
    ):
        """
        Initialize LLM configuration.

        Args:
            ai_provider: AI provider to use ("openai" or "azure")
            temperature: Temperature for response generation (0.0-1.0)
            azure_api_key: Azure OpenAI API key
            azure_endpoint: Azure OpenAI endpoint URL
            azure_deployment: Azure OpenAI deployment name
            azure_api_version: Azure OpenAI API version
        """
        self.ai_provider = ai_provider
        self.temperature = temperature
        self.azure_api_key = azure_api_key
        self.azure_endpoint = azure_endpoint
        self.azure_deployment = azure_deployment
        self.azure_api_version = azure_api_version

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "ai_provider": self.ai_provider,
            "temperature": self.temperature,
            "azure_api_key": self.azure_api_key,
            "azure_endpoint": self.azure_endpoint,
            "azure_deployment": self.azure_deployment,
            "azure_api_version": self.azure_api_version,
        }

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Create configuration from environment variables."""
        return cls(
            ai_provider=os.getenv("AI_PROVIDER", "azure").lower(),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
            azure_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            azure_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        )
