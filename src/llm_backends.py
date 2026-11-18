"""
Backward compatibility layer redirecting to LiveOllamaClient.
Zero canned outputs, zero heuristic fallbacks.
"""

from .llm_client import LiveOllamaClient

OllamaClient = LiveOllamaClient
