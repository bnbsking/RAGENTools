from dataclasses import dataclass


@dataclass
class Document:
    """Same as LangChain Document"""
    page_content: str
    metadata: dict