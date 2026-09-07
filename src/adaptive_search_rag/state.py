from typing import TypedDict


class RAGState(TypedDict):
    question: str
    route: str
    contexts: list[str]
    answer: str
