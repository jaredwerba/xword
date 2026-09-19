import os

# Unit tests must not open LangSmith traces. Scale races already blew the
# monthly unique-traces cap; pytest importing tavily/jev would otherwise
# decorate with traceable() from .env.
os.environ["LANGCHAIN_TRACING_V2"] = "false"
