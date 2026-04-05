from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver

# runtime.store  — long-term persistent memory across sessions
store = InMemoryStore()

# MemorySaver — short-term session checkpointing
checkpointer = MemorySaver()
