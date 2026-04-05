
from langgraph.graph import MessagesState

class MeetingAgentState(MessagesState):
    processed: bool = False   # runtime.state — was transcript analyzed this session?
    stored: bool = False      # runtime.state — was result saved to store?
    remaining_steps: int = 20
