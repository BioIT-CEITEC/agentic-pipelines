from enum import Enum

class StrEnum(str, Enum):
    def __str__(self):
        return self.value
    
    def __repr__(self):
        return self.value

class MODELS(StrEnum):
    GPT4o="gpt-4o-2024-08-06"
    GPT4o_mini="gpt-4o-mini-2024-07-18"
    OPENROUTER_GPT4o="openrouter/openai/gpt-4"
    OPENROUTER_SONNET_37="anthropic/claude-3.7-sonnet"
    # GPT4_1="openai/gpt-4.1-2025-04-14"
    GPT4_1="gpt-4.1-2025-04-14"
    GPT4_1_mini="gpt-4.1-mini-2025-04-14"
    GEMINI_2_5="google/gemini-2.5-pro-preview-03-25"
    GPT_O4_mini="openai/o4-mini"