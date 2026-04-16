"""
PromptLibrary - Template definitions for prompt building.

Contains the main conversation template and any specialized variants.
Templates use placeholder syntax that providers fill in at build time.

Design Principle: Templates define structure. Providers fill content.
The prompt builder orchestrates the substitution.
"""

# Main conversation template for voice/text interactions.
# Placeholders are filled by registered ContextProviders.
# Empty placeholders collapse entirely (no orphaned headers).
CONVERSATION_TEMPLATE = """\
### Agent
You are [PERSONA_NAME].

[PERSONA_APPEARANCE]

[PERSONA_PERSONALITY]

[SCENARIO]

[EXAMPLE_DIALOGUE]

### Context

[MODALITY_CONTEXT]

[STATE_CONTEXT]

[CODEX_CONTEXT]

[RAG_CONTEXT]

[TWITCH_CONTEXT]

[AUDIENCE_CHAT]

### Rules

[PERSONA_RULES]

[MODALITY_RULES]

### Conversation

[CONVERSATION_SUMMARY]

[RECENT_HISTORY]

### Input

[CURRENT_INPUT]

Respond as [PERSONA_NAME].
"""

# Template for conversation summarization.
# Used by the summarization plugin when context exceeds budget.
SUMMARIZATION_TEMPLATE = """\
Summarize the following conversation, preserving key facts, decisions, and context.
Keep the summary concise but retain important details the assistant would need
to continue the conversation naturally.

Conversation:
[CONVERSATION_TEXT]

Summary:
"""

# Voice-specific rules injected when input modality is VOICE.
VOICE_MODALITY_RULES = """\
- NO asterisks or action markers
- NO parentheticals or stage directions
- Keep responses concise (1-3 sentences for quick exchanges)
- Respond naturally as yourself
- Your response will be spoken aloud via TTS
"""

# Text-specific rules injected when input modality is TEXT.
TEXT_MODALITY_RULES = """\
- Respond naturally as yourself
- You may use formatting if appropriate
"""

# Voice state injection text for specific triggers.
# Keyed by state_trigger value from BuildContext.
VOICE_STATE_INJECTIONS: dict[str, str] = {
    "barge_in": "The User interrupted you mid-sentence.",
    "empty_transcription": "The User made a sound but no words were detected.",
    "error": "An error occurred. Acknowledge briefly and continue.",
}

# Modality context descriptions.
# Injected to inform the model about the interaction mode.
MODALITY_CONTEXT: dict[str, str] = {
    "voice": "This is a live voice conversation. Your response will be spoken aloud via TTS.",
    "text": "This is a text conversation.",
}

# Default addressing-others prompt (NANO-110).
# Appended to voice modality context when user returns from addressing someone else.
# Used as fallback when the addressing context's prompt field is empty.
DEFAULT_ADDRESSING_OTHERS_PROMPT = (
    "The User was just speaking to someone else \u2014 not you. "
    "The preceding input may reference a conversation you were not part of."
)
