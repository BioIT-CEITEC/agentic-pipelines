import json
from datetime import datetime
from pathlib import Path
from typing import List


async def save_message_history(
    message_history: List,
    file_path: Path,
    format_type: str = "json"
) -> None:
    """Save message history to file with proper formatting."""
    
    def extract_message_data(msg, index: int) -> dict:
        """Extract data from Pydantic AI message objects."""
        if hasattr(msg, 'model_dump'):
            # Pydantic model - use model_dump()
            msg_data = msg.model_dump()
            return {
                "id": index,
                "timestamp": msg_data.get("timestamp", datetime.now().isoformat()),
                "role": msg_data.get("role", "unknown"),
                "content": str(msg_data.get("content", "")),
                "metadata": msg_data.get("metadata", {}),
                "raw_data": msg_data
            }
        elif hasattr(msg, '__dict__'):
            # Object with attributes
            return {
                "id": index,
                "timestamp": getattr(msg, 'timestamp', datetime.now().isoformat()),
                "role": getattr(msg, 'role', 'unknown'),
                "content": str(getattr(msg, 'content', str(msg))),
                "metadata": getattr(msg, 'metadata', {}),
                "type": type(msg).__name__
            }
        elif isinstance(msg, dict):
            # Dictionary
            return {
                "id": index,
                "timestamp": msg.get("timestamp", datetime.now().isoformat()),
                "role": msg.get("role", "unknown"),
                "content": str(msg.get("content", "")),
                "metadata": msg.get("metadata", {})
            }
        else:
            # Fallback for other types
            return {
                "id": index,
                "timestamp": datetime.now().isoformat(),
                "role": "unknown",
                "content": str(msg),
                "metadata": {},
                "type": type(msg).__name__
            }
    
    if format_type.lower() == "json":
        try:
            messages_data = [extract_message_data(msg, i) for i, msg in enumerate(message_history)]
            
            history_data = {
                "timestamp": datetime.now().isoformat(),
                "total_messages": len(message_history),
                "messages": messages_data
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False, default=str)
                
        except Exception as e:
            # Fallback: save as string representation
            fallback_data = {
                "timestamp": datetime.now().isoformat(),
                "total_messages": len(message_history),
                "error": f"Failed to serialize messages: {str(e)}",
                "messages_str": [str(msg) for msg in message_history]
            }
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(fallback_data, f, indent=2, ensure_ascii=False)
    
    elif format_type.lower() == "markdown":
        with open(file_path.with_suffix('.md'), 'w', encoding='utf-8') as f:
            f.write(f"# Workflow Design History\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for i, msg in enumerate(message_history):
                msg_data = extract_message_data(msg, i)
                f.write(f"## Message {i+1}\n\n")
                f.write(f"**Role:** {msg_data['role']}\n\n")
                f.write(f"**Type:** {msg_data.get('type', 'Unknown')}\n\n")
                f.write(f"**Content:**\n{msg_data['content']}\n\n")
                f.write("---\n\n")


async def save_full_message_history(
    workflow_history: List,
    snakemake_history: List,
    design: 'WorkflowDesign',
    user_request: str,
    context: 'BioinformaticsContext',
    file_path: Path
) -> None:
    """Save all agent messages without formatting."""
    
    all_messages = {
        "workflow_messages": workflow_history,
        "snakemake_messages": snakemake_history
    }
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(all_messages, f, indent=2, default=str)