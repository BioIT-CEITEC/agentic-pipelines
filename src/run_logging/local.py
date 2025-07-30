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
    
    if format_type.lower() == "json":
        history_data = {
            "timestamp": datetime.now().isoformat(),
            "total_messages": len(message_history),
            "messages": [
                {
                    "id": i,
                    "timestamp": msg.get("timestamp", ""),
                    "role": msg.get("role", "unknown"),
                    "content": msg.get("content", ""),
                    "metadata": msg.get("metadata", {})
                }
                for i, msg in enumerate(message_history)
            ]
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)
    
    elif format_type.lower() == "markdown":
        with open(file_path.with_suffix('.md'), 'w', encoding='utf-8') as f:
            f.write(f"# Workflow Design History\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for i, msg in enumerate(message_history):
                f.write(f"## Message {i+1}\n\n")
                f.write(f"**Role:** {msg.get('role', 'unknown')}\n\n")
                f.write(f"**Content:**\n{msg.get('content', '')}\n\n")
                f.write("---\n\n")