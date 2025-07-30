import wandb
from dataclasses import asdict

def setup_logging(config, api_key, dir="/tmp/wandb"):
    wandb.login(key=api_key)
    wandb.init(
        dir=config.workspace_dir / dir,
        entity="bioit-ai",
        project="agentic-pipelines",
        tags=config.tags,
        config=asdict(config),
        name=config.agent_id,
    )