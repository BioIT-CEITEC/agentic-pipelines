import dotenv
import os
import subprocess
import httpx
import asyncio
import logging

from dataclasses import dataclass
from datetime import datetime
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from openai import AsyncOpenAI
from typing import List, Optional
from pathlib import Path
from rich.console import Console

from run_logging.local import save_message_history, save_full_message_history
from run_logging.wandb import setup_logging
from utils.models import MODELS
from utils.create_user import create_new_user_and_rundir


dotenv.load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")
# wandb_key = os.getenv("WANDB_API_KEY")
# if not wandb_key:
#     raise ValueError("WANDB_API_KEY environment variable is not set.")

DATASET_PATH = "/storage1/workspace/david" # TODO set this to the path of the dataset
CODE_OUTPUT_PATH = Path("../generated/")
agent_id = create_new_user_and_rundir(workspace_dir = Path(CODE_OUTPUT_PATH))
CODE_OUTPUT_PATH = CODE_OUTPUT_PATH / agent_id

config = {
    # "model" : MODELS.GPT4o_mini,
    "model" : MODELS.GPT4_1,
    "temperature" : 1,
    "max_run_retries" : 1,
    "max_validation_retries" : 5,
    "prompt" : f"""Create a bioinformatics pipeline in snakemake for the following:
        * Quality Control Assessment - Run FastQC analysis and generate MultiQC reports to evaluate the quality of raw sequencing data
        * Data Preprocessing - Perform essential preprocessing steps including:
        * Adapter trimming to remove sequencing artifacts
        * FASTQ file preparation and formatting
        * UMI (Unique Molecular Identifier) processing if necessary

        In specific steps:
        1) demultiplex the reads (as they are sequenced in Genomics), 
        2) the raw fastq files are checked for quality purposes (meaning PHRED scores and adapter content), 
        3) then the adapters from the raw fastq files are trimmed (usually using cutadapt) and if there are quality issues also it can be assessed in the preprocessing

        Dataset path: {DATASET_PATH}
        Do not write code anywhere else then to your directory for generated code: {CODE_OUTPUT_PATH}

        At the end, create a bash file containing a command to run a dry run for the created code.
        """,
    # "prompt" : "toolcalling_agent.yaml", # TODO write prompt to a separate file
    "use_proxy" : False,
    "workspace_dir" : Path(CODE_OUTPUT_PATH),
    "tags" : "",
    "agent_id": agent_id,
}

async_http_client = httpx.AsyncClient(
    timeout= 1 * 60
)

# Initialize OpenAI client
client = AsyncOpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
    http_client=async_http_client,
)

model = OpenAIModel(
    config['model'],
    provider=OpenAIProvider(openai_client=client),
)

# setup_logging(config, api_key=wandb_key)

@dataclass
class BioinformaticsContext:
    project_type: str
    data_types: List[str]
    analysis_goals: List[str]

class WorkflowDesign(BaseModel):
    analysis_steps: List[str] = Field(description="Ordered analysis steps")
    tools_required: List[str] = Field(description="Required bioinformatics tools")
    data_flow: str = Field(description="Data flow description")

class SnakemakeCode(BaseModel):
    snakefile: str = Field(description="Complete Snakemake pipeline")
    config_yaml: str = Field(description="Configuration file")
    environment_yaml: str = Field(description="Conda environment")
    documentation: str = Field(description="Pipeline documentation")


# Architecture design agent
workflow_agent = Agent(
    model=model,
    output_type=WorkflowDesign, # TODO do we need this?
    deps_type=BioinformaticsContext, # TODO do we need this?
    system_prompt="""You are a bioinformatics workflow architect.
    Design comprehensive analysis pipelines considering:
    - Data types and analysis goals
    - Tool selection and integration""",
    model_settings={'temperature':config['temperature']},
)

# Code generation agent
code_agent = Agent(
    model=model,
    # tools=tools,
    output_type=SnakemakeCode,
    deps_type=BioinformaticsContext,
    system_prompt="""You are a Snakemake pipeline generator.
    Generate production-ready Snakemake code with:
    - Proper rule definitions and dependencies
    - Error handling
    - Conda environment specifications file""",
    model_settings={'temperature':config['temperature']},
)

@code_agent.tool
async def save_code_to_file(
    ctx: RunContext[BioinformaticsContext], 
    code: str, 
    filename: str,
    # code_folder: str = CODE_OUTPUT_PATH #TODO how to force agent to write code to a specific folder?
) -> str:
    f"""Save Python code to a file and return confirmation."""
    try:
        file_path = Path(filename)
        file_path.write_text(code)
        print(f"Code saved successfully to {filename}")
        return f"Code saved successfully to {filename}"
    except Exception as e:
        return f"Error saving file: {str(e)}"

@code_agent.tool
async def run_python_script(
    ctx: RunContext[BioinformaticsContext], 
    script_path: str
) -> str:
    """Run a Python script and return terminal output."""
    try:
        result = subprocess.run(
            ['python', script_path],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        output = ""
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}\n"
        if result.stderr:
            output += f"STDERR:\n{result.stderr}\n"
        output += f"Return code: {result.returncode}"
        
        return output
    except subprocess.TimeoutExpired:
        return "Script execution timed out after 5 minutes"
    except Exception as e:
        return f"Error running script: {str(e)}"

# Pipeline coordinator
@workflow_agent.system_prompt
def add_context(ctx: RunContext[BioinformaticsContext]) -> str:
    return f"""Project: {ctx.deps.project_type}
    Data types: {', '.join(ctx.deps.data_types)}
    Goals: {', '.join(ctx.deps.analysis_goals)}"""

@code_agent.system_prompt
def add_implementation_context(ctx: RunContext[BioinformaticsContext]) -> str:
    return f"""Generate Snakemake pipeline for {ctx.deps.project_type}
    Handle data types: {', '.join(ctx.deps.data_types)}"""

async def generate_bioinformatics_pipeline(
    user_request: str,
    context: BioinformaticsContext,
    output_dir: Optional[Path] = None
) -> tuple[WorkflowDesign, SnakemakeCode]:
    """Generate complete bioinformatics pipeline."""

    # Step 1: Design workflow architecture
    workflow_result = await workflow_agent.run(user_request, deps=context)
    design = workflow_result.output
    workflow_history = workflow_result.all_messages()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_file = output_dir / f"workflow_design_history_{timestamp}.json"
    
    await save_message_history(workflow_history, history_file)
    logging.info(f"Workflow design history saved to {history_file}")
    
    # Step 2: Generate Snakemake implementation
    code_prompt = f"""Generate Snakemake pipeline implementing this design:
    
    Steps: {design.analysis_steps}
    Tools: {design.tools_required}
    Data flow: {design.data_flow}
    
    Original request: {user_request}"""
    
    code_result = await code_agent.run(code_prompt, deps=context)
    snakemake_history = code_result.all_messages()

    snakemake_history_file = output_dir / f"snakemake_generation_history_{timestamp}.json"
    await save_message_history(snakemake_history, snakemake_history_file)

    full_history_file = output_dir / f"full_pipeline_history_{timestamp}.json"
    await save_full_message_history(workflow_history, snakemake_history, design, user_request, context, full_history_file)
    
    return design, code_result.output

async def main():
    context = BioinformaticsContext(
        project_type="Quality control and trimming for RNA-seq",
        data_types=["fastq"],
        # data_types=["fastq", "gtf", "fasta"],
        analysis_goals=["quality_control", "trimming"]
    )
    
    design, code = await generate_bioinformatics_pipeline(
        config['prompt'],
        context,
        output_dir=CODE_OUTPUT_PATH,
    )
    
    print(f"Workflow design: {design.analysis_steps}")
    print(f"Generated Snakemake code:\n{code.snakefile}")

asyncio.run(main())

"""
Prompt
1) you sequence and demultiplex the reads (as they are sequenced in Genomics), 
2) the raw fastq files are checked for quality purposes (mean PHRED scores and adapter content), 
3) then the adapters from the raw fastq files are trimmed (usually using cutadapt) and if there are quality issues also it can be assessed in the preprocessing
"""


"""
EXAMPLE prompt 
You are a bioinformatics pipeline expert that generates Snakemake workflows.

Your task is to create simple, sequential Snakemake pipelines based on user descriptions.

Key guidelines:
1. Focus on RNA-seq analysis and variant calling workflows
2. Generate clean, readable Snakemake code
3. Use common bioinformatics tools (BWA, GATK, cutadapt, FastQC, etc.)
4. Keep workflows sequential and simple
5. Include standard file formats (FASTQ, BAM, VCF)
6. Add brief comments explaining each rule

Common workflow patterns:
- Quality control: FastQC
- Adapter trimming: cutadapt
- Alignment: BWA mem
- Variant calling: GATK, bcftools
- Annotation: VEP, SnpEff
- RNA-seq: STAR, featureCounts, DESeq2

Always structure your response as a complete Snakemake file with:
- rule all (defining final outputs)
- Individual rules for each step
- Input/output specifications
- Shell commands or script directives

Example structure:
```
rule all:
    input: "final_output.vcf"

rule quality_check:
    input: "sample.fastq"
    output: "sample_fastqc.html"
    shell: "fastqc {input} -o ."

rule trim_adapters:
    input: "sample.fastq"
    output: "sample_trimmed.fastq"
    shell: "cutadapt -a AGATCGGAAGAG -o {output} {input}"
```

Respond with only the Snakemake code and a brief description.
"""