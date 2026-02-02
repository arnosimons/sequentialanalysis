# sequentialanalysis

sequentialanalysis is a self-contained Python 3.10 module for automating sequential analysis in the German tradition of Objektive Hermeneutik. It translates the sequential, step-wise procedure described by Wernet ([2009]([url](https://doi.org/10.1007/978-3-531-91729-0))) into a controlled and inspectable pipeline that processes a text sequence by sequence. Across iterative rounds, it generates alternative readings, confronts them with emerging inner context, and incrementally builds and revises a case structure hypothesis (Fallstrukturhypothese) from traceable intermediate outputs.

The module exposes an `analyze` function that iterates through a segmented protocol (a list of sequences) and produces an auditable trace for each round. Each round is split into three stages using chained model calls via the OpenAI Responses API with typed output schemas:

- Stage 1 (`tell_stories`): generate multiple plausible everyday situations in which the fragment would occur verbatim and make sense, without any case context. This deliberately opens up interpretive space and surfaces an everyday understanding of the sequence before any contextual narrowing.
- Stage 2 (`form_readings`): consolidate the situation set into a small number of context-lean meaning types by extracting what the scenarios share structurally. For each reading, the pipeline specifies the fragment’s typical function within the scenario, links the reading back to the situations that exemplify it, and identifies a best-fitting example. The result is a compact set of contestable candidate meanings that remain independent of the concrete case, ready to be tested in Stage 3.
- Stage 3 (`confront_with_context`): test the candidate readings against the inner context (the previously analyzed sequences) and the outer context (basic case knowledge, for example that the material is an interview), track what is expected versus surprising at each step, and update the running case structure hypothesis while keeping viable alternatives open rather than closing them too early. It also generates explicit predictions about plausible next sequences under each reading, so that emerging expectations can be checked against what actually follows in the protocol. Optionally, expert knowledge can be provided to nudge the model toward specific interpretion perspectives.

All prompts are stored as external text files and loaded at runtime, and every intermediate product plus response metadata is logged in structured JSON. This design makes the analysis process reproducible and easier to debug, since you can see whether issues originate in situation generation, reading formation, or context confrontation.

## Highlights

- **Three-stage pipeline**: telling stories, forming readings, and context confrontation.
- **Auditable trace**: structured outputs per round with response metadata.
- **Externalized prompts**: prompt files live in `_prompts/` and are loaded at runtime.
- **Configurable runs**: tune model, temperature, and reasoning settings via `SequenzAnalyseConfig`.

## Installation

Open Terminal (macOS or Linux) or PowerShell / Command Prompt (Windows) and run:

```bash
python -m pip install "git+https://github.com/arnosimons/sequentialanalysis.git"
```

## Basic use case (copy/paste)

Paste this into a Python file, a notebook, or the Python REPL:

```python
from sequentialanalysis import SequentialAnalysis, SequentialAnalysisConfig
from sequentialanalysis import save_as_json

from openai import OpenAI

client = OpenAI(api_key=YOUR_OPENAI_API_KEY)

# A minimal protocol split into sequences (each entry is one segment).
sequences = [
    "A: Was macht das Modell?",
    "B: Es erzählt uns Geschichten.",
    "A: Und der Kontext?",
    "B: Kommt erst später!"
]

# The outer context (case knowledge) used in stage 3.
outer_context = "Ein Dialog zwischen A und B."

# Pick OpenAI model
config = SequentialAnalysisConfig(
    model="gpt-5-nano",
)

analyzer = SequentialAnalysis(
    client=client, 
    config=config, 
)

# Run the full analysis
result = analyzer.analyze(
    sequences, 
    outer_context, 
)

# Optional: provide expert knowledge:
expert_context = (
    "Sprachmodelle können Texte generieren, die wie Geschichten klingen. "
    "Man muss nur aufpassen, welchen Kontext man ihnen dabei gibt."
)
result = analyzer.analyze(
    sequences, outer_context, 
    expert_context=expert_context,
    expert_context_enforcement="high",
)

# The full result is a nested dict with all rounds and stages.
print(result.data["rounds"][0].keys())

# Save the results
out_path = save_as_json(
    result.data, 
    output_dir="out", 
)
```

## Requirements

- Python 3.10
- OpenAI Python SDK (`openai`) and a valid `OPENAI_API_KEY` environment variable

```
export OPENAI_API_KEY="your-key-here"
```
