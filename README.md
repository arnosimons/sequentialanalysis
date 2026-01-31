# sequenzanalyse

sequenzanalyse is a self-contained Python 3.10 module for automating sequential analysis in the tradition of Objektive Hermeneutik. It translates the sequential, step-wise procedure described by Wernet (2009) into a controlled and inspectable pipeline that processes a text sequence by sequence. Across iterative rounds, it generates alternative readings, confronts them with emerging inner context, and incrementally builds and revises a case structure hypothesis (Fallstrukturhypothese) from traceable intermediate outputs.
The module exposes an `analyse` function that iterates through a segmented protocol (a list of sequences) and produces an auditable trace for each round. Each round is split into three stages using chained model calls via the OpenAI Responses API with typed output schemas:
Stage 1 (story telling): generate multiple plausible everyday situations in which the fragment would occur verbatim and make sense, without any case context. This deliberately opens up interpretive space and surfaces an everyday understanding of the sequence before any contextual narrowing.
Stage 2 (readings): consolidate the situation set into a small number of context-lean meaning types by extracting what the scenarios share structurally. For each reading, the pipeline specifies the fragment’s typical function within the scenario, links the reading back to the situations that exemplify it, and identifies a best-fitting example. The result is a compact set of contestable candidate meanings that remain independent of the concrete case, ready to be tested in Stage 3.
Stage 3 (confrontation): test the candidate readings against the inner context (the previously analyzed sequences) and the outer context (basic case knowledge, for example that the material is an interview), track what is expected versus surprising at each step, and update the running case structure hypothesis while keeping viable alternatives open rather than closing them too early. It also generates explicit predictions about plausible next sequences under each reading, so that emerging expectations can be checked against what actually follows in the protocol.
All prompts are stored as external text files and loaded at runtime, and every intermediate product plus response metadata is logged in structured JSON. This design makes the analysis process reproducible and easier to debug, since you can see whether issues originate in situation generation, reading formation, or context confrontation.

## Highlights

- **Three-stage pipeline**: situation generation, context-lean readings, and context confrontation.
- **Auditable trace**: structured outputs per round with response metadata.
- **Externalized prompts**: prompt files live in `_prompts/` and are loaded at runtime.
- **Configurable runs**: tune model, temperature, and reasoning settings via `SequenzAnalyseConfig`.

## Basic use case (copy/paste)

```python
from sequenzanalyse import analyse, SequenzAnalyseConfig

# A minimal protocol split into sequences (each entry is one segment).
sequenzen = [
    "Ich wollte eigentlich nur kurz vorbei schauen.",
    "Dann hat sich das Gespräch doch länger gezogen.",
    "Am Ende bin ich ohne Ergebnis gegangen.",
]

# The outer context (case knowledge) used in stage 3.
äußerer_kontext = "Interview about a workplace conflict."

# Optional: tweak model settings.
config = SequenzAnalyseConfig(
    model="gpt-5-nano",
    temperature=0.7,
)

result = analyse(sequenzen, äußerer_kontext, config=config)

# The full result is a nested dict with all rounds and stages.
print(result["runden"][0].keys())
```

## Requirements

- Python 3.10
- OpenAI Python SDK (`openai`) and a valid `OPENAI_API_KEY` environment variable

```
export OPENAI_API_KEY="your-key-here"
```
