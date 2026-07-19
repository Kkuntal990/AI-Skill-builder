# HLD: Source-Grounded Capability Linking for MLEvolve — MVP

**Status:** research proposal. Build + offline runtime **implemented** as of 2026-07-18 —
schema 0.2, the build-time capability compiler (builder 2.3.0), and the sidecar capability
linker (1.1.0) are built and passing (E0/E1/deterministic-E2 gates green); the semantic
`--with-selector` replay and the gated live pilot are pending. See
[the plan](capability-linker-mvp-plan.md) §6 for live build status and §2 for the
source-verified literature positioning that supersedes/refines §3 and §18 below.

**Date:** 2026-07-15 (build-status + R18 correction added 2026-07-18)

**Scope:** minimum system and experiments needed to accept or reject the capability-linking
hypothesis before committing to a general Skill IR, graph executor, or self-improving skill
library

**Companion documents:** [builder HLD](hld.md),
[skill-shape principles](skill-shape-principles.md),
[runtime retrieval design](../eval/skill-retrieval-design.md),
[Stage 2 methodology](../eval/stage2.md), and
[MLEvolve plugin README](../../infra/agents/mlevolve/README.md)

---

## 1. Executive decision

The existing system already performs two valuable operations:

1. It turns documentation and repositories into progressive-disclosure skill packages.
2. It selects skill packages and references independently for each MLEvolve code-generation
   node.

The proposed MVP does **not** replace either system. It adds one optional build artifact and one
experimental runtime path:

```text
Existing output                              Experimental output
---------------                              -------------------
SKILL.md + references/        plus           capabilities.json
        |                                            |
        +----------------------+---------------------+
                               |
                               v
                  existing MLEvolve injection seam
                               |
               legacy package/ref selection OR
               capability-unit linking per node
```

The MVP exists to falsify the following proposition:

> Technical documentation can be compiled into source-grounded, operational ML capability units,
> and linking only the units compatible with each MLEvolve node's current search state produces
> more relevant and actionable guidance than injecting a selected skill package and its reference
> files.

End-task score is confirmatory and secondary. The primary MVP evidence is whether the capability
representation is faithful, whether the linker responds correctly to node state, and whether the
agent adopts the linked procedure without violating its applicability conditions.

No paper-level novelty claim should be made from the representation alone. Structured skill
contracts, resource-to-skill compilation, compact runtime interfaces, state-triggered skills,
hierarchical MLE skills, and graph-structured execution all have close precedents [R7–R15]. The
working research gap is narrower:

> source-grounded ML documentation capabilities linked against the internal search state of a
> graph-search MLE agent, independently at draft, improvement, debugging, and evolution nodes.

This is a hypothesis to test, not a “first” claim.

---

## 2. Research questions and hypotheses

### RQ1 — Representational capacity

Can one small capability schema encode useful guidance from heterogeneous ML systems such as PEFT,
vLLM, TransformerLens, Ray Data, and VERL without forcing everything into an experimental-action
shape?

**H1:** After the schema is frozen on four development libraries, a fifth held-out library can be
encoded without adding a new top-level field, while preserving applicability, procedure,
requirements, outputs, validation, and provenance for at least 80% of sampled workflows.

### RQ2 — Build fidelity

Can the current builder produce these units from documentation without adding unsupported claims or
requiring extensive human restructuring?

**H2:** At least 90% of sampled capability claims are supported by a cited source location, and no
more than 20% of generated fields require material human correction.

### RQ3 — State-aware linking

Does a compact capability contract make it easier to select applicable guidance than the current
package-and-reference selector?

**H3:** On historical node replay and controlled state perturbations, the capability linker reduces
irrelevant or contraindicated guidance while retaining relevant procedures and staying within the
legacy prompt budget.

### RQ4 — Independent node delivery

Does re-linking at every MLEvolve node matter, or is one task-level selection sufficient?

**H4:** Per-node linking changes selections when stage, hardware, available artifacts, or failure
state changes, and leads to more correct adoption or fewer applicability violations than freezing
the initial draft-node selection for the whole trajectory.

### Secondary question — task performance

Do the mechanism improvements translate into valid submissions, better task scores, or reduced
time-to-valid-solution?

This remains secondary in the MVP because the small number of tasks and seeds will not support a
strong general performance claim.

---

## 3. Contribution boundary and related work

The MVP must be positioned against the following neighboring systems.

| Work | Established capability | Consequence for this project |
|---|---|---|
| Skill Seekers [R7] | Converts documentation, repositories, PDFs, notebooks, and other sources into `SKILL.md`, references, and other structured knowledge assets. | “Documentation to skill” is not sufficient novelty. |
| RESOURCE2SKILL [R8] | Distills multimodal human-created resources into executable skills stored in a hierarchical Skill Wiki, with retrieval, composition, provenance, and online acquisition. | Source diversity, hierarchical storage, and resource-to-skill compilation are prior art. |
| SkillFoundry [R9] | Mines heterogeneous scientific resources into validated skill packages containing scope, I/O, execution steps, environment assumptions, provenance, and tests. | A structured scientific skill contract plus validation is not sufficient novelty. |
| Anything2Skill [R10] | Extracts external knowledge into structured contracts with invocation conditions, contraindications, actions, workflows, constraints, outputs, evidence, and confidence. | The proposed fields are engineering apparatus, not the headline contribution. |
| SkillSmith [R11] | Compiles raw skill packages offline into minimal boundary-guided runtime interfaces. | Compact compiled runtime interfaces are already explored. |
| HASP [R12] | Uses executable Program Functions that activate on runtime state and modify actions or inject context. | Generic state-conditioned skill activation is prior art. The MVP remains prompt-guidance, not an action-rewriting harness. |
| GraSP [R13] | Compiles retrieved callable skills into typed DAGs, verifies nodes, and performs bounded local graph repair. | The MVP must not claim typed skill graphs or local repair as novel and does not implement them. |
| HASTE [R14] | Organizes reusable MLE skills into global, domain, and competition tiers and reports advantages for scoped loading on MLE-Bench Lite. | Reusable skills and scoped loading for MLE agents are prior art. |
| AutoMind and DS-Agent [R15, R16] | Retrieve external knowledge or prior cases to improve data-science/MLE planning and code generation. | Knowledge retrieval for MLE agents is not new. |
| SkillsBench [R17] | Uses matched no-skill/curated-skill conditions and finds focused bundles outperform exhaustive bundles. | Paired evaluation and strict context control are required. |

### Conditional contribution if the MVP succeeds

The defensible system contribution would be the combination of:

1. **Documentation-derived ML capability units:** operational units built from package sources,
   retaining actionable conditions, validation, and fine-grained provenance.
2. **MLE artifact and resource compatibility:** linking considers model/data/checkpoint-like
   artifacts, runtime capabilities, and hardware constraints rather than semantic similarity alone.
3. **Search-node-aware delivery:** the available guidance is re-linked for draft, debug, improve,
   and evolution nodes using their different code/error/search contexts.
4. **Mechanism-level evidence:** build fidelity, selection correctness, adoption, and
   contraindication compliance are measured before aggregate task lift.

The evaluation harness is supporting infrastructure, not a primary contribution.

---

## 4. Existing-system constraints

The MVP must respect the following repository and runtime facts.

### 4.1 Builder

The current builder already performs source gathering, intent capture, structural planning,
decision-tree generation, critic/repair, deterministic Skill Contract construction, reference
synthesis, source provenance stamping, security scanning, and build-time behavioral evaluation
[R1–R3]. The capability compiler should consume those artifacts rather than introduce a second
source-gathering pipeline.

### 4.2 MLEvolve integration

The sidecar currently:

- loads a small library of `SKILL.md` packages and `references/*.md`;
- exposes a Tier-0 catalog;
- runs a temperature-zero selector for each draft, improve, debug, and evolution node;
- gives the selector the task, stage, hardware, and selected parent code/error context;
- injects selected bodies and references through the existing implementation-guideline seam;
- records selection and injected-character telemetry.

MLEvolve cannot read skill files interactively during a run, so progressive disclosure is simulated
at prompt-construction time. The new path must use the same seam [R4–R6].

> **Correction (2026-07-18, source-verified — see plan §2.5).** MLEvolve [R18] *natively* ships
> "Retrospective Memory": a curated cold-start model-guidance KB consulted at initialization plus
> per-step BM25/FAISS retrieval over its own experience, queried at planning (plan text) and
> debugging (error message). So per-node, state-conditioned *retrieval* is not novel even within
> the host agent — the defensible delta is the payload and gate (doc-derived contracted units +
> deterministic hardware/artifact compatibility filtering + contraindication-compliance eval),
> not per-node retrieval as such. **E3 must hold the native-memory configuration identical across
> all arms** (plan §4.5): whether the vendored snapshot @26bde89 enables it is an open audit item.

### 4.3 Library scale

The staged library currently contains three skills. This MVP cannot validate large-library
retrieval, distractor scaling, or hierarchical catalog search. Task-level pooling and embedding
pre-ranking remain behind their existing scale triggers in the retrieval design [R5].

### 4.4 Methodological isolation

The no-skill cell must continue to receive the same harness rules and runtime environment as the
with-skill cells. The capability path must be activated only through treatment configuration.

### 4.5 Operational safety

No Kubernetes trajectory Job may be started without explicit user approval. Offline builds,
validation, replay, and synthetic perturbation tests should precede any live run.

---

## 5. Goals and non-goals

### Goals

- Determine whether a small capability representation can cover heterogeneous ML skill families.
- Generate the representation from the current builder with source grounding.
- Select capability units using the MLEvolve node context already available to the sidecar.
- Enforce a few reliable hard compatibility constraints before semantic selection.
- Compare task-once and per-node delivery.
- Preserve the legacy path as the default and as an experimental control.
- Produce telemetry sufficient to explain a positive or negative result.

### Non-goals

- Replacing `SKILL.md` or the progressive-disclosure bundle.
- Designing a universal ontology for all ML objects.
- Executing capabilities directly as tools or programs.
- Generating or repairing a GraSP-style task DAG.
- Learning skill utility online.
- Revising the skill library automatically from trajectories.
- Solving task-level retrieval at 25–50+ skills.
- Establishing statistically definitive end-task improvements in the first pilot.
- Reworking the Stage 2 grader or attribution framework.

---

## 6. High-level architecture

```text
Documentation / repository / examples / vetted community sources
                              |
                              v
                  Existing skill-builder pipeline
       RESOLVE -> FETCH -> INTENT -> PLAN -> WRITE -> CRITIC/REPAIR
                              |
                  +-----------+-----------+
                  |                       |
                  v                       v
       SKILL.md + references/     experimental capability compiler
                                          |
                                          v
                                 capabilities.json
                                          |
                            deterministic manifest validator
                                          |
                                          v
                                  staged skill library
                                          |
                +-------------------------+-------------------------+
                |                                                   |
                v                                                   v
       legacy skill/ref selector                       capability loader/index
                                                                    |
MLEvolve node state ------------------------------------------------+
(task, stage, parent code/error, hardware, optional artifact hints) |
                                                                    v
                                                        capability linker
                                                1. hard compatibility filter
                                                2. LLM semantic selection
                                                3. dependency closure
                                                4. budgeted rendering
                                                                    |
                                                                    v
                                                  implementation guideline
                                                                    |
                                                                    v
                                                    MLEvolve code generation
                                                                    |
                                                                    v
                                                    selection/adoption telemetry
```

### Architectural principle

There are two graphs, but the MVP implements neither as a general graph engine:

- `depends_on` relationships in the manifest express a small reusable dependency closure.
- MLEvolve's existing search graph supplies changing runtime state across nodes.

The system does not compile or execute a task DAG. It selects a compact set of guidance units for
one code-generation node at a time.

---

## 7. Experimental capability artifact

### 7.1 File choice

The MVP uses `capabilities.json`, stored beside `SKILL.md`.

JSON is chosen because the current MLEvolve loader intentionally avoids a YAML dependency and
already parses only minimal `SKILL.md` frontmatter. Strict JSON also matches the builder's existing
structured LLM-output patterns. If the experiment succeeds, the durable format can be reconsidered
without changing the research result.

```text
infra/skills/<skill>/
├── SKILL.md
├── capabilities.json       # optional; experimental
├── references/
├── scripts/
└── templates/
```

Skills without `capabilities.json` continue to use the legacy path.

### 7.2 Minimal conceptual schema

```json
{
  "schema_version": "0.1",
  "skill_name": "peft-tuning",
  "source_snapshot": {
    "url": "https://github.com/huggingface/peft",
    "content_sha256": "...",
    "fetched_at": "..."
  },
  "capabilities": [
    {
      "id": "lora-single-gpu-finetuning",
      "kind": "procedure",
      "title": "Fine-tune with LoRA on one GPU",
      "summary": "Attach and train LoRA adapters while leaving base weights frozen.",
      "applicability": {
        "when": [
          "A parameter-efficient model update is acceptable."
        ],
        "avoid_when": [
          "The task requires full-parameter fine-tuning."
        ]
      },
      "requires": {
        "runtime": ["python", "gpu"],
        "hardware": {
          "min_gpu_count": 1
        },
        "artifacts": ["ml:model", "ml:training-dataset"]
      },
      "produces": ["ml:adapter-checkpoint", "ml:training-metrics"],
      "expected_outcomes": [
        "Fewer trainable parameters than full fine-tuning."
      ],
      "depends_on": [],
      "procedure": {
        "steps": [
          "Select target modules.",
          "Attach the LoRA configuration.",
          "Verify trainable parameters before training.",
          "Train and save the adapter."
        ],
        "reference_files": ["lora-methods.md"],
        "scripts": []
      },
      "validation": [
        "Only intended adapter parameters are trainable.",
        "Training loss remains finite.",
        "The saved adapter reloads successfully."
      ],
      "recovery": [
        "If target modules do not match, enumerate model module names and rebind them."
      ],
      "delivery_hints": {
        "mlevolve_stages": ["draft", "improve", "debug"]
      },
      "sources": [
        {
          "source_id": "peft-docs",
          "locator": "LoRA conceptual guide",
          "reference_anchor": "references/lora-methods.md#basic-lora"
        }
      ]
    }
  ]
}
```

The example is illustrative. The builder must derive final content from the fetched source
snapshot.

### 7.3 Capability kinds

The MVP supports four kinds only.

| Kind | Meaning | Examples |
|---|---|---|
| `reference` | A fact, constraint, compatibility rule, or choice boundary that informs another action. | tokenizer/model compatibility, supported quantization format |
| `procedure` | A bounded transformation that the agent can implement and validate. | tokenize a dataset, attach PEFT adapters, construct an evaluation harness |
| `diagnostic` | A read-only inspection or probe that produces evidence. | activation tracing, memory profiling, checkpoint validation |
| `workflow` | Coordination of several procedures or systems. | distributed training, RL post-training, serving deployment |

These kinds intentionally span factual, constructive, transformative, observational, and
orchestration knowledge. The schema must not require every unit to be an experiment or to claim a
causal effect.

### 7.4 Artifact and runtime names

Artifacts are open, namespaced strings rather than a fixed ontology:

```text
ml:model
ml:tokenizer
ml:dataset
ml:checkpoint
ml:evaluation-report
transformer-lens:activation-cache
vllm:serving-endpoint
ray:data-pipeline
```

A small common namespace is useful for composition; library-specific namespaces preserve breadth.
A new library-specific artifact type does not count as a schema change.

Runtime capabilities use a small controlled vocabulary for facts the sidecar can actually know:

```text
python
gpu
multi-gpu
multi-node
network
persistent-service
credentials
```

Declaring a requirement never grants that capability.

### 7.5 Required versus optional fields

All capabilities require:

- `id`, `kind`, `title`, and `summary`;
- at least one positive applicability condition;
- either inline procedure steps or a reference pointer;
- at least one source locator.

`procedure` and `workflow` units additionally require at least one validation condition.

`diagnostic` units require an observable output or report artifact.

`reference` units need not produce an artifact or have recovery steps.

Hardware, MLEvolve-stage hints, dependencies, scripts, and recovery guidance are optional.

### 7.6 Source grounding

For the MVP, a source pointer is valid when it identifies:

- the fetched source snapshot;
- a resolvable document/page/section locator or repository path;
- the generated reference anchor, when the capability uses a generated reference.

The compiler should prefer paraphrased claims over long copied spans. Fine-grained source-span
hashing is desirable but not required to decide whether the MVP hypothesis holds.

---

## 8. Build-time capability compiler

### 8.1 Integration point

The builder receives an experimental flag:

```text
--emit-capabilities
```

When disabled, behavior and output remain unchanged.

When enabled, capability generation occurs after the body and references exist, because it needs
the final repaired content and resolved file paths, but before final write/promote:

```text
PLAN + INTENT + SKILL CONTRACT + repaired SKILL.md + references + source metadata
                                      |
                                      v
                         one structured compiler call
                                      |
                                      v
                           deterministic validation
                                      |
                         valid         |       invalid
                           |            |          |
                           v            |          v
                  capabilities.json     |   capability experiment fails
                                        |   legacy skill may still ship
```

Capability failure must not fail an otherwise valid legacy skill during the MVP. The build report
records the two outcomes separately.

### 8.2 Compiler responsibilities

The compiler should:

1. Identify 3–7 operational units rather than converting every section.
2. Preserve the gate next to the action, consistent with the existing skill-shape rule.
3. Separate structural outputs from expected, non-guaranteed outcomes.
4. Attach source locators to each unit.
5. Reuse existing references and scripts instead of duplicating them.
6. Express prerequisites only when supported by the source or the deterministic Skill Contract.
7. Abstain from generating a unit when evidence is insufficient.

### 8.3 Deterministic validation

The experimental manifest is invalid if:

- JSON parsing or schema-version checks fail;
- capability IDs are duplicated;
- a reference or script path does not exist;
- a dependency points to an unknown capability;
- dependency edges contain a cycle;
- an executable unit lacks validation;
- an artifact or runtime entry is malformed;
- a source locator is absent;
- a delivery stage is outside the known MLEvolve stage set;
- any text fails the existing security or prompt-injection checks.

Warnings, rather than hard failures, cover:

- a unit with no recovery guidance;
- a library-specific artifact namespace;
- an applicability condition that is meaningful but not machine-checkable;
- multiple units pointing to the same reference section.

### 8.4 Human-audit artifact

For the MVP corpus, the builder should also emit or make derivable a review table containing:

```text
skill / capability / field / generated claim / source locator /
supported? / material correction? / reviewer note
```

This is an experiment artifact, not part of the runtime package.

---

## 9. Runtime node profile

The capability linker reuses existing selector context. It should not add a general state database.

```text
NodeProfile
├── stage: draft | improve | debug | evolution
├── task_text: task instruction after harness-header removal
├── hardware
│   ├── description
│   ├── gpu_count, when reliably known
│   └── vram_gb, when reliably known
├── parent, absent for draft
│   ├── code_head
│   ├── error_tail
│   └── root_cause_analysis
├── runtime_capabilities, from environment configuration
└── artifact_hints, optional and non-authoritative in the MVP
```

Artifact hints may be inferred semantically from the visible task and parent code. The MVP does not
introduce an AST/dataflow extractor. Because those hints may be wrong, they are not used as hard
filters.

Only stage, explicitly configured runtime capabilities, and reliably parsed hardware facts may
cause deterministic rejection.

---

## 10. Capability linker

### 10.1 Delivery modes

One treatment configuration selects the path:

```text
MLEVAL_SKILL_DELIVERY_MODE = legacy | capability_task | capability_node
```

An empty skill library remains the no-skill baseline.

| Mode | Behavior | Research role |
|---|---|---|
| `legacy` | Current per-node selection of skill bodies and references. | Existing-system control |
| `capability_task` | Link once at the initial draft node and reuse the same rendered bundle for every later node. | Isolates structured representation without node adaptation |
| `capability_node` | Recompute compatible units independently for every code-generation node. | Complete MVP treatment |

The production default remains `legacy` until all MVP gates pass.

### 10.2 Linking pipeline

```text
loaded capability units + NodeProfile
                  |
                  v
       A. deterministic hard filter
                  |
                  v
       B. temperature-zero LLM selection
                  |
                  v
       C. deterministic dependency closure
                  |
                  v
       D. budgeted capability rendering
```

#### A. Hard compatibility filter

Reject a unit only for facts the system knows:

- its stage allowlist excludes the current stage;
- a required configured runtime capability is absent;
- a declared GPU requirement conflicts with a reliably known CPU-only runtime;
- minimum GPU count or VRAM exceeds a reliably known value;
- a required dependency is absent or itself hard-incompatible.

Every rejection is logged with a reason code.

#### B. Semantic selection

The existing temperature-zero model selector receives compact catalog entries for remaining units
and returns:

```json
{
  "selections": [
    {
      "capability_id": "peft-tuning/lora-single-gpu-finetuning",
      "reason": "The draft must train adapters on one available GPU.",
      "reference_files": ["lora-methods.md"]
    }
  ],
  "decline_reason": ""
}
```

The selector may abstain. It must not be prompted to choose a skill when none applies.

#### C. Dependency closure

For each selected unit:

1. Add declared dependencies recursively.
2. Drop the selected unit if a dependency is missing or hard-incompatible.
3. Preserve a deterministic topological order in the rendered context.

This is a bounded manifest closure, not a task-graph compiler.

#### D. Budgeted rendering

The renderer injects a compact execution brief:

```text
## Linked ML Capabilities

### peft-tuning/lora-single-gpu-finetuning
Why linked: ...
Apply when: ...
Do not apply when: ...
Requirements: ...
Procedure:
1. ...
2. ...
Validate:
- ...
Recovery:
- ...
Source: ...
```

Default experimental limits:

- at most three selected capability units before dependency closure;
- at most one full reference file total;
- no raw manifest injection;
- no complete `SKILL.md` body in capability modes.

Exact token/character limits should be set from replay distributions before the live pilot, not
chosen solely by intuition.

### 10.3 Unknown facts

The linker uses three-valued reasoning:

- `false`: known incompatibility, so reject;
- `true`: known compatibility, so retain;
- `unknown`: do not hard-filter; present the condition to the selector and renderer.

Security, credentials, and destructive-operation requirements are exceptions: unknown means the
unit must not be represented as currently executable.

### 10.4 Failure behavior

| Failure | Behavior |
|---|---|
| Manifest absent | Legacy mode loads the skill normally. Capability modes exclude it and log `manifest_absent`; they do not mix in legacy content. |
| Manifest invalid at load | Legacy mode remains available. Capability modes exclude the skill and log the validation error. |
| Selector declines | Catalog/manifest awareness only; inject no capability. |
| Selector transport fails | Retry once, then inject nothing rather than all capabilities. |
| Dependency missing/incompatible | Drop dependent selection and log the dependency failure. |
| Renderer exceeds budget | Remove optional reference, then lowest-ranked non-dependency units. |
| Entire capability path fails | Do not break MLEvolve code generation; fall back to legacy only when the configured experimental protocol explicitly allows it. |

For clean experiments, silent fallback from a capability arm to legacy content is prohibited. A
fallback changes the treatment and must be visible in telemetry.

---

## 11. Telemetry and observability

Extend the existing append-only selection event rather than creating a second unrelated log.

Each capability-node event records:

- sidecar and capability-schema versions;
- delivery mode and MLEvolve stage;
- loaded skill and capability counts;
- known node-profile facts, excluding secrets and full source code;
- candidate capability IDs;
- hard-filtered IDs and reason codes;
- selector output and decline/error reason;
- dependency-added and dependency-dropped IDs;
- final rendered capability order;
- selected reference files;
- injected body/reference characters or tokens;
- truncation decisions;
- legacy fallback, which should normally be false.

Where stable node IDs are unavailable at selection time, retain the current time/stage join strategy
and document its limits.

### Adoption evidence

Selection is not adoption. The experiment should derive a separate post-run record:

```text
trajectory / node / selected capability / adopted? /
applicability respected? / validation used? / evidence / reviewer
```

Use deterministic probes when possible and blinded human or held-out-model review when semantic
judgment is unavoidable. The selector model must not grade its own adoption.

---

## 12. MVP experiments

Experiments run in increasing cost order. Failure at an earlier gate stops later work.

### E0 — Manual schema capacity probe

**Purpose:** test whether the representation is plausible before changing the builder.

Manually encode 3–5 representative capabilities for each of four development libraries:

| Library | Category | Representation pressure |
|---|---|---|
| PEFT | fine-tuning | conditional procedures, checkpoints, hardware gates |
| vLLM | inference serving | persistent services, runtime requirements, endpoints |
| TransformerLens | interpretability | read-only diagnostics and observation artifacts |
| Ray Data | data processing | transformations, pipelines, distributed execution |

Freeze schema `0.1`, then encode a held-out fifth library:

| Held-out library | Category | Reason |
|---|---|---|
| VERL or TorchTitan | post-training/distributed training | stresses orchestration, multi-GPU requirements, and recovery |

**Measures:**

- workflow coverage;
- information lost or forced into generic prose;
- new top-level fields requested;
- ambiguous artifact types;
- whether reference/diagnostic/procedure/workflow kinds are sufficient.

**Pass gate:**

- at least 80% of sampled useful workflows retain applicability, procedure, requirements,
  validation, and failure information;
- the held-out library needs no new top-level schema field;
- remaining loss can be represented through namespaced artifacts or optional prose fields.

If this fails, revise or abandon the schema before builder work.

### E1 — Automated build fidelity

**Purpose:** determine whether the builder can generate the cards, not merely whether a human can
write them.

For the same five libraries:

1. Run the existing builder with capability emission.
2. Sample all cards if there are at most 30; otherwise stratify by library and kind.
3. Have two reviewers label source support and material correction needs.
4. Compare generated output with the manually authored capacity probe.

**Primary metrics:**

- claim-level source-support precision;
- applicability correctness;
- operational completeness;
- validation adequacy for executable units;
- dead-reference/dependency rate;
- proportion of fields needing material correction;
- reviewer agreement;
- generation cost and latency.

**Pass gate:**

- source-support precision at least 0.90;
- applicability correctness at least 0.85;
- at least 0.80 of executable units contain a usable validation method;
- no unresolved references or dependency cycles after deterministic validation;
- at most 0.20 of fields require material correction.

If automated generation fails while E0 passes, retain the representation but improve the compiler
before runtime integration.

### E2 — Offline node replay and perturbation

**Purpose:** validate linking without GPU trajectories.

Use historical `select_skills` contexts from `prompts.jsonl` and selection telemetry. Construct a
gold capability label set for approximately 30–50 nodes, stratified across draft, improve, debug,
and evolution stages.

Compare:

1. legacy package/reference selection;
2. capability-task selection;
3. capability-node selection.

For comparison, a legacy-selected package is treated as exposing all capability units represented
in its `SKILL.md`; selected references expose the units mapped to those references. This estimates
the amount of relevant and irrelevant procedural content actually placed in context.

Create controlled paired perturbations:

- one GPU versus multiple GPUs;
- sufficient versus insufficient VRAM;
- draft versus debug stage;
- training error versus serving error;
- checkpoint absent versus present;
- tokenizer/model mismatch versus unrelated Python error;
- network/service capability available versus unavailable.

**Metrics:**

- selected-unit precision and recall;
- abstention accuracy;
- hard-constraint violation rate;
- dependency completeness;
- expected response to relevant perturbations;
- invariance to irrelevant wording perturbations;
- prompt characters/tokens;
- selector error and empty-treatment rates.

**Pass gate:**

- no increase in missed critical procedures relative to legacy selection;
- at least 30% relative reduction in irrelevant or contraindicated exposed units;
- expected selection change on at least 80% of relevant perturbation pairs;
- no material selection change on at least 80% of irrelevant perturbation pairs;
- median injected context no larger than legacy;
- no unresolved dependency enters rendered context.

If E2 fails, do not spend GPU time. Diagnose representation, state visibility, and selector failure
separately.

### E3 — Small live MLEvolve pilot

**Purpose:** test adoption and the value of independent node linking.

Start with a library containing only the audited PEFT and vLLM artifacts. Suggested tasks:

- `gsm8k` or `boolq`: training/fine-tuning guidance;
- `llama-inference`: inference and serving guidance;
- `house-prices`: unrelated control relative to the staged PEFT/vLLM library.

Run four matched arms:

| Arm | Skill content | Selection frequency | Question answered |
|---|---|---|---|
| `no_skill` | none | none | What does MLEvolve do unaided? |
| `legacy_node` | `SKILL.md` + selected references | every node | What does the current system do? |
| `capability_task` | compact capability bundle | once at draft | Does structured content help without node adaptation? |
| `capability_node` | compact capability bundle | every node | Does independent node linking add value? |

Begin with one seed: 3 tasks × 4 arms = 12 trajectories. This is a smoke and mechanism study, not
a performance estimate. If execution is correct and the mechanism metrics are promising, extend to
three total seeds: 36 trajectories.

The live run requires explicit user approval and the normal `make ab-plan` safety workflow.

**Primary outcomes:**

- selected-capability adoption in generated code;
- applicability/contraindication compliance;
- validation steps implemented or executed;
- debug selection alignment with the observed parent failure;
- difference between task-once and per-node selections;
- time or nodes to first valid solution;
- injected context and selector cost.

**Secondary outcomes:**

- final task score;
- valid-submission rate;
- wall-clock and GPU use;
- total LLM tokens and cost.

**MVP success signal:**

- `capability_node` improves adoption or reduces applicability violations over `legacy_node` on at
  least two tasks;
- `capability_node` outperforms `capability_task` on state-dependent debug/improve nodes;
- the unrelated control maintains a high abstention rate;
- there is no severe valid-submission or task-score regression;
- any apparent benefit is not explained solely by receiving more context.

The pilot is insufficient for a final paper claim. A positive result authorizes a powered follow-up;
a negative result prevents a much larger implementation commitment.

---

## 13. Decision matrix and kill criteria

| Observed result | Interpretation | Decision |
|---|---|---|
| E0 fails across heterogeneous libraries | Representation is too narrow. | Redesign or abandon the capability artifact. |
| E0 passes; E1 grounding fails | Representation is viable, compiler is not. | Improve build prompts/grounding; do not touch runtime. |
| E1 passes; E2 selection fails | Node state or linker is inadequate. | Improve state visibility/selection; do not run live jobs. |
| E2 passes; task-once equals per-node | Structured units may help, node-independent injection is unsupported. | Drop per-node novelty claim; simplify runtime. |
| Per-node selections change but are ignored | Delivery/adoption is the bottleneck. | Revise rendering or abandon prompt-only control. |
| Mechanism improves; task scores do not | Small-task noise or weak downstream value. | Run a narrowly powered follow-up only if effect is operationally meaningful. |
| Mechanism and task outcomes improve | Core hypothesis is supported. | Plan the larger compiler/linker study and library expansion. |

Stop or reconsider the MVP if any of the following occurs:

- more than 20% of generated claims are unsupported;
- the held-out library requires a fundamentally different schema;
- the linker chooses nearly identical units across meaningful state perturbations;
- dependency closure regularly increases context to legacy size;
- capability mode silently falls back to legacy in more than a trivial fraction of nodes;
- per-node delivery adds selector cost without changing adoption or recovery behavior;
- apparent gains come only from extra prompt tokens or treatment-specific harness changes.

---

## 14. Compatibility, safety, and reproducibility

### Backward compatibility

- `SKILL.md` remains required and authoritative for normal use.
- `capabilities.json` is optional.
- Legacy mode remains the default.
- Skills without manifests continue to load.
- Manifest failure cannot break MLEvolve code generation.

### Sidecar invariants

- Preserve import order and the current `sys.meta_path` patching strategy.
- Preserve the `entrypoint.sh` lifecycle and analyzer/grader outputs.
- Do not modify vendored MLEvolve unless the sidecar seam proves insufficient.
- Version every behavior-changing selector, renderer, or prompt modification.

### Security

- Reuse source IPI scanning and skill security validation.
- Never treat a declared capability as permission to access a network, credential, service, or
  destructive operation.
- Do not log secrets or full environment values.
- Keep source claims and runtime requirements separate from executable authority.

### Reproducibility

Persist for every experiment:

- source snapshot URL/hash/date;
- builder and schema versions;
- generated and reviewed capability manifests;
- selector model, temperature, and prompt version;
- node-profile facts used for linking;
- filtered, selected, dependency-added, and rendered IDs;
- delivery mode and context limits;
- task, seed, agent image, and sidecar version.

---

## 15. Expected implementation surface for detailed planning

This section identifies component boundaries, not a task-by-task implementation plan.

| Component | Expected responsibility | Compatibility requirement |
|---|---|---|
| Builder compiler stage | Optionally generate and validate `capabilities.json`. | Flag off must be behavior-identical. |
| Capability schema/validator | Structural and referential validation. | No runtime LLM dependency. |
| Skill loader extension | Load optional manifests beside existing bodies/references. | Existing skills load unchanged. |
| `capability_linker` module | Hard filtering, selector call, dependency closure, rendering. | Must never break code generation. |
| Existing injector dispatch | Select legacy/task-once/per-node path. | Preserve current patch seam and cache lifecycle. |
| Selection telemetry extension | Explain filtering, selection, closure, budget, and fallback. | Append-only and analyzer-compatible. |
| Offline replay extension | Score capability selection and perturbation behavior. | No GPU dependency. |
| Experiment configuration | Thread delivery mode and limits into manifests/jobs. | Preserve no-skill isolation. |

Likely code locations for Claude to inspect during detailed planning:

- `agents/ai-skill-builder/skills/build-skill-from-docs/scripts/skill_builder.py`
- `agents/ai-skill-builder/skills/build-skill-from-docs/scripts/prompts/`
- `infra/agents/mlevolve/mlevolve_sidecar/skill_retriever.py`
- `infra/agents/mlevolve/mlevolve_sidecar/skill_injector.py`
- `infra/agents/mlevolve/mlevolve_sidecar/selection_logger.py`
- `scripts/replay_skill_selector.py`
- `infra/agents/mlevolve/entrypoint.sh`
- `infra/agents/mlevolve/job.yaml.tmpl`
- nearby smoke and focused tests

---

## 16. Deferred questions

Do not resolve these in the MVP unless an experiment demonstrates the need:

- Should manifests become YAML or remain JSON?
- Should artifacts use a formal ontology?
- Should applicability become a restricted predicate DSL?
- Should capability dependencies form a richer graph with alternatives and conflicts?
- Should the system learn card utility from outcomes?
- Should failed trajectories revise cards automatically?
- Should the linker retrieve from hundreds of capabilities using embeddings or a hierarchy?
- Should capabilities become directly executable?
- Should a future runtime perform local graph repair?

Each question substantially increases scope and already has relevant prior art. None is required to
test H1–H4.

---

## 17. Instructions for detailed planning

The detailed implementation plan derived from this HLD should:

1. Preserve the four experiment gates E0–E3 in order.
2. Implement no live-runtime changes until E0 passes.
3. Implement no cluster experiment until E1 and E2 pass.
4. Keep all new behavior behind explicit experimental configuration.
5. Prefer additions over refactoring the existing legacy path.
6. Specify exact unit tests, fixtures, log-schema changes, and rollback behavior.
7. Separate behavior-neutral telemetry changes from treatment-changing selector/renderer changes.
8. Identify which decisions require empirical data rather than choosing defaults prematurely.
9. Keep the paper claim conditional on results and avoid “first” language.
10. Treat final task-score evaluation as validation of the mechanism, not as the system's main
    contribution.

---

## 18. References

### Local design and implementation

- **[R1]** AI-Skill Builder high-level design. [Local document](hld.md).
- **[R2]** Skill-shape principles, including the rule that a precondition travels with its action.
  [Local document](skill-shape-principles.md).
- **[R3]** Skill reliability checklist and build-time quality gates.
  [Local document](skill-reliability-checklist.md).
- **[R4]** MLEvolve sidecar and skill-injection architecture.
  [Local README](../../infra/agents/mlevolve/mlevolve_sidecar/README.md).
- **[R5]** Runtime skill-retrieval design, current N=3 constraint, telemetry, replay, and deferred
  scale triggers. [Local document](../eval/skill-retrieval-design.md).
- **[R6]** Stage 2 paired MLEvolve evaluation methodology.
  [Local document](../eval/stage2.md).

### External systems and literature

- **[R7]** Yusuf Karaaslan. *Skill Seekers: Convert documentation websites, GitHub repositories,
  PDFs, and other sources into AI skills and structured knowledge assets.*
  [GitHub](https://github.com/yusufkaraaslan/Skill_Seekers).
- **[R8]** Fan et al. *RESOURCE2SKILL: Distilling Executable Agent Skills from Human-Created
  Multimodal Resources.* arXiv:2606.29538, 2026.
  [Paper](https://arxiv.org/abs/2606.29538).
- **[R9]** Shen et al. *SKILLFOUNDRY: Building Self-Evolving Agent Skill Libraries from
  Heterogeneous Scientific Resources.* arXiv:2604.03964, 2026.
  [Paper](https://arxiv.org/abs/2604.03964).
- **[R10]** Pan et al. *Anything2Skill: Compiling External Knowledge into Reusable Skills for
  Agents.* arXiv:2606.09316, 2026.
  [Paper](https://arxiv.org/abs/2606.09316).
- **[R11]** Xu et al. *SkillSmith: Compiling Agent Skills into Boundary-Guided Runtime
  Interfaces.* arXiv:2605.15215, 2026.
  [Paper](https://arxiv.org/abs/2605.15215).
- **[R12]** Liu et al. *Harnessing LLM Agents with Skill Programs.* arXiv:2605.17734, 2026.
  [Paper](https://arxiv.org/abs/2605.17734).
- **[R13]** Xia et al. *GraSP: Graph-Structured Skill Compositions for LLM Agents.*
  arXiv:2604.17870, 2026. [Paper](https://arxiv.org/abs/2604.17870).
- **[R14]** Kim, Talebirad, and Zaiane. *Why Solve It Twice? Hierarchical Accumulation of Skills
  for Transfer-Efficient ML Engineering.* arXiv:2606.30911, 2026.
  [Paper](https://arxiv.org/abs/2606.30911).
- **[R15]** Ou et al. *AutoMind: Adaptive Knowledgeable Agent for Automated Data Science.*
  arXiv:2506.10974, 2025. [Paper](https://arxiv.org/abs/2506.10974).
- **[R16]** Guo et al. *DS-Agent: Automated Data Science by Empowering Large Language Models with
  Case-Based Reasoning.* arXiv:2402.17453, 2024.
  [Paper](https://arxiv.org/abs/2402.17453).
- **[R17]** Li et al. *SkillsBench: Benchmarking How Well Agent Skills Work Across Diverse Tasks.*
  arXiv:2602.12670, 2026. [Paper](https://arxiv.org/abs/2602.12670).
- **[R18]** Du et al. *MLEvolve: A Self-Evolving Framework for Automated Machine Learning
  Algorithm Discovery.* arXiv:2606.06473, 2026.
  [Paper](https://arxiv.org/abs/2606.06473) ·
  [code](https://github.com/InternScience/MLEvolve).
- **[R19]** Chan et al. *MLE-bench: Evaluating Machine Learning Agents on Machine Learning
  Engineering.* arXiv:2410.07095, 2024.
  [Paper](https://arxiv.org/abs/2410.07095) ·
  [code](https://github.com/openai/mle-bench).

Most 2026 works above are recent preprints. Their reported results should be treated as author
claims unless independently reproduced. The related-work boundary should be updated immediately
before paper submission.
