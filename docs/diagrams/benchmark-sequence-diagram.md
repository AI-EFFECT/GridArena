# Benchmark Sequence Diagram

Request flow shared by all four benchmark modules (Phase Identification,
Topology Discovery, State Estimation, Voltage Control): a participant
downloads noise-corrupted training data and a submission template, then
uploads a script that GridArena runs on hidden test data and scores against
ground truth the participant never sees. See
[API Reference › Code submissions](../api/index.md#code-submissions) for the
sandbox and limits, and
[API Reference › Overview](../api/index.md#difficulty-and-noise-profiles)
for the exact difficulty/noise profile values.

```mermaid
sequenceDiagram
    actor Participant
    participant Router as Benchmark Router
    participant Runner as Submission Runner
    participant Sandbox as Sandbox (harness.py + script)
    participant DB as PostgreSQL

    Participant->>Router: GET /{benchmark}/training-...?difficulty=...
    Router->>DB: fetch training grids' data
    Router-->>Participant: noisy training data with labels
    Participant->>Router: GET /{benchmark}/submission-template
    Router-->>Participant: blank script with predict()

    Participant->>Router: POST /{benchmark}/submissions (script + settings)
    Router->>Runner: validate upload, queue run
    Router-->>Participant: 202 submission_id (queued)

    Runner->>DB: fetch test grids' data + ground truth
    Runner->>Runner: build inputs (seeded noise, random keys), keep truth
    Runner->>Sandbox: run predict() once per input
    Sandbox-->>Runner: outputs (+ log)
    Runner->>Runner: score outputs vs ground truth, delete inputs

    Participant->>Router: GET /{benchmark}/submissions/{submission_id}
    Router-->>Participant: status, score, log

    Note over Runner,Sandbox: Ground truth and real identifiers never reach the script
```
