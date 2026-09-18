# IIRA 2.0 Implementation Plan Suite

**Approved design:** `docs/superpowers/specs/2026-08-27-iira2-design.md`

## Execution Order

1. `2026-08-27-iira2-foundation-assets-plan.md`
   - Repository, typed config, compact status, asset manifests/downloads, target probe, offline bundle boundary, legacy read-only audit.
   - Exit: host foundation tests pass; storage/environment evidence exists; RAD-DINO/Qwen assets are `VERIFIED`, `PARTIAL`, or explicitly `FAILED`.
2. `2026-08-27-iira2-data-qwen-plan.md`
   - Restricted-data audits, ontology/splits/DICOM, duplicate/provenance audit, frozen Qwen adapter, probability/calibration, direct/SelfCheck caches.
   - Exit: Gate 1 inputs and a deterministic direct-Qwen cache exist; unavailable restricted data remains explicit.
3. `2026-08-27-iira2-kbcsv2-plan.md`
   - RAD-DINO sensor, classifier/localizer, calibration/reliability, freeze seal, evidence cache, evidence-conditioned Qwen revision cache.
   - Exit: Gate 2 result and immutable evidence/revision caches exist.
4. `2026-08-27-iira2-macro-controller-plan.md`
   - Three-action environment, fixed baselines, Supervised Router, Contextual Bandit, PPO-Options, terminal Macro-CISPO.
   - Exit: all four P0 algorithms pass synthetic update/evaluation parity and freeze audits.
5. `2026-08-27-iira2-evaluation-ascend-plan.md`
   - Metrics/statistics, interventions/stress, experiment locks/gates, compact reports, Ascend profiles, offline acceptance.
   - Exit: source/host/NPU/offline status is evidence-backed; locked VinDr evaluation runs only after Gates 0-5 permit it.

## Cross-Plan Contracts

```text
ProjectPaths + ResolvedConfig + RunSummary
    -> SampleRecord + Pathology + AgentResponseCache
    -> ExternalEvidence + EvidenceCache + revision AgentResponseCache
    -> ControllerObservation + MacroTransition + ControllerAlgorithm
    -> PredictionRecord + GateDecision + OfflineAcceptance
```

- Later plans may extend earlier modules only through an explicit `Modify:` declaration.
- All caches are content-addressed by upstream config/model/data/preprocessing/calibrator identities.
- Exchange outputs never accept internal row objects directly.
- A later phase cannot reinterpret `PARTIAL`, `BLOCKED`, or failed gates as success.

## Design Traceability

| Design sections | Implemented by |
|---|---|
| 1-3 goals, paths, architecture | Foundation plus all component plans |
| 4-5 freeze boundary and Qwen | Data/Qwen; Macro Controller freeze audits |
| 6 KBCSv2 | KBCSv2 |
| 7-8 data, labels, preprocessing | Data/Qwen |
| 9-10 belief, actions, reward | Macro Controller |
| 11 four P0 algorithms | Macro Controller |
| 12 non-P0 boundary | Experiment registry rejects them from P0 |
| 13-15 matrix, metrics, gates | Evaluation/Ascend |
| 16 Ascend 910C | Foundation probe; Evaluation/Ascend runtime |
| 17 downloads and manifests | Foundation/Assets |
| 18 switches and compact output | Foundation config/status; Evaluation reporting/launcher |
| 19 offline acceptance | Evaluation/Ascend |
| 20 tests | Every plan |
| 21 execution order | This index and phase exits |
| 22 readiness states | Foundation status; Evaluation acceptance |
| 23 frozen decisions | Global constraints in every plan |

## Stop Conditions

- Do not start large Qwen download without storage plan plus 20% headroom.
- Do not build an NPU wheelhouse before the target runtime probe resolves exact compatibility.
- Do not train KBCSv2 when required authorized data is absent.
- Do not generate evidence or revision caches before the KBCSv2 freeze seal.
- Do not run large Macro-CISPO after a failed complementarity gate.
- Do not run VinDr test before the experiment lock is hash-sealed.
- Do not claim `OFFLINE_READY` from host-only, simulated, or network-enabled evidence.
