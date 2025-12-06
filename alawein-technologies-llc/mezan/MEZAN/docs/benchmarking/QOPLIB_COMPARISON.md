# QOPLIB Comparison

Context
- QAP is a special case of quadratic assignment; QOPLIB catalogs quadratic optimization problems more broadly.

Comparison Focus
- Data formats and loaders: ensure MEZAN’s QAPFlow adapter can be extended for QOP formats where applicable.
- Objectives: clarify when problems reduce to (A,B) forms vs requiring different backends.

Integration Strategy
- Keep QAPFlow strictly QAP; add future backends for broader quadratic forms where needed.
- Maintain separate schemas for clarity; avoid overloading inputs.

