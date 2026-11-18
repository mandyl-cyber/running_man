# MM 0003 fixed-start causal evaluation

同一 E7 checkpoint、同一 MM reference、clean deterministic evaluation；只改变精确 reset frame。

| segment | start frame | episodes | success | completion | mean last frame | anchor/joint/other failure |
|---|---:|---:|---:|---:|---:|---:|
| full_clip | 0 | 100 | 0.0% | 0.3054 | 97.7 | 100.0%/0.0%/0.0% |
| mm_approach | 71 | 100 | 0.0% | 0.3745 | 119.8 | 100.0%/0.0%/0.0% |
| skill_recovery | 104 | 100 | 100.0% | 1.0000 | 320.0 | 0.0%/0.0%/0.0% |

## Interpretation

- `mm_clean_0003` is an imported/ranked candidate identifier, not a paired transform of original `0003`; its manifest provenance points to a different generated-pool trajectory.
- start 104 covers the complete skill and recovery suffix.
- start 71 covers the replaced MM approach suffix before entering the same skill.
- If start 104 passes while starts 0/71 fail, the executable skill/recovery is not the primary blocker; the causal blocker is the MM approach and the state it delivers to skill entry.
