# Second prospective C36 time-to-verified-accuracy cohort (v2)

This cohort is registered before any of its runs. It repeats the v1 cohort ([FORMAL-TTA-20261021.md](FORMAL-TTA-20261021.md)) with three changes and nothing else. The machine-readable specification is [protocol-formal-tta-v2-20261021.json](protocol-formal-tta-v2-20261021.json).

**Why a second cohort.** The v1 cohort is not qualified (paper Section 6.13). Four of its runs passed physical acceptance at the first inspection, but all four frozen audits failed. The audit compared cancelled sums recomputed on x86 with the values saved on the ARM GPU host, under a tolerance relative to the result. The v1 outcome stays as registered. Nothing from v1 is pooled with v2.

**The three changes.**
- **Seeds.** 2013091829, 1342776288, 8730574, 1790307490 and 2126574302, from the v1 SHA-256 rule with the label `flashns-c36-formal-tta-v2-20261021`. The ten seeds observed before are excluded ([registration](formal-tta-v2-seed-registration-20261021.json)).
- **Holdout.** The new physical holdout seed is 1005993036, from the same rule.
- **Auditor.** The independent auditor is revision 2, [`box376.audit_c36_tta_v2`](src/box376/audit_c36_tta_v2.py).
  - For the four summed raw arrays it requires `|difference| <= 2*gamma_16*S` elementwise, where `S` is the sum of the absolute values of the terms. The four scalar metrics built from them get the bound implied by the triangle inequality.
  - The bound is a priori. It is not fitted to the observed v1 differences, which reach at most 0.08 of it.
  - Every other check is unchanged. Each cohort writes its audits into its own directory.

**Auditor validation (development, CPU).** Revision 2 was run end to end on the four complete v1 archives, and all four pass ([record](audit-v2-development-validation-20261021.json)). Every check after the v1 failure point therefore ran on ARM-produced data before this registration: regional metrics, all nine datasets, initial and boundary values, holdout regeneration, trajectory re-integration and the TTA timestamps. Five CPU tests ([`test_audit_tta_v2`](src/box376/test_audit_tta_v2.py)) model the failure: a 4-ulp difference in a cancelled sum fails the v1 tolerance and passes the bound, and differences beyond the bound fail. This validation does not change the v1 outcome.

**Unchanged.** Everything else is v1's:
- the eta = 1/32 instance, data and FP64 network;
- Adam4000 and dense SSBroyden;
- the new-seed checks (fresh dense parameter AD and the fixed 131-point native subset);
- the inspection policy (first inspection after 1,000 active training seconds, a second at the 4,000 s guard);
- the TTA definition, the physical gates and the success, summary and denominator rules;
- one GH200 per seed in an independent 4 h job.

The driver [`run_c36_tta_v2`](src/box376/run_c36_tta_v2.py) is the v1 driver bound to this protocol. The batch script [`start_c36_tta_v2.sbatch`](src/box376/start_c36_tta_v2.sbatch) is a template whose account and root are filled in at submission. Five audited passes qualify this cohort. Otherwise every entry is reported, with no success-only median. There is no baseline and no speedup claim.
