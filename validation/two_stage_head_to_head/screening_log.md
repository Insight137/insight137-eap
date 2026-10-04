# Screening log — two-stage decision head-to-head

Search run: 2026-10-04. Status: **provisional, not yet frozen** (see "Open before freezing").
No model was run on any held-out study during search, extraction, or checking.

## 1. What was searched

- Semantic Scholar citation lists: all 621 records citing Tversky & Shafir 1992 and all 676 citing Shafir & Tversky 1992, keyword-filtered on title and abstract, then screened.
- Web searches: about 45 queries combining "disjunction effect" with "two-stage gamble", "two-step gamble", "prisoner's dilemma", "sure-thing principle", "replication", "preregistered"; two Chinese-language queries.
- PubMed and Europe PMC keyword queries.
- Reference lists and compilation tables: Broekaert et al. 2020; Ziano et al. 2021; Pothos & Busemeyer 2009; Moreira & Wichert 2016; Huang et al. 2019; Surov et al. 2019.
- FORRT replication atlas entries for the two 1992 papers (only Ziano et al. 2021 is listed).

Coverage limits: OpenAlex and Google Scholar were unavailable, so citation screening rests on Semantic Scholar alone. The Chinese-language search was shallow (no CNKI access).

## 2. Rules applied

The four inclusion criteria are in the pre-registration, section 4.2. Applying them raised nine questions the draft did not settle. They were settled on 2026-10-04, **after the extracted values had been seen but before any model was run**. Each is a structural rule, applied the same way to every study.

**Accepted as written by Jus on 2026-10-04 (option A: all nine, no rows dropped).** Any later change to these rules must be logged here with its date and reason.

| # | Clarification |
|---|---|
| C1 | The first-stage outcome must be set externally and told to the participant. Paradigms where participants generate it themselves (categorization-decision) are out. |
| C2 | True sequential-move games, where the participant is a second mover, are out of scope. They are a different game form from a simultaneous game with leaked information. |
| C3 | Total N is enough when per-condition n is not reported. |
| C4 | A cluster is an independent sample defined by the experimental design (a separate experiment or randomised group), never by the authors' post-hoc split of participants. |
| C5 | No stimulus-level exclusions: every qualifying triple from an eligible sample is included. |
| C6 | When counts are reported, proportion = count / n. When the complement is reported (cooperation), the target rate is 1 − it. |
| C7 | Marginals obtained by summing a tabulated joint distribution, or by an n-weighted mean of tabulated subgroup values, count as tabulated. |
| C8 | A value is excluded when the source is internally inconsistent about which condition it belongs to. |
| C9 | A value known only from secondary sources is admitted provisionally if two independent secondary sources agree. It is flagged, and a pre-specified sensitivity analysis drops it. |

## 3. Included (18 independent samples, 34 triples)

Values and source locations are in `studies.csv`. "Checked" means the reviewer confirmed each number in the cited source text or page image.

| Cluster | Study | Paradigm | Design | Triples | Source | Checked |
|---|---|---|---|---|---|---|
| KKP2001-E1…E4 | Kühberger, Komunska & Perner 2001, Exps 1–4 | gamble | 3 between, 1 within | 4 | secondary (Ziano Table 1; Surov Table 1 agrees) | yes, in the secondary source |
| LB2007 | Lambdin & Burdsal 2007, coin condition | gamble | within | 1 | secondary (Ziano Table 1; publisher preview and Surov agree) | yes, in the secondary source |
| SUROV2019 | Surov et al. 2019 | gamble | within | 1 | primary | yes |
| BBP2020-E1B | Broekaert et al. 2020, Exp 1 between | gamble | between | 1 (X = 0.5) | primary, text | yes |
| BBP2020-E1W | Broekaert et al. 2020, Exp 1 within | gamble | within | 1 (X = 2) | primary, Table 2 | yes |
| BBP2020-E2KU, -E2UK | Broekaert et al. 2020, Exp 2 order groups | gamble | within | 5 + 5 | primary, Supplement Table S7 | yes; X = 2 cross-checked against Table 2 |
| ZIANO2021-B, -W | Ziano et al. 2021 | gamble | between, within | 1 + 1 | primary (accepted manuscript) | yes |
| POTHOS2011-PD, -CARD | Pothos et al. 2011 | PD, card game | within | 4 + 4 | primary (accepted manuscript), Table 1 | yes |
| HG2010-E1, -E2 | Hristova & Grinberg 2010 | PD | within | 2 + 2 | primary, text | yes |
| TESAR2018 | Tesař 2018 thesis (journal version: Decision 2020) | PD | between | 1 | primary (thesis) | yes |
| XIN2026 | Xin, Liu, Yan & Li 2026 | PD | within | 1 | primary, text | yes |

## 4. Excluded, with reason

| Study | Reason |
|---|---|
| Shafir & Tversky 1992; Croson 1999; Li & Taplin 2002; Busemeyer, Matthew & Wang 2006; Hristova & Grinberg 2008 | Development set: used to build or verify the method |
| Tversky & Shafir 1992 (two-stage gamble) | Used in an earlier Insight137 experiment |
| Mukhopadhyay, Nagaraj & Roy 2017 (arXiv:1703.00223) | C8: the text labels the "lost" figure as "did not know"; also no cell n for win/lose |
| Hashimoto et al. 2025; Watabe et al. 1996; Hayashi et al. 1999; other sequential-PD studies | C2: sequential-move game |
| Waddup et al. 2021; Wang, Busemeyer & deBuys 2022; Kvam et al. 2014; Bell et al. 2017 | Not the three-condition design |
| Li, Jiang, Dunn & Wang 2012; Wang, Li & Jiang 2012 | Rating-scale responses, not a binary choice (from abstracts and snippets) |
| Peng, Miao & Xiao 2013; Fisher, Larue & Schmidt 2025 | No unknown condition |
| Moreira & Wichert 2016 Table 2 rows; Xin et al. 2022 Table 2 rows | Secondary aggregates that match no single experiment in the primaries |
| Theory, review and modelling papers (Yukalov & Sornette; Busemeyer, Asano & Lu 2024; Gelastopoulos & Le Mens; Surov-group 2022; others) | No new human data |

## 5. Pending: full text needed before the list can be frozen

| Study | What is missing |
|---|---|
| Kühberger, Komunska & Perner 2001 (OBHDP 85) | Primary values and per-condition n for Exps 1, 2 and 4 |
| Lambdin & Burdsal 2007 (OBHDP 103) | The two die conditions (only proportions in a secondary table seen) |
| Li, Taplin & Zhang 2007 (Information Sciences 177) | "Six experiments" on the Prisoner's Dilemma; nothing beyond the abstract seen |
| Li, Wang, Rao & Li 2010 (Adaptive Behavior 18) | Gain, loss and prison-sentence framings; nothing beyond the abstract seen |
| Bagassi & Macchi 2006; Sun, Li & Li 2008 | May contain a standard gamble arm; abstract only |
| Tesař 2020 (Decision 7) | Confirm the journal article reports the same counts as the thesis |
| Broekaert et al. 2020, Exp 1 | Other payoff levels are figure-only; raw data for Exp 1 not posted |

## 6. Inconsistencies found in sources

- Broekaert et al. 2020: the Table S7 caption swaps the two risk-group totals relative to the main text; Experiment 1 sample sizes differ between the main text (118 / 114 / 94) and one supplement passage.
- Ziano et al. 2021: Table 2 gives different N for Kühberger Exps 2 and 3 than its own Table 1.
- Tversky & Shafir 1992: three secondary sources give three different values for the within-subject unknown condition (.36, .35, 34%).
- Li & Taplin 2002: 0.82 / 0.77 / 0.72 in Moreira & Wichert 2016, but 83 / 66 / 60 in Pothos & Busemeyer 2009. The development-set table uses the former.

## 7. Open before freezing

1. Obtain the pending full texts (section 5) and add, correct, or drop rows.
2. Re-run the citation screen on a second index if one is available.
3. Record the freeze date and the SHA-256 of `studies.csv` in the pre-registration.
