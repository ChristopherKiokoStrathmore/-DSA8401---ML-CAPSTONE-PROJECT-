# Label rubric

Names were checked on 18 Nov 2026 against [MULTI-HEAD-](https://github.com/ChristopherKiokoStrathmore/MULTI-HEAD-).

`lib/types.ts` types each prediction as three strings (`issue`, `sentiment`, `urgency`). It does not list the class names. `lib/format.ts` only changes underscores to spaces for display. `lib/urgency.ts` maps the urgency string onto `low`, `medium`, or `emergency`. The ten issue names and the sentiment pair are the strings in `eval/README.md` ("Known issue classes"). This repo uses those strings unchanged.

## Issue (10 classes)

Source file: `issue_retagged_clean_1500.csv`, loaded as `issue.csv` (`text`, `label`). A byte-identical second copy of that file exists in the raw export. Drop the copy. Do not load it.

Provenance: retagged by content (Chris). Counts below are the label-file schema counts. `python -m capstone.splits` recomputes them when the real directory is passed. They were not recomputed in this environment.

| Label | Schema count | What it covers |
| --- | ---: | --- |
| Non_Actionable | 384 | No request to act on: thanks, greetings, or a message with no service ask |
| Network_Issues | 274 | Voice or data service failing: drops, no signal, outage |
| Data_Bundle_Problems | 146 | A bought bundle or data balance did not load or expired early |
| Router_WiFi_5G | 143 | Home router, Wi-Fi, or 5G equipment |
| SIM_Line_Services | 134 | SIM, line activation, PUK, or number services |
| Product_Enquiry | 133 | A question about offers, tariffs, or how a product works |
| Airtel_Money_Transfer | 119 | Sending or receiving Airtel Money, till, or paybill |
| App_Rewards | 97 | The Airtel app, login, or rewards balance |
| Complaint_General | 39 | A complaint that does not fit a more specific class |
| Airtel_Money_Reversal | 31 | A request to reverse a money transfer |

These definitions are the working rules for this repository, written from the class names. They are not a separate annotation manual from Chris. If his guide differs, his guide wins and this table should be edited.

Split: grouped by `dup_group_id`, stratified 70/15/15, seed 42. Gold rows (below) are removed from train and validation after the split. They can remain in the issue test split.

## Sentiment (binary)

| Label | Meaning |
| --- | --- |
| negative | The customer is dissatisfied, complaining, or asking for a fix |
| not_negative | Neutral or positive. This is not a three-way positive class |

Primary labels: `sentiment_judgment_1500.csv`, loaded as `sentiment_hand.csv`. Same texts as the issue file. Schema counts: negative 853, not_negative 647.

Platform-derived labels: `sentiment_binary_1500.csv`, loaded as `sentiment_platform.csv`. Built from the workbook's noisy `Sentiment` column. Schema counts: negative 702, not_negative 798. This file is not a training source. It is only the agreement comparison.

Test: `sentiment_test_150.csv`, loaded as `sentiment_test.csv` (`row_id`, `message`, `your_sentiment`). 150 hand-annotated rows, negative 93, not_negative 57. The schema says this set is exact-disjoint from the 1,500.

Train and validation are the hand file after gold removal, split in the 70:15 ratio (the non-test part of 70/15/15), seed 42. The 150 is the test set and is not carved out of the 1,500.

### Agreement

Recomputed on the real files, as stated in the label-file schema (18 Nov 2026), platform-derived against hand:

| | value |
| --- | ---: |
| paired texts | 1500 |
| raw agreement | 0.711 |
| Cohen's kappa | 0.428 |

`python -m capstone.agreement --labels <dir>` is the recomputation. It joins on normalised masked text and prints agreement and kappa. This environment does not have the real files, so the two figures above were not re-run here. A smoke run prints a harness check on the synthetic files and must not be quoted as this table.

## Urgency (ordinal)

`lib/urgency.ts` uses `low`, `medium`, and `emergency`. The files store a number. The loader maps it:

| `urg` | Label |
| ---: | --- |
| 0.0 | low |
| 0.5 | medium |
| 1.0 | emergency |

Train file: `urgency_train.csv` (`base_idx`, `text`, `urg`, `source`, `weight`). 868 rows. Schema counts: urg 1.0 = 347, 0.5 = 342, 0.0 = 179. Source: distilled (LLM silver) 688, gold 180. `weight` is the per-row training weight and is kept on the manifest. The loader accepts source `gold`, `distilled`, and `silver` (`silver` is stored as `distilled`). Any other source string fails the load.

Test file: `urgency_test.csv`, same columns. 120 rows, all gold. Schema counts: 0.5 = 52, 1.0 = 51, 0.0 = 17. This file is the urgency test. It is not reshuffled.

Validation is carved from the gold rows inside the 868 that do not exact-match or near-duplicate the gold registry. The target is about 60 rows, grouped and stratified on the urgency label, seed 42, so the later emergency threshold is chosen on gold-like labels. When a file has fewer than 60 such rows, the carve is one third, so a small file still leaves gold in train. Mixed gold/distilled duplicate groups stay in train. The rest of the 868 is train. Source and weight are copied onto every urgency manifest record.

## Gold registry

Gold is `urgency_test.csv` (120) plus `sentiment_test.csv` (150).

Every train and validation pool drops a row whose normalised masked text is an exact match or a near-duplicate of that gold. Near-duplicates use the same character 3-gram Dice groups as the workbook pipeline (`dup_group_id`). Gemini is not a training source. The overlap is still counted.

Schema-reported exact overlaps (label-file schema, 18 Nov 2026):

| Comparison | Exact overlaps |
| --- | ---: |
| urgency test in the issue/sentiment 1,500 | 9 |
| urgency test in `gemini_relabeled.csv` | 23 |
| urgency train vs urgency test | 0 |

`reports/manifests/registry.json` recomputes these, plus near-duplicate overlaps, from the directory it was given. On the synthetic sample the numbers are a harness check. On the real directory they are the recomputation. The table above is the schema citation, because the real files were not in this environment.

## What is recomputed, and what is only reported

| Figure | Status |
| --- | --- |
| Agreement 0.711 and kappa 0.428 | Recomputed on the real files according to the label-file schema. Cited here. Not re-run in this environment. The script will recompute them. |
| Class counts in the tables above | Same: schema counts from the files, cited, not re-run here |
| Exact overlaps 9, 23, and 0 | Same |
| Any number that appears only in the proposal and not in the label files | **reported** |

No other proposal figure was available to recompute. If a proposal number is not in the table above, treat it as reported.

## Loader filenames

Point `--labels` at a directory that already uses these names:

| Real export | Loader name |
| --- | --- |
| `issue_retagged_clean_1500.csv` | `issue.csv` |
| `sentiment_judgment_1500.csv` | `sentiment_hand.csv` |
| `sentiment_binary_1500.csv` | `sentiment_platform.csv` |
| `sentiment_test_150.csv` | `sentiment_test.csv` |
| `urgency_train.csv` | `urgency_train.csv` |
| `urgency_test.csv` | `urgency_test.csv` |
| `gemini_relabeled.csv` | `gemini_relabeled.csv` |

`data/sample/labels/` is a fictional stand-in with the same columns. It deliberately contains a few overlaps so removal and the leakage test have a case to fire. Those planted overlaps are not the 9 and the 23.

## Duplicate groups

Label files do not carry the workbook row id. `dup_group_id` in the manifests is computed on the label texts with the same near-duplicate function (character 3-gram MinHash, Dice at least 0.80, seed 42). Joining labels back to the workbook by masked text is later work and is not done here.

Manifests store `file`, `row_id`, `dup_group_id`, and `text_sha256`. They do not store message text. Urgency records also store `source`, `weight`, and the mapped `urgency`. `row_id` is the file's `row_id` column, else `base_idx`, else the 0-based data-row index.
