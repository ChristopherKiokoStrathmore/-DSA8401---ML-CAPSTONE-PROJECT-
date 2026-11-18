# Label rubric

Names were checked on 18 Nov 2026 against [MULTI-HEAD-](https://github.com/ChristopherKiokoStrathmore/MULTI-HEAD-).

`lib/types.ts` types each prediction as three strings (`issue`, `sentiment`, `urgency`). It does not list the class names. `lib/format.ts` only changes underscores to spaces for display. `lib/urgency.ts` maps the urgency string onto `low`, `medium`, or `emergency`. The ten issue names and the sentiment pair are the strings in `eval/README.md` ("Known issue classes"). This repo uses those strings unchanged.

## Issue (10 classes)

Source file: `issue_retagged_clean_1500.csv`, loaded as `issue.csv` (`text`, `label`). A byte-identical second copy of that file exists in the raw export. Drop the copy. Do not load it.

Provenance: retagged by content (Chris). Class counts below are from the label files. The real grouped split, seed 42, after gold removal, is train 1,028 / validation 226 / test 224.

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

Test: `sentiment_test_to_annotate(in).csv`, loaded as `sentiment_test.csv` (`row_id`, `message`, `your_sentiment`). 150 hand-annotated rows, negative 93, not_negative 57. The schema says this set is exact-disjoint from the 1,500.

Train and validation are the hand file after gold removal, split in the 70:15 ratio (the non-test part of 70/15/15), seed 42. The 150 is the test set and is not carved out of the 1,500.

Real hand-file split after gold removal, seed 42: train 1,215 / validation 260. Test is the separate 150 and is not carved from the 1,500.

### Agreement

Recomputed from the real files, platform-derived against hand:

| Join | n | raw agreement | Cohen's kappa |
| --- | ---: | ---: | ---: |
| unique normalised text | 1499 | 0.711 | 0.427 |
| row-aligned | 1500 | 0.711 | 0.428 |

`python -m capstone.agreement --labels <dir>` joins on normalised masked text, which is the 1,499-row figure. A smoke run on the synthetic files is a harness check and is not this table.

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

Real split, seed 42: train 794 / validation 60 / test 120. Every validation row is gold. Exact overlap between urgency train and urgency test is 0.

## Gold registry

Gold is `urgency_test.csv` (120) plus `sentiment_test.csv` (150).

Every train and validation pool drops a row whose normalised masked text is an exact match or a near-duplicate of that gold. Near-duplicates use the same character 3-gram Dice groups as the workbook pipeline (`dup_group_id`). Gemini is not a training source. The overlap is still counted.

Exact overlaps recomputed on the real files:

| Comparison | Exact overlaps |
| --- | ---: |
| urgency test in the issue/sentiment 1,500 | 9 |
| urgency test in `gemini_relabeled.csv` | 23 |
| urgency train vs urgency test | 0 |

`reports/manifests/registry.json` recomputes these, plus near-duplicate overlaps, from the directory it is given. On the synthetic sample the numbers are a harness check. The table above is the real-file recomputation.

## Real workbook audit

`prepare_data` on the real export: 526,369 rows in, 379,575 kept. Dropped as empty: 47,004. Dropped as non-linguistic: 98,317. Dropped as auto receipts: 1,473. These are aggregate counts. No message text is stored in the audit.

## What is recomputed, and what is only reported

| Figure | Status |
| --- | --- |
| Agreement 0.711, kappa 0.427 on 1,499 unique texts, and 0.711 / 0.428 row-aligned on 1,500 | recomputed from the real label files |
| Exact overlaps 9, 23, and 0 | recomputed from the real label files |
| Split sizes issue 1028/226/224, sentiment 1215/260/150, urgency 794/60/120 | recomputed from the real manifests, seed 42 |
| Workbook 526,369 in and 379,575 kept | recomputed from the real workbook audit |
| Class counts in the tables above | counts from the label files |
| Any number that appears only in the proposal and not in these files | **reported** |

## Loader filenames

Point `--labels` at a directory that already uses these names:

| Real export | Loader name |
| --- | --- |
| `issue_retagged_clean_1500.csv` | `issue.csv` |
| `sentiment_judgment_1500.csv` | `sentiment_hand.csv` |
| `sentiment_binary_1500.csv` | `sentiment_platform.csv` |
| `sentiment_test_to_annotate(in).csv` | `sentiment_test.csv` |
| `urgency_v3_train (2).csv` | `urgency_train.csv` |
| `urgency_v3_test (2).csv` | `urgency_test.csv` |
| `gemini_relabeled.csv` | `gemini_relabeled.csv` |

The sentiment-test and urgency names include parentheses and a space. Rename them exactly. `FILE_MAP` in `src/capstone/labels.py` is the same table.

`data/sample/labels/` is a fictional stand-in with the same columns. It deliberately contains a few overlaps so removal and the leakage test have a case to fire. Those planted overlaps are not the 9 and the 23.

## Duplicate groups

Label files do not carry the workbook row id. `dup_group_id` in the manifests is computed on the label texts with the same near-duplicate function (character 3-gram MinHash, Dice at least 0.80, seed 42). Joining labels back to the workbook by masked text is later work and is not done here.

Manifests store `file`, `row_id`, `dup_group_id`, and `text_sha256`. They do not store message text. Urgency records also store `source`, `weight`, and the mapped `urgency`. `row_id` is the file's `row_id` column, else `base_idx`, else the 0-based data-row index.
