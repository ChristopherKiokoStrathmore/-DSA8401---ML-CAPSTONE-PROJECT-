# Evaluation

> Test-split evaluation. The emergency threshold was chosen on validation only. Not a training loss.

Emergency threshold rule: On validation only: maximise emergency recall subject to false-emergency rate <= 0.20. Ties: lower false-emergency rate, then a higher threshold. If no threshold meets the cap, use the lowest false-emergency rate.

| model | threshold | val recall | val false-emergency rate | met cap |
| --- | ---: | ---: | ---: | --- |
| majority | 0.95 | 0.000 | 0.000 | True |
| keyword | 0.80 | 0.095 | 0.026 | True |
| tfidf | 0.45 | 0.714 | 0.154 | True |
| afroxlmr | 0.50 | 0.810 | 0.179 | True |

## issue

n = 224

Class support (gold n):

- Airtel_Money_Reversal: 4
- Airtel_Money_Transfer: 19
- App_Rewards: 14
- Complaint_General: 6
- Data_Bundle_Problems: 22
- Network_Issues: 41
- Non_Actionable: 57
- Product_Enquiry: 20
- Router_WiFi_5G: 21
- SIM_Line_Services: 20

| model | macro-F1 | 95% CI |
| --- | ---: | --- |
| majority | 0.041 | [0.033, 0.048] |
| keyword | 0.411 | [0.318, 0.474] |
| tfidf | 0.636 | [0.542, 0.714] |
| afroxlmr | 0.656 | [0.595, 0.707] |

Per-class F1 (n is the gold support):

| model | Airtel_Money_Reversal | Airtel_Money_Transfer | App_Rewards | Complaint_General | Data_Bundle_Problems | Network_Issues | Non_Actionable | Product_Enquiry | Router_WiFi_5G | SIM_Line_Services |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| majority | 0.00 (n=4) | 0.00 (n=19) | 0.00 (n=14) | 0.00 (n=6) | 0.00 (n=22) | 0.00 (n=41) | 0.41 (n=57) | 0.00 (n=20) | 0.00 (n=21) | 0.00 (n=20) |
| keyword | 0.67 (n=4) | 0.19 (n=19) | 0.21 (n=14) | 0.00 (n=6) | 0.59 (n=22) | 0.62 (n=41) | 0.54 (n=57) | 0.09 (n=20) | 0.77 (n=21) | 0.44 (n=20) |
| tfidf | 0.40 (n=4) | 0.74 (n=19) | 0.81 (n=14) | 0.20 (n=6) | 0.67 (n=22) | 0.81 (n=41) | 0.78 (n=57) | 0.47 (n=20) | 0.75 (n=21) | 0.73 (n=20) |
| afroxlmr | 0.00 (n=4) | 0.76 (n=19) | 0.88 (n=14) | 0.29 (n=6) | 0.83 (n=22) | 0.89 (n=41) | 0.81 (n=57) | 0.65 (n=20) | 0.79 (n=21) | 0.67 (n=20) |

Paired difference versus TF-IDF (model minus TF-IDF), macro-F1:

- majority: -0.596, 95% CI [-0.673, -0.501]
- keyword: -0.226, 95% CI [-0.360, -0.103]
- afroxlmr: 0.020, 95% CI [-0.062, 0.117]

## sentiment

n = 150

Class support (gold n):

- negative: 93
- not_negative: 57

| model | macro-F1 | 95% CI |
| --- | ---: | --- |
| majority | 0.383 | [0.353, 0.409] |
| keyword | 0.383 | [0.353, 0.409] |
| tfidf | 0.758 | [0.685, 0.825] |
| afroxlmr | 0.800 | [0.727, 0.863] |

Per-class F1 (n is the gold support):

| model | negative | not_negative |
| --- | ---: | ---: |
| majority | 0.77 (n=93) | 0.00 (n=57) |
| keyword | 0.77 (n=93) | 0.00 (n=57) |
| tfidf | 0.80 (n=93) | 0.71 (n=57) |
| afroxlmr | 0.84 (n=93) | 0.76 (n=57) |

Paired difference versus TF-IDF (model minus TF-IDF), macro-F1:

- majority: -0.375, 95% CI [-0.454, -0.291]
- keyword: -0.375, 95% CI [-0.454, -0.291]
- afroxlmr: 0.043, 95% CI [-0.017, 0.099]

## urgency

n = 120

Class support (gold n):

- low: 17
- medium: 52
- emergency: 51

| model | macro-F1 | 95% CI |
| --- | ---: | --- |
| majority | 0.199 | [0.170, 0.227] |
| keyword | 0.215 | [0.155, 0.272] |
| tfidf | 0.633 | [0.526, 0.723] |
| afroxlmr | 0.687 | [0.589, 0.770] |

| model | QWK | emergency recall | false-emergency rate | n emergency | n other |
| --- | ---: | ---: | ---: | ---: | ---: |
| majority | 0.000 | 0.000 | 0.000 | 51 | 69 |
| keyword | 0.044 | 0.000 | 0.043 | 51 | 69 |
| tfidf | 0.554 | 0.667 | 0.174 | 51 | 69 |
| afroxlmr | 0.687 | 0.804 | 0.145 | 51 | 69 |

Per-class F1 (n is the gold support):

| model | low | medium | emergency |
| --- | ---: | ---: | ---: |
| majority | 0.00 (n=17) | 0.00 (n=52) | 0.60 (n=51) |
| keyword | 0.29 (n=17) | 0.36 (n=52) | 0.00 (n=51) |
| tfidf | 0.48 (n=17) | 0.67 (n=52) | 0.74 (n=51) |
| afroxlmr | 0.59 (n=17) | 0.67 (n=52) | 0.80 (n=51) |

Paired difference versus TF-IDF (model minus TF-IDF), macro-F1:

- majority: -0.434, 95% CI [-0.536, -0.324]
- keyword: -0.418, 95% CI [-0.511, -0.307]
- afroxlmr: 0.055, 95% CI [-0.020, 0.135]

Taxonomy picks: 30 (cap 30).
