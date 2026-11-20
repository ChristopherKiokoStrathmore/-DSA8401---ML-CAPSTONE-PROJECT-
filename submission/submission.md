# Airtel Kenya multi-head message classifier

Chris Kioko

DSA 8401 Applied Machine Learning

Strathmore

This submission is a multi-head AfroXLMR-base classifier for Airtel Kenya customer messages written in English, Swahili, and Sheng. A shared encoder predicts one of 10 issue classes, binary sentiment (negative or not_negative), and ordinal urgency (low, medium, or emergency) with a CORN head. On the real test split, AfroXLMR macro-F1 is 0.656 (95% CI [0.595, 0.707]) for issue (n = 224), 0.800 (95% CI [0.727, 0.863]) for sentiment (n = 150), and 0.687 (95% CI [0.589, 0.770]) for urgency (n = 120). Those estimates and intervals are the `reports/results.json` figures rounded to three decimals, the same rounding `evaluate.py` writes into `reports/results.md`. The paired macro-F1 differences versus TF-IDF are +0.020 [-0.062, 0.117], +0.043 [-0.017, 0.099], and +0.055 [-0.020, 0.135], and each interval includes zero.

Repository: https://github.com/ChristopherKiokoStrathmore/-DSA8401---ML-CAPSTONE-PROJECT-

Report: https://github.com/ChristopherKiokoStrathmore/-DSA8401---ML-CAPSTONE-PROJECT-/blob/main/report/report.pdf

Frontend: https://dsa-8401-ml-capstone-project.vercel.app/

API docs (Swagger): https://api-production-09d942cd.up.railway.app/docs

Health: https://api-production-09d942cd.up.railway.app/health
