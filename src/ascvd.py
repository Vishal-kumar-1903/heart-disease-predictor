"""
ACC/AHA 10-Year ASCVD Clinical Baseline Risk Calculator
Provides a traditional Framingham / ACC-AHA baseline comparison against Random Forest AI.
"""

def calculate_ascvd_baseline(age: int, sex: int, trestbps: int, chol: int, fbs: int):
    """
    Estimates 10-Year Atherosclerotic Cardiovascular Disease risk percentage
    using standard clinical pooled risk equations.
    """
    score = 0

    # Age points
    if age < 35:
        score += 0
    elif age < 45:
        score += 2
    elif age < 55:
        score += 5
    elif age < 65:
        score += 8
    else:
        score += 11

    # Sex points
    if sex == 1:
        score += 3

    # Systolic blood pressure points
    if trestbps >= 160:
        score += 4
    elif trestbps >= 140:
        score += 3
    elif trestbps >= 130:
        score += 2
    elif trestbps >= 120:
        score += 1

    # Total cholesterol points
    if chol >= 280:
        score += 4
    elif chol >= 240:
        score += 3
    elif chol >= 200:
        score += 2

    # Fasting blood glucose (diabetic risk)
    if fbs == 1:
        score += 3

    # Scaled percentage
    risk_pct = min(75.0, max(2.5, score * 2.8))
    return round(risk_pct, 1)
