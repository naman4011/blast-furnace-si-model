import numpy as np
import shap



def suggest_corrections(si_pred, pi_low, pi_high, top_features):
    # site_target_si must be defined per plant
    target_si = 0.45
    delta = si_pred - target_si
    recs = []
    # Corrective actions
    actions = []
    severity = "low" if abs(delta)<0.05 else "medium" if abs(delta)<0.1 else "high"
    if delta > 0:
        actions = [
            "Slightly reduce theoretical combustion temperature (e.g., -10 to -20°C).",
            "Reduce oxygen enrichment / enriching oxygen flow marginally.",
            "Verify burden distribution and permeability index; check for channeling.",
            "Validate instrumentation (O2/flow/pressure sensors) if anomalies persist."
        ]
    else:
        actions = [
            "Slightly increase theoretical combustion temperature (e.g., +10 to +20°C).",
            "Increase oxygen enrichment marginally if stable.",
            "Check raw material chemistry for SiO2 fluctuations.",
            "Verify top gas and blast pressure controls."
        ]

    # Basic rules based on SI prediction and top features

    if si_pred > pi_high:
        # High SI: try to reduce temperature and reducing potential
        if 'ThCoTe' in top_features or 'HoBlTe' in top_features:
            actions.append("Reduce theoretical combustion temperature slightly (-10 to -20°C).")
        if 'CoInSeVa' in top_features:
            actions.append("Reduce coal injection rate in small steps and monitor top gas O2.")
        if 'OxEnRa' in top_features:
            actions.append("Consider slightly decreasing oxygen enrichment to reduce overly reducing zones.")
        actions.append("Check burden permeability and distribution for channeling.")
    elif si_pred < pi_low:
        # Low SI: try to raise temperature or reducing potential
        if 'ThCoTe' in top_features:
            actions.append("Increase theoretical combustion temperature slightly (+10 to +20°C).")
        if 'OxEnRa' in top_features:
            actions.append("Increase oxygen enrichment marginally if safe (improves combustion) — but measure effect on SI.")
        if 'CoInSeVa' in top_features:
            actions.append("Increase coal injection slightly to increase reducing gases.")
        actions.append("Confirm raw material (sinter/pellet) chemistry; check for high flux/slag uptake.")
    else:
        actions.append("SI within expected bounds — continue monitoring.")

    # always include monitoring instructions
    actions.append("Observe SI, top gas O2, and permeability over next 30–60 minutes; escalate if no improvement.")
    action_block = {
            "target_si": target_si,
            "delta": round(delta,3),
            "severity": severity,
            "recommendations": actions
        }
    return action_block

