BEARING_KBS = {
    "Normal": {
        "status": "NORMAL",
        "severity": "LOW",
        "actions": [
            "Continue routine operation if other machine indicators are normal.",
            "Continue condition monitoring and compare vibration against "
            "a baseline recorded at a comparable speed and load.",
            "Follow the manufacturer's preventive maintenance schedule."
        ],
        "reason": "No bearing fault was detected by the model."
    },

    "Inner Race": {
        "status": "FAULT_DETECTED",
        "severity": "HIGH",
        "component": "Inner race",
        "possible_causes": [
            "Raceway fatigue or spalling",
            "Inadequate or incorrect lubrication",
            "Shaft misalignment or excessive loading",
            "Incorrect shaft fit or mounting damage"
        ],
        "actions": [
            "Arrange a prompt bearing inspection and review vibration trends.",
            "Check lubricant type, quantity, condition, and delivery.",
            "Inspect shaft alignment, shaft fit, seating, and runout.",
            "Check operating loads and bearing temperature.",
            "If damage is confirmed, replace the bearing as required "
            "and inspect the shaft and mating components.",
            "Record the physical damage and correct the verified root cause."
        ]
    },

    "Ball": {
        "status": "FAULT_DETECTED",
        "severity": "HIGH",
        "component": "Rolling element (ball)",
        "possible_causes": [
            "Rolling-element fatigue or surface damage",
            "Contamination or inadequate lubrication",
            "Improper mounting or impact damage",
            "Cage or related component damage"
        ],
        "actions": [
            "Arrange a prompt bearing inspection and review vibration trends.",
            "Inspect rolling elements, raceways, cage, and lubricant "
            "for pitting, dents, wear, and debris.",
            "Check seals and lubricant for contamination or moisture.",
            "Verify the mounting method and operating loads.",
            "If damage is confirmed, replace the bearing as required "
            "and inspect the entire bearing assembly.",
            "Correct the verified cause before returning to service."
        ]
    },

    "Outer Race": {
        "status": "FAULT_DETECTED",
        "severity": "HIGH",
        "component": "Outer race",
        "possible_causes": [
            "Raceway fatigue or spalling",
            "Housing distortion or incorrect housing fit",
            "Shaft-to-housing misalignment",
            "Excessive or uneven loading"
        ],
        "actions": [
            "Arrange a prompt bearing inspection and review vibration trends.",
            "Inspect the housing bore, bearing seat, and mounting surface.",
            "Check for looseness, distortion, poor seating, and wear.",
            "Verify shaft alignment and operating loads.",
            "If damage is confirmed, replace the bearing as required "
            "and inspect the housing and mating components.",
            "Correct any verified fit, alignment, or loading issue."
        ]
    },

    "unknown_fault": {
        "status": "FAULT_DETECTED_TYPE_UNKNOWN",
        "severity": "HIGH",
        "actions": [
            "Treat the fault indication as requiring investigation.",
            "Repeat the measurement and check signal quality.",
            "Review vibration, temperature, noise, and operating conditions.",
            "Arrange an inspection to determine the fault location.",
            "Do not infer a component-specific cause until verified."
        ]
    },

    "conflicting_prediction": {
        "status": "PREDICTION_CONFLICT",
        "severity": "REVIEW",
        "actions": [
            "Flag the prediction for review.",
            "Check model outputs, label mapping, and confidence scores.",
            "Repeat the measurement or use an independent diagnostic check.",
            "Do not issue a component-specific maintenance instruction "
            "until the predictions are reconciled."
        ]
    },

    "invalid_input": {
        "status": "INVALID_INPUT",
        "severity": "REVIEW",
        "actions": [
            "Check the fault prediction and fault-type label.",
            "Verify that model outputs were mapped correctly."
        ]
    }
}

def get_maintenance_action(fault, fault_type):
    # Normalize the model outputs
    fault = int(fault)
    fault_type = str(fault_type) if str(fault_type) in [
        'Normal', 'Ball (Drive-End)', 'Ball (Fan-End)', 'Inner Race (Drive-End)', 'Inner Race (Fan-End)',
        'Outer Race (Drive-End)', 'Outer Race (Fan-End)'
    ] else "Unknown"
    fault_type = fault_type.split("(")[0].strip()

    # Check for inconsistent predictions
    if fault == 0 and fault_type != "Normal":
        return {
            "status": "Prediction Conflict",
            "fault": fault,
            "fault_type": fault_type,
            "actions": [
                "Review the conflicting model predictions.",
                "Check confidence scores and repeat the measurement."
            ]
        }

    if fault == 1 and fault_type == "Normal":
        return {
            "status": "Fault Detected - Type Uncertain",
            **BEARING_KBS["unknown_fault"]
        }

    # Select the matching rule
    if fault == 0 and fault_type == "Normal":
        rule = BEARING_KBS["Normal"]
    elif fault == 1:
        rule = BEARING_KBS.get(fault_type, BEARING_KBS["unknown_fault"])
    else:
        return {
            "status": "Invalid Prediction",
            "actions": ["Verify the model output."]
        }

    return {
        "status": "Fault Detected" if fault else "Normal",
        **rule
    }