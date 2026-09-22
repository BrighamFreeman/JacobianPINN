# Import the Jacobian engine from engine.py into a different file
from engine import JacobianPINNEngine

engine = JacobianPINNEngine(
    sine_model_path="/content/sine_odd_and_even_loss.keras",
    power_model_path="/content/power_model.keras",
    exponential_model_path="/content/exponential_model.keras"
)

system = [

    {
        "name": "f1",

        "terms": [

            {
                "variable": "x",
                "function": "power",
                "coefficient": 2.0,
                "parameter": 3.0
            },

            {
                "variable": "y",
                "function": "sine",
                "coefficient": 4.0,
                "parameter": 2.0
            }
        ]
    },

    {
        "name": "f2",

        "terms": [

            {
                "variable": "x",
                "function": "sine",
                "coefficient": 3.0,
                "parameter": 1.0
            },

            {
                "variable": "y",
                "function": "exp",
                "coefficient": 1.0,
                "parameter": 2.0
            }
        ]
    }
]

point = {
    "x": 1.0,
    "y": 0.5
}

J = engine.jacobian(
    system=system,
    variable_values=point
)

print(J)
