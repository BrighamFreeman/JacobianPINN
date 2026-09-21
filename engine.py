import numpy as np
import tensorflow as tf


class JacobianPINNEngine:

    # --------------------------------------------------------
    # 1. Initialize engine and load models
    # --------------------------------------------------------

    def __init__(
        self,
        sine_model_path,
        power_model_path,
        dtype=tf.float64
    ):

        self.DTYPE = dtype

        self.sine_model = tf.keras.models.load_model(
            sine_model_path,
            compile=False
        )

        self.power_model = tf.keras.models.load_model(
            power_model_path,
            compile=False
        )

        self.exponential_model = tf.keras.models.load_model(
            exponential_model_path,
            compile=False
        )

        # ----------------------------------------------------
        # Function-family registry
        #
        # This avoids a long if/elif chain later.
        # ----------------------------------------------------

        self.derivative_registry = {
            "sine": self._sine_derivative,
            "power": self._power_derivative,
            "exp": self._exponential_derivative
        }


    # --------------------------------------------------------
    # 2. Power PINN inference
    # --------------------------------------------------------

    def _power_derivative(
        self,
        variable_value,
        coefficient,
        parameter
    ):
        """
        Evaluate derivative of:

            coefficient * x^parameter

        using the trained Power PINN.
        """

        x = tf.constant(
            [[float(variable_value)]],
            dtype=self.DTYPE
        )

        power = tf.constant(
            [[float(parameter)]],
            dtype=self.DTYPE
        )

        coefficient = tf.constant(
            [[float(coefficient)]],
            dtype=self.DTYPE
        )

        with tf.GradientTape() as tape:

            tape.watch(x)

            model_input = tf.concat(
                [
                    x,
                    power,
                    coefficient
                ],
                axis=1
            )

            prediction = self.power_model(
                model_input,
                training=False
            )

        derivative = tape.gradient(
            prediction,
            x
        )

        return float(
            derivative.numpy()[0, 0]
        )


    # --------------------------------------------------------
    # 3. Sine PINN inference
    # --------------------------------------------------------

    def _sine_derivative(
        self,
        variable_value,
        coefficient,
        parameter
    ):
        """
        Evaluate derivative of:

            coefficient * sin(parameter * x)

        using the trained Sine PINN.

        Here:
            coefficient = amplitude
            parameter   = frequency
        """

        x = tf.constant(
            [[float(variable_value)]],
            dtype=self.DTYPE
        )

        amplitude = tf.constant(
            [[float(coefficient)]],
            dtype=self.DTYPE
        )

        frequency = tf.constant(
            [[float(parameter)]],
            dtype=self.DTYPE
        )

        with tf.GradientTape() as tape:

            tape.watch(x)

            model_input = tf.concat(
                [
                    x,
                    amplitude,
                    frequency
                ],
                axis=1
            )

            prediction = self.sine_model(
                model_input,
                training=False
            )

        derivative = tape.gradient(
            prediction,
            x
        )

        return float(
            derivative.numpy()[0, 0]
        )


    # --------------------------------------------------------
    # 4. Route one term to the appropriate PINN
    # --------------------------------------------------------

    def _route_derivative(
        self,
        term,
        variable_value
    ):

        function_type = (
            term["function"]
            .strip()
            .lower()
        )

        coefficient = float(
            term["coefficient"]
        )

        parameter = float(
            term["parameter"]
        )

        # ----------------------------------------------------
        # Make sure the requested function family exists
        # ----------------------------------------------------

        if function_type not in self.derivative_registry:

            raise ValueError(
                f"Unsupported function type: "
                f"{function_type}"
            )

        # ----------------------------------------------------
        # Look up the appropriate derivative function
        # ----------------------------------------------------

        derivative_function = (
            self.derivative_registry[
                function_type
            ]
        )

        return derivative_function(
            variable_value=variable_value,
            coefficient=coefficient,
            parameter=parameter
        )


    # --------------------------------------------------------
    # 5. Calculate one partial derivative
    # --------------------------------------------------------

    def _calculate_partial(
        self,
        equation,
        target_variable,
        variable_values
    ):

        target_variable = (
            target_variable
            .strip()
            .lower()
        )

        derivative_total = 0.0

        for term in equation["terms"]:

            term_variable = (
                term["variable"]
                .strip()
                .lower()
            )

            # ------------------------------------------------
            # Term does not depend on requested variable
            # ------------------------------------------------

            if term_variable != target_variable:
                continue

            # ------------------------------------------------
            # Get value of x, y, z, etc.
            # ------------------------------------------------

            variable_value = (
                variable_values[
                    target_variable
                ]
            )

            # ------------------------------------------------
            # Route this term to the correct PINN
            # ------------------------------------------------

            term_derivative = (
                self._route_derivative(
                    term=term,
                    variable_value=variable_value
                )
            )

            derivative_total += (
                term_derivative
            )

        return derivative_total


    # --------------------------------------------------------
    # 6. Public Jacobian method
    # --------------------------------------------------------

    def jacobian(
        self,
        system,
        variable_values,
        variables=None
    ):
        """
        Construct the Jacobian matrix.

        Example:

            F(x,y) = [f1(x,y), f2(x,y)]

                  | df1/dx   df1/dy |
            J  =  |                 |
                  | df2/dx   df2/dy |
        """

        # ----------------------------------------------------
        # If variables are not explicitly supplied,
        # use the keys from variable_values.
        # ----------------------------------------------------

        if variables is None:

            variables = tuple(
                variable_values.keys()
            )

        jacobian_rows = []

        # ----------------------------------------------------
        # Each equation becomes one row
        # ----------------------------------------------------

        for equation in system:

            row = []

            # ------------------------------------------------
            # Each independent variable becomes one column
            # ------------------------------------------------

            for variable in variables:

                partial = (
                    self._calculate_partial(
                        equation=equation,
                        target_variable=variable,
                        variable_values=variable_values
                    )
                )

                row.append(
                    partial
                )

            jacobian_rows.append(
                row
            )

        return np.array(
            jacobian_rows,
            dtype=np.float64
        )


    # --------------------------------------------------------
    # 7. Optional method for registering future models
    # --------------------------------------------------------

    def register_function_family(
        self,
        name,
        derivative_function
    ):
        """
        Add another supported function family.

        Example:

            engine.register_function_family(
                "cosine",
                cosine_derivative_function
            )
        """

        name = (
            name
            .strip()
            .lower()
        )

        self.derivative_registry[
            name
        ] = derivative_function
    # ------------------------------------------------------------
    #  8.  Public Help Function
    # ------------------------------------------------------------
    def help(self):
      """
      Display usage information for the Jacobian PINN engine.
      """

      help_text = """
      ============================================================
      JacobianPINNEngine Help
      ============================================================

      PURPOSE
      -------
      Build Jacobian matrices from mixed mathematical function
      families using trained PINN models.

      Currently supported function types:
          - power
          - sine


      BASIC SETUP
      -----------
      Create the engine:

          engine = JacobianPINNEngine(
              sine_model_path="sine_model.keras",
              power_model_path="power_model.keras"
          )


      SYSTEM FORMAT
      -------------
      Each equation is represented as a dictionary containing
      a list of terms.

      Example:

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
              }
          ]

      This represents:

          f1(x,y) = 2x^3 + 4sin(2y)


      TERM FORMAT
      -----------
      Each term requires:

          variable
              Independent variable used by the term.

          function
              Function family.

              Supported values:
                  "power"
                  "sine"

          coefficient
              Leading coefficient or amplitude.

          parameter
              Meaning depends on function family.

              power:
                  coefficient * x^parameter

              sine:
                  coefficient * sin(parameter * x)


      EVALUATION POINT
      ----------------
      Provide variable values as a dictionary:

          point = {
              "x": 1.0,
              "y": 2.0
          }


      BUILDING THE JACOBIAN
      ---------------------
      Call:

          J = engine.jacobian(
              system=system,
              variable_values=point
          )

      Then:

          print(J)


      EXAMPLE
      -------
      For:

          f1(x,y) = 2x^3 + 4sin(2y)
          f2(x,y) = 3sin(x) + y^4

      use:

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
                          "function": "power",
                          "coefficient": 1.0,
                          "parameter": 4.0
                      }
                  ]
              }
          ]

          point = {
              "x": 1.0,
              "y": 2.0
          }

          J = engine.jacobian(
              system,
              point
          )

          print(J)


      JACOBIAN STRUCTURE
      ------------------
      For:

          F(x,y) = [f1(x,y), f2(x,y)]

      the output is:

              | df1/dx   df1/dy |
          J = |                 |
              | df2/dx   df2/dy |


      ADDING FUNCTION FAMILIES
      ------------------------
      Additional derivative handlers can be registered using:

          engine.register_function_family(
              "cosine",
              cosine_derivative_function
          )


      MODEL LIMITATIONS
      -----------------
      The numerical accuracy and supported input domain depend on
      the training ranges of the underlying PINN models.

      Inputs outside those ranges may produce extrapolation error.

      ============================================================
      """

      print(help_text)
