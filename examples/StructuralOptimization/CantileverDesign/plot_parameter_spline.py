import numpy as np
import splinepy as sp

# Create macro spline
macro_spline = sp.Bezier(
    degrees=[1, 1],
    control_points=[
        [0.0, 0.0],
        [2.0, 0.0],
        [0.0, 1.0],
        [2.0, 1.0],
    ],
)

# Create parameters spline
parameter_spline_degrees = [0, 0]
parameter_spline_cps_dimensions = [24, 12]
parameter_spline = sp.BSpline(
    degrees=parameter_spline_degrees,
    knot_vectors=[
        (
            [0] * parameter_spline_degrees[i]
            + np.linspace(
                0,
                1,
                parameter_spline_cps_dimensions[i]
                - parameter_spline_degrees[i]
                + 1,
            ).tolist()
            + [1] * parameter_spline_degrees[i]
        )
        for i in range(len(parameter_spline_degrees))
    ],
    control_points=np.ones((np.prod(parameter_spline_cps_dimensions), 1)),
)

# Read data from file
directory = (
    "/Users/jzwar/Git/pygadjoints/examples/"
    "StructuralOptimization/CantileverDesign/"
)
file_name = "log_file_iterations.csv"
iterations = np.genfromtxt(directory + file_name, delimiter=",")
best_value_id = np.argmin(iterations[:, 1])
parameters = iterations[best_value_id, 2:]

parameter_spline.cps[:] = parameters.reshape(-1, 1)
macro_spline.spline_data["parameter_spline"] = parameter_spline
macro_spline.show_options["data"] = "parameter_spline"
macro_spline.show_options["control_points"] = False
macro_spline.show_options["knots"] = False
macro_spline.show_options["cmap"] = "jet"

macro_spline.show(resolutions=1000)
