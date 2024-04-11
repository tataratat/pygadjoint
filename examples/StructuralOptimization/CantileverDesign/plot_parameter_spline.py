import matplotlib.pyplot as plt
import numpy as np
import splinepy as sp
from vedo import Plotter

# Color scheme
TUWIEN_COLOR_SCHEME = {
    "blue": (0, 102, 153),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "blue_1": (84, 133, 171),
    "blue_2": (114, 173, 213),
    "blue_3": (166, 213, 236),
    "blue_4": (223, 242, 253),
    "grey": (100, 99, 99),
    "grey_1": (157, 157, 156),
    "grey_2": (208, 208, 208),
    "grey_3": (237, 237, 237),
    "green": (0, 126, 113),
    "green_1": (106, 170, 165),
    "green_2": (162, 198, 194),
    "green_3": (233, 241, 240),
    "magenta": (186, 70, 130),
    "magenta_1": (205, 129, 168),
    "magenta_2": (223, 175, 202),
    "magenta_3": (245, 229, 239),
    "yellow": (225, 137, 34),
    "yellow_1": (238, 180, 115),
    "yellow_2": (245, 208, 168),
    "yellow_3": (153, 239, 225),
}
# Export helper

# Create a plotter
plotter = Plotter(
    shape=(1, 1),  # Only one field
    N=1,  # Number of things to plot
    size=[4000, 2000],  # Resolution, i.e. pixel density
    sharecam=True,
    offscreen=True,
    title="",
    bg=(255, 255, 255),
    axes=0,
)


# Create macro spline
macro_spline = sp.helpme.create.box(2, 1).bspline
macro_spline_creator = macro_spline.copy()

# Create parameters spline
parameter_spline_degrees = [1] * 2
parameter_spline_cps_dimensions = [12, 6]
tiling = [24, 12]
scaling_factor = 5


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
macro_spline.insert_knots(0, parameter_spline.kvs[0])
macro_spline.insert_knots(1, parameter_spline.kvs[1])

# Read data from file
parent_directory = (
    "/Users/jzwar/Git/pygadjoints/examples/"
    "StructuralOptimization/CantileverDesign/results/CantileverDesign/"
)
child_directory = (
    f"deg_{parameter_spline_degrees[0]}_n_{parameter_spline_cps_dimensions[0]}"
    f"_{parameter_spline_cps_dimensions[1]}_d/"
)
directory = parent_directory + child_directory
iterations_file_name = "log_file_iterations.csv"
iterations = np.genfromtxt(directory + iterations_file_name, delimiter=",")
sensitivities_file_name = "log_file_sensitivities.csv"
sensitivities = np.genfromtxt(
    directory + sensitivities_file_name, delimiter=","
)


# Plot convergence
plt.semilogy(iterations[:, 1], "-x")
plt.semilogy(
    np.linalg.norm(
        sensitivities[:, 1 : np.prod(parameter_spline_cps_dimensions) + 1],
        axis=1,
    )
)
print(f"Improvement : {1 - np.min(iterations[:,1]) / iterations[0,1]}")
# plt.show()


best_value_id = np.argmin(iterations[:, 1])
parameters = iterations[best_value_id, 2:]

parameter_spline.cps[:] = parameters.reshape(-1, 1) / scaling_factor
vmin = np.min(iterations[:, 2:]) / 5
vmax = np.max(iterations[:, 2:]) / 5
viz_lizt = []
for i, (patch, para) in enumerate(
    zip(macro_spline.extract.beziers(), parameter_spline.extract.beziers())
):
    para.cps[:] = parameter_spline.cps[i, 0]
    patch.spline_data["parameter_spline"] = para
    patch.show_options["data"] = "parameter_spline"
    patch.show_options["control_points"] = False
    patch.show_options["knots"] = False
    # patch.show_options["knot_c"] = TUWIEN_COLOR_SCHEME["black"]
    patch.show_options["cmap"] = "jet"
    patch.show_options["vmin"] = vmin
    patch.show_options["vmax"] = vmax
    viz_lizt.append(
        patch.show(
            lighting="off",
            resolutions=3,
            return_showable=True,
        )["spline"]
    )

plotter.show(viz_lizt, zoom="tightest")
plotter.screenshot(directory + "MacroSpline.png")

generator = sp.microstructure.Microstructure(
    deformation_function=macro_spline_creator,
    tiling=tiling,
    microtile=sp.microstructure.tiles.DoubleLattice(),
)


for i in range(iterations.shape[0]):
    plotter.clear()
    print(i)
    parameters = iterations[i, 2:]
    parameter_spline.cps[:] = parameters.reshape(-1, 1) / scaling_factor

    def parameter_function(x):
        return np.tile(parameter_spline.evaluate(x), [1, 2])

    generator.parametrization_function = parameter_function

    microstructure = generator.create()

    # Set show options
    microstructure.show_options["c"] = TUWIEN_COLOR_SCHEME["grey_1"]
    microstructure.show_options["control_points"] = False
    microstructure.show_options["knots"] = False
    microstructure_showable = microstructure.show(
        lighting="off",
        resolutions=3,
        return_showable=True,
    )

    # Set show_options
    plotter.show(microstructure_showable["spline"], zoom="tightest")
    plotter.screenshot(directory + f"Iteration_{i:03}.png")
