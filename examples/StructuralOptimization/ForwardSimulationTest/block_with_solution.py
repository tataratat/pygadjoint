import numpy as np
import splinepy as spp

import pygadjoints as pyg

# Define boundary conditions
E = 200000
nu = 0.25
lambda_ = (E * nu) / ((1 + nu) * (1 - 2 * nu))
mu_ = (E) / (2 * (1 + nu))
print(f"Lambda : {lambda_}")
print(f"    Mu : {mu_}")

boundary_conditions_options = [
    {
        # F - function (source)
        "tag": "Function",
        "attributes": {"type": "FunctionExpr", "id": "1", "dim": "3"},
        "text": "\n    ",
        "children": [
            {
                "tag": "c",
                "attributes": {"index": "0"},
                "text": f"-6 * {mu_} * y * z",
            },
            {
                "tag": "c",
                "attributes": {"index": "1"},
                "text": f"2 * {mu_} * x * z",
            },
            {
                "tag": "c",
                "attributes": {"index": "2"},
                "text": f"10 * {mu_} * x * y",
            },
        ],
    },
    {
        "tag": "boundaryConditions",
        "attributes": {"multipatch": "0", "id": "2"},
        "children": [
            {
                "tag": "Function",
                "attributes": {
                    "type": "FunctionExpr",
                    "dim": "3",
                    "index": "0",
                },
                "children": [
                    {
                        "tag": "c",
                        "attributes": {"index": "0"},
                        "text": "27 / 7 * x * x * y * z",
                    },
                    {
                        "tag": "c",
                        "attributes": {"index": "1"},
                        "text": "- 1 / 7 * x * y * y * z",
                    },
                    {
                        "tag": "c",
                        "attributes": {"index": "2"},
                        "text": "- 29 / 7 * x * y * z * z",
                    },
                ],
            },
        ],
    },
    {
        "tag": "Function",
        "attributes": {
            "type": "FunctionExpr",
            "dim": "3",
            "id": "3",
        },
        "children": [
            {
                "tag": "c",
                "attributes": {"index": "0"},
                "text": "27 / 7 * x * x * y * z",
            },
            {
                "tag": "c",
                "attributes": {"index": "1"},
                "text": "- 1 / 7 * x * y * y * z",
            },
            {
                "tag": "c",
                "attributes": {"index": "2"},
                "text": "- 29 / 7 * x * y * z * z",
            },
        ],
    },
]

for i_surface in range(1, 2):
    boundary_conditions_options[1]["children"].append(
        {
            "tag": "bc",
            "attributes": {
                "type": "Dirichlet",
                "function": str(0),
                "unknown": str(0),
                "name": f"BID{i_surface}",
            },
        }
    )

# Create a simple geometry
n_refine = 6
microtile_patches = spp.helpme.create.box(3, 4, 3.5).bspline
microtile_patches.elevate_degrees([0, 1, 2])
microtile_patches.cps += 1 * np.random.random(microtile_patches.cps.shape)
microtile_patches.insert_knots(0, np.linspace(0, 1, n_refine + 2))
microtile_patches.insert_knots(1, np.linspace(0, 1, n_refine + 2))
microtile_patches.insert_knots(2, np.linspace(0, 1, n_refine + 2))
microtile_patches.show()
microtile = spp.Multipatch([microtile_patches])
microtile.determine_interfaces()
# microtile.boundaries_from_continuity()
spp.io.gismo.export(
    "mini_example.xml",
    microtile,
    options=boundary_conditions_options,
    as_base64=False,
    labeled_boundaries=True,
)
print("Test 1")

linear_elasticity_solver = pyg.LinearElasticityProblem()
linear_elasticity_solver.init("mini_example.xml", 0, 0, False)
linear_elasticity_solver.set_material_constants(lambda_, mu_)
linear_elasticity_solver.assemble()
linear_elasticity_solver.solve_linear_system()
linear_elasticity_solver.export_paraview("solution", False, 3375000, True)
linear_elasticity_solver.export_xml("solution_2")
