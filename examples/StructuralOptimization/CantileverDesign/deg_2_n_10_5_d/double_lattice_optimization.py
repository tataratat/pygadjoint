import numpy as np
import scipy
import splinepy as sp

import pygadjoints

###
# SIMULATION PARAMETERS
###
ULTRA_VERBOSE = True
N_THREAD = 1

###
# MATERIAL PARAMETERS
###
ACTIVATE_SOURCE_FUNCTION = False
Youngs_modulus = 1e6
poisson_ratio = 0.3
lame_lambda_ = (
    Youngs_modulus
    * poisson_ratio
    / ((1 - 2 * poisson_ratio) * (1 + poisson_ratio))
)
lame_mu_ = Youngs_modulus / (2 * (1 + poisson_ratio))
number_of_tiles_with_load = 2
density_ = 1

source_function_ = [0.0, 0.0]
neumann_force_ = [0, -4500]
dirichlet_value = [0.0, 0.0]
dim = 2

print(f"Youngs Modulus    : {Youngs_modulus}")
print(f"Poisson's ratio   : {poisson_ratio}")
print(f"First Lame para   : {lame_lambda_}")
print(f"Second Lame param : {lame_mu_}")

# Define function parameters
GISMO_OPTIONS = [
    {
        # F - function (source)
        "tag": "Function",
        "attributes": {
            "type": "FunctionExpr",
            "id": f"{1}",
            "dim": f"{dim}",
        },
        "children": [
            {
                "tag": "c",
                "attributes": {"index": "0"},
                "text": f"{source_function_[0]}",
            },
            {
                "tag": "c",
                "attributes": {"index": "1"},
                "text": f"{source_function_[1]}",
            },
        ],
    },
    {
        # Boundary Conditions
        "tag": "boundaryConditions",
        "attributes": {"multipatch": "0", "id": "2"},
        "children": [
            {
                "tag": "Function",
                "attributes": {
                    "type": "FunctionExpr",
                    "dim": f"{dim}",
                    "index": "0",
                    "c": "2",
                },
                "children": [
                    {
                        "tag": "c",
                        "attributes": {"index": "0"},
                        "text": f"{dirichlet_value[0]}",
                    },
                    {
                        "tag": "c",
                        "attributes": {"index": "1"},
                        "text": f"{dirichlet_value[1]}",
                    },
                ],
            },
            {
                "tag": "Function",
                "attributes": {
                    "type": "FunctionExpr",
                    "dim": f"{dim}",
                    "index": "1",
                    "c": "2",
                },
                "children": [
                    {
                        "tag": "c",
                        "attributes": {"index": "0"},
                        "text": f"{neumann_force_[0]}",
                    },
                    {
                        "tag": "c",
                        "attributes": {"index": "1"},
                        "text": f"{neumann_force_[1]}",
                    },
                ],
            },
        ],
    },
]


# Set boundary conditions on all boundary elements of the multipatch (-1)
GISMO_OPTIONS[1]["children"].append(
    {
        "tag": "bc",
        "attributes": {
            "type": "Dirichlet",
            "function": str(0),
            "unknown": str(0),
            "name": f"BID{2}",
        },
    }
)

GISMO_OPTIONS[1]["children"].append(
    {
        "tag": "bc",
        "attributes": {
            "type": "Neumann",
            "function": str(1),
            "unknown": str(0),
            "name": f"BID{6}",
        },
    }
)

if not ACTIVATE_SOURCE_FUNCTION:
    GISMO_OPTIONS.pop(0)


class Optimizer:
    def __init__(
        self,
        macro_spline,
        microtile,
        para_spline,
        identifier_function_neumann,
        n_threads=12,
        tiling=[24, 12],
        scaling_factor_objective_function=100,
        n_refinements=1,
        write_logfiles=False,
        max_volume=1.5,
        objective_function_type=1,
        macro_ctps=None,
        parameter_default_value=0.1,
        parameter_scaling=1,
    ):
        self.parameter_default_value = parameter_default_value
        self.n_refinements = n_refinements
        self.microtile = microtile
        self.interfaces = None
        self.macro_spline = macro_spline.bspline
        self.macro_spline_original = self.macro_spline.copy()
        self.para_spline = para_spline.bspline
        self.identifier_function_neumann = identifier_function_neumann
        self.tiling = tiling
        self.scaling_factor_objective_function = (
            scaling_factor_objective_function
        )
        self.linear_solver = pygadjoints.LinearElasticityProblem()
        self.linear_solver.set_number_of_threads(n_threads)
        self.linear_solver.set_objective_function(objective_function_type)
        self.linear_solver.set_material_constants(lame_lambda_, lame_mu_)
        self.last_parameters = None
        self.iteration = 0
        self.write_logfiles = write_logfiles
        self.max_volume = max_volume
        self.macro_ctps = macro_ctps
        self.parameter_scaling = parameter_scaling

    def prepare_microstructure(self):
        def parametrization_function(x):
            """
            Parametrization Function (determines thickness)
            """
            return np.tile(self.para_spline.evaluate(x), [1, 2])

        def parameter_sensitivity_function(x):
            basis_function_matrix = np.zeros(
                (x.shape[0], self.para_spline.control_points.shape[0])
            )
            basis_functions, support = self.para_spline.basis_and_support(x)
            np.put_along_axis(
                basis_function_matrix, support, basis_functions, axis=1
            )
            return np.tile(
                basis_function_matrix.reshape(x.shape[0], 1, -1), [1, 2, 1]
            )

        # Initialize microstructure generator and assign values
        generator = sp.microstructure.Microstructure()
        generator.deformation_function = self.macro_spline
        generator.tiling = self.tiling
        generator.microtile = self.microtile
        generator.parametrization_function = parametrization_function
        generator.parameter_sensitivity_function = (
            parameter_sensitivity_function
        )

        # Creator for identifier functions
        def identifier_function(deformation_function, face_id):
            boundary_spline = deformation_function.extract.boundaries()[
                face_id
            ]

            def identifier_function(x):
                distance_2_boundary = boundary_spline.proximities(
                    queries=x,
                    initial_guess_sample_resolutions=[21]
                    * boundary_spline.para_dim,
                    tolerance=1e-9,
                    return_verbose=True,
                )[3]
                return distance_2_boundary.flatten() < 1e-8

            return identifier_function

        multipatch = generator.create(
            contact_length=0.5, macro_sensitivities=len(self.macro_ctps) > 0
        )

        # Reuse existing interfaces
        if self.interfaces is None:
            multipatch.determine_interfaces()
            for i in range(self.macro_spline.dim * 2):
                multipatch.boundary_from_function(
                    identifier_function(generator.deformation_function, i)
                )
            if self.identifier_function_neumann is not None:
                multipatch.boundary_from_function(
                    self.identifier_function_neumann, mask=[5]
                )

            self.interfaces = multipatch.interfaces
        else:
            multipatch.interfaces = self.interfaces
        sp.io.gismo.export(
            self.get_filename(),
            multipatch=multipatch,
            options=GISMO_OPTIONS,
            export_fields=True,
            as_base64=True,
            field_mask=(
                np.arange(0, self.para_spline.cps.shape[0]).tolist()
                + (
                    np.array(self.macro_ctps) + self.para_spline.cps.shape[0]
                ).tolist()
            ),
        )

    def ensure_parameters(self, parameters, increase_count=True):
        # Check if anything changed since last call
        if self.last_parameters is not None and np.allclose(
            self.last_parameters, parameters
        ):
            return

        # Apply Parameter Scaling
        inverse_scaling = 1 / self.parameter_scaling

        if increase_count:
            self.iteration += 1

        # Something differs (or first iteration)
        self.para_spline.cps[:] = (
            parameters[: self.para_spline.cps.shape[0]].reshape(-1, 1)
            * inverse_scaling
        )
        self.macro_spline.cps.ravel()[self.macro_ctps] = (
            parameters[self.para_spline.cps.shape[0] :]
            + self.macro_spline_original.cps.ravel()[self.macro_ctps]
        )
        self.prepare_microstructure()
        if self.last_parameters is None:
            # First iteration
            self.linear_solver.init(
                self.get_filename(), self.n_refinements, 0, True
            )
        else:
            self.linear_solver.update_geometry(
                self.get_filename(), topology_changes=False
            )

        self.linear_solver.read_control_point_sensitivities(
            self.get_filename() + ".fields.xml"
        )
        self.control_point_sensitivities = (
            self.linear_solver.get_control_point_sensitivities()
        )
        self.last_parameters = parameters.copy()

        # Notify iteration evaluator
        self.current_objective_function_value = None
        self.ctps_sensitivity = None

    def evaluate_iteration(self, parameters):
        self.ensure_parameters(parameters)
        if self.current_objective_function_value is not None:
            return self.current_objective_function_value

        # There is no current solution all checks have been performed
        self.linear_solver.assemble()
        self.linear_solver.solve_linear_system()
        self.current_objective_function_value = (
            self.linear_solver.objective_function()
            * self.scaling_factor_objective_function
        )

        #
        if self.iteration == 1:
            self.linear_solver.export_multipatch_object("multipatch_initial")
            self.linear_solver.export_paraview("initial", False, 3**2, True)

        # Write into logfile
        with open("log_file_iterations.csv", "a") as file1:
            file1.write(
                ", ".join(
                    str(a)
                    for a in (
                        [self.iteration]
                        + [self.current_objective_function_value]
                        + parameters.tolist()
                    )
                )
                + "\n"
            )

        return self.current_objective_function_value

    def evaluate_jacobian(self, parameters):
        # Make sure that current file is valid
        _ = self.evaluate_iteration(parameters)

        # Determine Lagrange multipliers
        self.linear_solver.solve_adjoint_system()
        ctps_sensitivities = (
            self.linear_solver.objective_function_deris_wrt_ctps()
        )
        parameter_sensitivities = (
            (ctps_sensitivities @ self.control_point_sensitivities)
            * self.scaling_factor_objective_function
            / self.parameter_scaling
        )

        # Write into logfile
        with open("log_file_sensitivities.csv", "a") as file1:
            file1.write(
                ", ".join(
                    str(a)
                    for a in (
                        [self.iteration]
                        + parameter_sensitivities.tolist()
                        + parameters.tolist()
                    )
                )
                + "\n"
            )
        return parameter_sensitivities

    def volume(self, parameters):
        self.ensure_parameters(parameters)
        volume = self.linear_solver.volume()

        # Write into logfile
        with open("log_file_volume.csv", "a") as file1:
            file1.write(
                ", ".join(
                    str(a)
                    for a in (
                        [self.iteration] + [volume] + parameters.tolist()
                    )
                )
                + "\n"
            )

        return self.max_volume - volume

    def volume_deriv(self, parameters):
        self.ensure_parameters(parameters)
        volume_sensitivities_ctps = self.linear_solver.volume_deris_wrt_ctps()
        volume_sensitivities = -(
            volume_sensitivities_ctps
            @ self.control_point_sensitivities
            / self.parameter_scaling
        )
        assert not np.any(np.isnan(self.control_point_sensitivities))

        # Write into logfile
        with open("log_file_volume_sensitivities.csv", "a") as file1:
            file1.write(
                ", ".join(
                    str(a)
                    for a in (
                        [self.iteration]
                        + (-volume_sensitivities).tolist()
                        + parameters.tolist()
                    )
                )
                + "\n"
            )
        return volume_sensitivities

    def constraint(self):
        return {"type": "ineq", "fun": self.volume, "jac": self.volume_deriv}

    def finalize(self, parameters):
        self.ensure_parameters(parameters, increase_count=False)
        self.linear_solver.assemble()
        self.linear_solver.solve_linear_system()
        self.linear_solver.export_multipatch_object("multipatch_optimized")
        self.linear_solver.export_paraview("optimized", False, 3**2, True)

    def optimize(self):
        # Initialize the optimization
        n_design_vars_para = self.para_spline.cps.size
        n_design_vars_macro = len(self.macro_ctps)
        initial_guess = np.empty(n_design_vars_macro + n_design_vars_para)
        initial_guess[:n_design_vars_para] = (
            np.ones(n_design_vars_para)
            * self.parameter_default_value
            * self.parameter_scaling
        )
        initial_guess[n_design_vars_para:] = 0

        optim = scipy.optimize.minimize(
            self.evaluate_iteration,
            initial_guess.ravel(),
            method="SLSQP",
            jac=self.evaluate_jacobian,
            bounds=(
                [
                    (
                        0.0111 * self.parameter_scaling,
                        0.207106 * self.parameter_scaling,
                    )
                    for _ in range(n_design_vars_para)
                ]
                + [(-0.5, 0.5) for _ in range(n_design_vars_macro)]
            ),
            constraints=self.constraint(),
            options={"disp": True},
            tol=1e-4,
        )
        # Finalize
        self.finalize(optim.x)
        print("Best Parameters : ")
        print(optim.x)
        print(optim)

    def get_filename(self):
        return (
            "lattice_structure_"
            + str(self.tiling[0])
            + "x"
            + str(self.tiling[1])
            + ".xml"
        )


def main():
    # Set the number of available threads (will be passed to splinepy and
    # pygdjoints)

    # Geometry definition
    tiling = [24, 12]
    parameter_spline_degrees = [2, 2]
    parameter_spline_cps_dimensions = [10, 5]
    parameter_default_value = 0.16914405585511014 / 5  # For volume density 0.3
    volume_density = 0.3

    scaling_factor_objective_function = 1 / 0.006497784904442101
    scaling_factor_parameters = 5
    n_refinemenets = 0

    sp.settings.NTHREADS = 1
    write_logfiles = True

    # Create parameters spline
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
        control_points=np.ones((np.prod(parameter_spline_cps_dimensions), 1))
        * parameter_default_value,
    )

    # Function for neumann boundary
    def identifier_function_neumann(x):
        return (
            x[:, 0]
            >= (tiling[0] - number_of_tiles_with_load) / tiling[0] * 2.0
            - 1e-12
        )

    macro_spline = sp.Bezier(
        degrees=[1, 1],
        control_points=[
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [2.0, 1.0],
        ],
    )

    dense_volume = macro_spline.integrate.volume()
    max_volume = dense_volume * volume_density
    print(f"Max Volume is:{max_volume} out of {dense_volume}")

    optimizer = Optimizer(
        microtile=sp.microstructure.tiles.DoubleLattice(),
        macro_spline=macro_spline,
        para_spline=parameter_spline,
        identifier_function_neumann=identifier_function_neumann,
        tiling=tiling,
        scaling_factor_objective_function=scaling_factor_objective_function,
        n_refinements=n_refinemenets,
        n_threads=1,
        write_logfiles=write_logfiles,
        max_volume=max_volume,
        objective_function_type=2,
        macro_ctps=[],
        parameter_default_value=parameter_default_value,
        parameter_scaling=scaling_factor_parameters,
    )

    # Try some parameters
    optimizer.optimize()

    exit()


if "__main__" == __name__:
    main()
