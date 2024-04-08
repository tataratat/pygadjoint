#include <gismo.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <chrono>

#ifdef PYGADJOINTS_USE_OPENMP
#include <omp.h>
#endif

#include "pygadjoints/custom_expression.hpp"
#include "pygadjoints/timer.hpp"

namespace pygadjoints {

using namespace gismo;

namespace py = pybind11;

/// @brief
class DiffusionProblem {
  // Typedefs
  typedef gsExprAssembler<>::geometryMap geometryMap;
  typedef gsExprAssembler<>::variable variable;
  typedef gsExprAssembler<>::space space;
  typedef gsExprAssembler<>::solution solution;

  // using SolverType = gsSparseSolver<>::CGDiagonal;
  using SolverType = gsSparseSolver<>::BiCGSTABILUT;

public:
  DiffusionProblem() : expr_assembler_pde(1, 1) {
#ifdef PYGADJOINTS_USE_OPENMP
    omp_set_num_threads(std::min(omp_get_max_threads(), n_omp_threads));
#endif
  };

  gsStopwatch timer;

  /**
   * @brief Set the Material Constants
   *
   * @param lambda thermal diffusivity
   */
  void SetMaterialConstants(const real_t &thermal_diffusivity) {
    thermal_diffusivity_ = thermal_diffusivity;
  }

  /**
   * @brief Export the results as Paraview vtu file
   *
   * @param fname Filename
   * @param plot_elements Plot patch borders
   * @param sample_rate Samplerate (samples per element)
   * @return void
   */
  void ExportParaview(const std::string &fname, const bool &plot_elements,
                      const int &sample_rate, const bool &export_b64) {
    Timer timer("Exporting Paraview");
    // Generate Paraview File
    gsParaviewCollection collection("ParaviewOutput/" + fname,
                                    expr_evaluator_ptr.get());
    collection.options().setSwitch("plotElements", plot_elements);
    collection.options().setSwitch("base64", export_b64);
    collection.options().setInt("plotElements.resolution", sample_rate);
    collection.options().setInt("numPoints", sample_rate);
    collection.newTimeStep(&mp_pde);
    collection.addField(*solution_expression_ptr, "temperature");
    if (has_solution) {
      auto solution_given = expr_assembler_pde.getCoeff(
          analytical_solution, *geometry_expression_ptr);
      collection.addField(solution_given, "solution");
      collection.addField(*solution_expression_ptr - solution_given, "error");
      std::cout << "Error in L2 norm : "
                << std::sqrt(expr_evaluator_ptr->integral(
                       (*solution_expression_ptr - solution_given) *
                       (*solution_expression_ptr - solution_given)))
                << std::endl;
    }
    collection.saveTimeStep();
    collection.save();
  }

  /**
   * @brief Export the results as Paraview vtu file
   *
   * @param fname Filename
   * @return void
   */
  void ExportMultipatchSolution(const std::string &fname) {
    Timer timer("Exporting Multipatch-Solution");
    // Generate Paraview File
    gsMultiPatch<> mpsol;
    gsFileData<> output;
    solution_expression_ptr->extract(mpsol);
    output << mpsol;
    output.save(fname + ".xml");
  }

  /**
   * @brief Export the results in xml file
   *
   * @param fname Filename
   * @return double Elapsed time
   */
  double ExportXML(const std::string &fname) {
    // Export solution file as xml
    timer.restart();
    gsMultiPatch<> mpsol;
    gsMatrix<> full_solution;
    gsFileData<> output;
    output << solVector;
    solution_expression_ptr->extractFull(full_solution);
    output << full_solution;
    output.save(fname + ".xml");
    return timer.stop();
  }

#ifdef PYGADJOINTS_USE_OPENMP
  /**
   * @brief Set the Number Of Threads for OpenMP
   *
   * Somehow does not compile
   * @param n_threads
   */
  void SetNumberOfThreads(const int &n_threads) {
    n_omp_threads = n_threads;
    omp_set_num_threads(n_threads);
  }
#endif

  void ReadInputFromFile(const std::string &filename) {
    const Timer timer("ReadInputFromFile");
    // IDs in the text file (might change later)
    const int mp_id{0}, source_id{1}, bc_id{2}, solution_id{3}, ass_opt_id{4},
        target_function_id{11};

    has_source_id = false;
    has_solution = false;

    // Import mesh and load relevant information
    gsFileData<> fd(filename);
    fd.getId(mp_id, mp_pde);

    // Check if file has source functions
    if (fd.hasId(source_id)) {
      fd.getId(source_id, source_function);
      has_solution = true;
    }

    // Check if file has source functions
    if (fd.hasId(solution_id)) {
      fd.getId(solution_id, analytical_solution);
      has_source_id = true;
    }

    // Check if the file has a target function (Can also be applicable for
    // target deformation on elasticity problems)
    if (fd.hasId(target_function_id)) {
      fd.getId(target_function_id, target_solution);
      has_target_id = true;
    }

    // Read Boundary conditions
    fd.getId(bc_id, boundary_conditions);
    boundary_conditions.setGeoMap(mp_pde);

    // Check if Compiler options have been set
    if (fd.hasId(ass_opt_id)) {
      gsOptionList Aopt;
      fd.getId(ass_opt_id, Aopt);

      // Set Options in expression assembler
      expr_assembler_pde.setOptions(Aopt);
    }
  }

  void Init(const std::string &filename, const int number_of_refinements,
            const int number_degree_elevations,
            const bool print_summary = false) {
    const Timer timer("Initialisation");
    // Read input parameters
    ReadInputFromFile(filename);

    // Set number of refinements
    n_refinements = number_of_refinements;
    n_degree_elevations = number_degree_elevations;

    //! [Refinement]
    function_basis = gsMultiBasis<>(mp_pde, true);

    // p-refine
    function_basis.degreeElevate(n_degree_elevations);

    // h-refine each basis
    for (int r{0}; r < n_refinements; ++r) {
      function_basis.uniformRefine();
    }

    // Elements used for numerical integration
    expr_assembler_pde.setIntegrationElements(function_basis);

    // Set the dimension
    dimensionality_ = mp_pde.geoDim();

    // Set the discretization space
    basis_function_ptr =
        std::make_shared<space>(expr_assembler_pde.getSpace(function_basis, 1));

    // Solution vector and solution variable
    solution_expression_ptr = std::make_shared<solution>(
        expr_assembler_pde.getSolution(*basis_function_ptr, solVector));

    // Retrieve expression that represents the geometry mapping
    geometry_expression_ptr =
        std::make_shared<geometryMap>(expr_assembler_pde.getMap(mp_pde));

    basis_function_ptr->setup(boundary_conditions, dirichlet::l2Projection, 0);

    // Assign a Dof Mapper
    dof_mapper_ptr =
        std::make_shared<gsDofMapper>(basis_function_ptr->mapper());

    // Evaluator
    expr_evaluator_ptr = std::make_shared<gsExprEvaluator<>>(
        gsExprEvaluator<>(expr_assembler_pde));

    // Initialize the system
    expr_assembler_pde.initSystem();

    // Print summary:
    if (print_summary) {
      std::string padding{"        : "};
      std::cout << padding << "Simulation summary:\n"
                << padding << "--------------------\n"
                << padding << "Number of DOFs :\t\t"
                << expr_assembler_pde.numDofs() << "\n"
                << padding << "Number of patches :\t\t" << mp_pde.nPatches()
                << "\n"
                << padding << "Minimum spline degree: \t"
                << function_basis.minCwiseDegree() << "\n"
                << padding << "Maximum spline degree: \t"
                << function_basis.maxCwiseDegree() << "\n"
                << std::endl;

      boundary_conditions.print(std::cout, true);

      std::cout << "Source Function  : " << source_function << std::endl;
    }
  }

  void Assemble() {
    const Timer timer("Assemble");
    if (!basis_function_ptr) {
      throw std::runtime_error("Error");
    }

    // Auxiliary variables for readability
    const geometryMap &geometric_mapping = *geometry_expression_ptr;
    const space &basis_function = *basis_function_ptr;

    // Compute the system matrix and right-hand side
    auto bilin =
        thermal_diffusivity_ * igrad(basis_function, geometric_mapping) *
        igrad(basis_function, geometric_mapping).tr() * meas(geometric_mapping);

    // Assemble
    expr_assembler_pde.assemble(bilin);
    if (has_source_id) {
      // Set the boundary_conditions term
      auto source_function_expression =
          expr_assembler_pde.getCoeff(source_function, geometric_mapping);
      auto lin_form =
          basis_function * source_function_expression * meas(geometric_mapping);
      expr_assembler_pde.assemble(lin_form);
    }
    // Compute the Neumann terms defined on physical space
    auto g_N = expr_assembler_pde.getBdrFunction(geometric_mapping);

    // Neumann conditions
    expr_assembler_pde.assembleBdr(boundary_conditions.get("Neumann"),
                                   basis_function * g_N *
                                       nv(geometric_mapping).norm());

    system_matrix =
        std::make_shared<const gsSparseMatrix<>>(expr_assembler_pde.matrix());
    system_rhs = std::make_shared<gsMatrix<>>(expr_assembler_pde.rhs());

    // Clear for future evaluations
    expr_assembler_pde.clearMatrix(true);
    expr_assembler_pde.clearRhs();
  }

  void SolveLinearSystem() {
    const Timer timer("SolveLinearSystem");

    ///////////////////
    // Linear Solver //
    ///////////////////
    if ((!system_matrix) || (!system_rhs)) {
      gsWarn << "System matrix and system rhs are required for solving!"
             << std::endl;
      return;
    }
    // Initialize linear solver
    SolverType solver;
    solver.compute(*system_matrix);
    solVector = solver.solve(*system_rhs);
  }

  double ComputeVolume() {
    const Timer timer("ComputeVolume");

    // Compute volume of domain
    if (!expr_evaluator_ptr) {
      GISMO_ERROR("ExprEvaluator not initialized");
    }
    return expr_evaluator_ptr->integral(meas(*geometry_expression_ptr));
  }

  py::array_t<double> ComputeVolumeDerivativeToCTPS() {
    const Timer timer("ComputeVolumeDerivativeToCTPS");

    auto meas_expr = meas(*geometry_expression_ptr); // validated
    const space &basis_function = *basis_function_ptr;
    auto jacobian = jac(*geometry_expression_ptr); // validated
    auto inv_jacs = jacobian.ginv();               // validated

    // Initialize return value
    const int number_of_ctp_dofs{expr_assembler_pde.numDofs()};
    gsMatrix<double> sensitivities_wrt_ctps(1, dimensionality_ *
                                                   number_of_ctp_dofs);

    for (index_t i_dimension{0}; i_dimension < dimensionality_; i_dimension++) {
      Timer iteration_timer(" -> Assembling iteration : " +
                            std::to_string(i_dimension));
      // Calculate local entities
      auto djacdc = dJacdc(basis_function, i_dimension);
      auto aux_expr = (djacdc.tr().cwisetr() * inv_jacs); // validated

      auto meas_expr_dx = meas_expr * (aux_expr).trace(); // validated

      // Compute the derivative of the volume of the domain with respect to
      // the control points Auxiliary expressions
      expr_assembler_pde.assemble(meas_expr_dx.tr());
      sensitivities_wrt_ctps.block(0, i_dimension * number_of_ctp_dofs, 1,
                                   number_of_ctp_dofs) =
          expr_assembler_pde.rhs().transpose();
      expr_assembler_pde.clearRhs();
    }

    py::array_t<double> derivative(sensitivities_wrt_ctps.size());
    double *derivative_ptr = static_cast<double *>(derivative.request().ptr);
    for (int i{}; i < sensitivities_wrt_ctps.size(); i++) {
      derivative_ptr[i] = sensitivities_wrt_ctps(0, i);
    }
    return derivative;
  }

  double ComputeObjectiveFunction() {
    const Timer timer("ComputeObjectiveFunction");
    if (!geometry_expression_ptr) {
      throw std::runtime_error("Error no geometry expression found.");
    }
    if (!has_target_id) {
      std::cerr
          << "No target function defined, cannot evaluate objective function"
          << std::endl;
      return 0.0;
    }

    // Auxiliary
    solution &solution_expression = *solution_expression_ptr;
    geometryMap &geometric_mapping = *geometry_expression_ptr;
    auto target_solution_expr =
        expr_assembler_pde.getCoeff(target_solution, geometric_mapping);
    auto difference = solution_expression - target_solution_expr;

    // Compute objective function on neumann boundary
    real_t obj_value = expr_evaluator_ptr->integralBdrBc(
        boundary_conditions.get("Neumann"),
        (difference * difference) * nv(geometric_mapping).norm());

    return obj_value;
  }

  void SolveAdjointProblem() {
    const Timer timer("SolveAdjointProblem");

    // Auxiliary references
    const geometryMap &geometric_mapping = *geometry_expression_ptr;
    const space &basis_function = *basis_function_ptr;
    const solution &solution_expression = *solution_expression_ptr;
    auto target_solution_expr =
        expr_assembler_pde.getCoeff(target_solution, geometric_mapping);
    auto difference = solution_expression - target_solution_expr;

    //////////////////////////////////////
    // Derivative of Objective Function //
    //////////////////////////////////////
    expr_assembler_pde.clearRhs();

    // Note that we assemble the negative part of the equation to avoid a
    // copy after solving
    expr_assembler_pde.assembleBdr(boundary_conditions.get("Neumann"),
                                   2 * basis_function * difference *
                                       nv(geometric_mapping).norm());
    const auto objective_function_derivative = expr_assembler_pde.rhs();

    /////////////////////////////////
    // Solving the adjoint problem //
    /////////////////////////////////
    auto rhs_vector = expr_assembler_pde.rhs();

    // Initialize linear solver
    SolverType solverAdjoint;
    solverAdjoint.compute(*system_matrix);

    // solve adjoint function
    lagrange_multipliers_ptr = std::make_shared<gsMatrix<>>(
        -solverAdjoint.solve(expr_assembler_pde.rhs()));
    expr_assembler_pde.clearMatrix(true);
    expr_assembler_pde.clearRhs();
  }

  py::array_t<double> ComputeObjectiveFunctionDerivativeWrtCTPS() {
    const Timer timer("ComputeObjectiveFunctionDerivativeWrtCTPS");
    // Check if all required information is available
    if (!(geometry_expression_ptr && basis_function_ptr &&
          solution_expression_ptr && lagrange_multipliers_ptr)) {
      throw std::runtime_error(
          "Some of the required values are not yet initialized.");
    }

    // Initialize the sensitivity vector
    const int number_of_ctp_dofs = lagrange_multipliers_ptr->size();
    gsMatrix<double> sensitivities_wrt_ctps(1, dimensionality_ *
                                                   number_of_ctp_dofs);

    // Auxiliary references
    const geometryMap &geometric_mapping = *geometry_expression_ptr;
    const space &basis_function = *basis_function_ptr;
    const solution &solution_expression = *solution_expression_ptr;

    ////////////////////////////////
    // Derivative of the LHS Form //
    ////////////////////////////////

    // Auxiliary expressions
    auto jacobian = jac(geometric_mapping);   // validated
    auto inv_jacs = jacobian.ginv();          // validated
    auto meas_expr = meas(geometric_mapping); // validated

    // Here the whole thing gets a little more complicated. G+smo can only
    // assemble quadratic matrices, but the derivative matrix is non-quadratic,
    // as we derive the residual of a scalar field by a vector valued
    // expression. To circumvent this, we assemble the matrices for the
    // individual components and assmble block matrices in the following.

    for (index_t i_dimension{0}; i_dimension < dimensionality_; i_dimension++) {
      Timer iteration_timer(" -> Assembling iteration : " +
                            std::to_string(i_dimension));
      // Calculate local entities
      auto djacdc = dJacdc(basis_function, i_dimension);
      auto aux_expr = (djacdc.tr().cwisetr() * inv_jacs); // validated

      auto meas_expr_dx = meas_expr * (aux_expr).trace(); // validated

      // Original Forms
      auto i_grad_of_solution =
          igrad(solution_expression, geometric_mapping); // works fine
      auto i_grad_bf = igrad(basis_function, geometric_mapping);

      // Derivatives
      auto i_grad_of_solution_deriv =
          igrad(solution_expression, geometric_mapping) * aux_expr; // Validated
      auto i_grad_bf_deriv = igrad(basis_function, geometric_mapping) *
                             aux_expr; // Seems to work (tested vs numpy)

      // Start to combine the individual components of the biliner form
      auto first = -thermal_diffusivity_ *
                   (i_grad_bf * i_grad_of_solution_deriv.cwisetr()) *
                   meas_expr; // Validated

      auto second = -thermal_diffusivity_ *
                    frobenius(i_grad_bf_deriv, i_grad_of_solution) *
                    meas_expr; // Validated with numpy
      auto third =
          thermal_diffusivity_ *
          multiply(igrad(basis_function, geometric_mapping) *
                       igrad(solution_expression, geometric_mapping).cwisetr(),
                   meas_expr_dx); // Validated

      // Start assembly
      expr_assembler_pde.assemble(first + second + third);
      sensitivities_wrt_ctps.block(0, i_dimension * number_of_ctp_dofs, 1,
                                   number_of_ctp_dofs) =
          lagrange_multipliers_ptr->transpose() * expr_assembler_pde.matrix();

      // Prepare Matrix for future use
      expr_assembler_pde.clearMatrix(true);

      // Same for source term
      if (has_source_id) {
        // Linear Form Part
        auto source_function_expression =
            expr_assembler_pde.getCoeff(source_function, geometric_mapping);
        auto lin_form_derivative =
            multiply(basis_function * source_function_expression, meas_expr_dx);
        expr_assembler_pde.assemble(lin_form_derivative);

        // Add contribution to the global sensitivities vector
        sensitivities_wrt_ctps.block(0, i_dimension * number_of_ctp_dofs, 1,
                                     number_of_ctp_dofs) -=
            lagrange_multipliers_ptr->transpose() * expr_assembler_pde.matrix();

        // Prepare Matrix for future use
        expr_assembler_pde.clearMatrix(true);
      }
    }

    // Write eigen matrix into a py::array
    py::array_t<double> sensitivities_py(sensitivities_wrt_ctps.size());
    double *sensitivities_py_ptr =
        static_cast<double *>(sensitivities_py.request().ptr);
    for (int i{}; i < sensitivities_wrt_ctps.size(); i++) {
      sensitivities_py_ptr[i] = sensitivities_wrt_ctps(0, i);
    }

    // Clear for future evaluations
    expr_assembler_pde.clearMatrix(true);
    expr_assembler_pde.clearRhs();

    return sensitivities_py;
  }

  void ReadParameterSensitivities(
      std::string filename // Filename for parametrization
  ) {
    const Timer timer("ReadParameterSensitivities");
    gsFileData<> fd(filename);
    gsMultiPatch<> mp;
    fd.getId(0, mp);
    gsMatrix<index_t> patch_supports;
    fd.getId(10, patch_supports);

    const int design_dimension = patch_supports.col(1).maxCoeff() + 1;

    // Degree-elevations
    mp.degreeElevate(n_degree_elevations);

    // h-refine each basis
    for (int r = 0; r < n_refinements; ++r) {
      mp.uniformRefine();
    }

    // Start the assignment
    if (!dof_mapper_ptr) {
      throw std::runtime_error("System has not been initialized");
    }

    // Start the assignment
    const size_t totalSz = dof_mapper_ptr->freeSize();
    if (!ctps_sensitivities_matrix_ptr) {
      ctps_sensitivities_matrix_ptr = std::make_shared<gsMatrix<>>();
      ctps_sensitivities_matrix_ptr->resize(totalSz * dimensionality_,
                                            design_dimension);
    }
    // Reinit to zero
    (*ctps_sensitivities_matrix_ptr) *= 0.;

    // Rough overestimate to avoid realloations
    for (int patch_support{}; patch_support < patch_supports.rows();
         patch_support++) {
      const int j_patch = patch_supports(patch_support, 0);
      const int i_design = patch_supports(patch_support, 1);
      const int k_index_offset = patch_supports(patch_support, 2);
      for (size_t l_dof = 0; l_dof != dof_mapper_ptr->patchSize(j_patch, 0);
           l_dof++) {
        if (dof_mapper_ptr->is_free(l_dof, j_patch, 0)) {
          const int global_id = dof_mapper_ptr->index(l_dof, j_patch, 0);
          for (index_t k_dim = 0; k_dim != dimensionality_; k_dim++) {
            if (global_id + k_dim * totalSz >= totalSz * dimensionality_) {
              throw std::runtime_error("Dof Mapper is not working properly.");
            }
            ctps_sensitivities_matrix_ptr->operator()(
                global_id + k_dim * totalSz, i_design) =
                // static_cast<double>
                (mp.patch(j_patch).coef(l_dof,
                                        k_index_offset * dimensionality_));
          }
        }
      }
    }
  }

  py::array_t<double> GetParameterSensitivities() {
    if (!ctps_sensitivities_matrix_ptr) {
      throw std::runtime_error("CTPS Matrix has not been computed yet.");
    }

    const int _rows = ctps_sensitivities_matrix_ptr->rows();
    const int _cols = ctps_sensitivities_matrix_ptr->cols();
    py::array_t<double> matrix(_rows * _cols);
    double *matrix_ptr = static_cast<double *>(matrix.request().ptr);
    for (int i{}; i < _rows; i++) {
      for (int j{}; j < _cols; j++) {
        matrix_ptr[i * _cols + j] = (*ctps_sensitivities_matrix_ptr)(i, j);
      }
    }
    matrix.resize({_rows, _cols});
    return matrix;
  }

  void UpdateGeometry(const std::string &fname, const bool &topology_changes) {
    const Timer timer("UpdateGeometry");
    if (topology_changes) {
      throw std::runtime_error("Not Implemented!");
    }

    // Import mesh and load relevant information
    gsMultiPatch<> mp_new;

    gsFileData<> fd(fname);
    fd.getId(0, mp_new);

    // This update does not require refinement or elevation, in theory the mp is
    // not touched, only the solution field

    // Ignore all other information!
    if (mp_new.nPatches() != mp_pde.nPatches()) {
      throw std::runtime_error(
          "This does not work - I am fucked. Expected number of "
          "patches " +
          std::to_string(mp_pde.nPatches()) + ", but got " +
          std::to_string(mp_new.nPatches()));
    }
    // Manually update coefficients as to not overwrite any precomputed
    // values
    for (size_t patch_id{}; patch_id < mp_new.nPatches(); patch_id++) {
      if (mp_new.patch(patch_id).coefs().size() !=
          mp_pde.patch(patch_id).coefs().size()) {
        throw std::runtime_error(
            "This does not work - I am fucked. Expected number of "
            "coefficients " +
            std::to_string(mp_pde.patch(patch_id).coefs().size()) +
            ", but got " +
            std::to_string(mp_new.patch(patch_id).coefs().size()));
      }
      for (int i_coef = 0; i_coef != mp_pde.patch(patch_id).coefs().size();
           i_coef++) {
        mp_pde.patch(patch_id).coefs().at(i_coef) =
            mp_new.patch(patch_id).coefs().at(i_coef);
      }
    }
    // geometry_expression_ptr->copyCoefs(mp_new);
  }

private:
  // -------------------------
  /// First Lame constant
  real_t thermal_diffusivity_{1.};

  // -------------------------
  /// Expression assembler related to the forward problem
  gsExprAssembler<> expr_assembler_pde;

  /// Expression assembler related to the forward problem
  std::shared_ptr<gsExprEvaluator<>> expr_evaluator_ptr;

  /// Multipatch object of the forward problem
  gsMultiPatch<> mp_pde;

  /// Expression that describes the last calculated solution
  std::shared_ptr<solution> solution_expression_ptr = nullptr;

  /// Expression that describes the last calculated solution
  std::shared_ptr<space> basis_function_ptr = nullptr;

  /// Expression that describes the geometry's trial functions that are
  /// multi-dimensional
  std::shared_ptr<space> geometry_function_space_ptr = nullptr;

  /// Expression that describes the last calculated solution
  std::shared_ptr<geometryMap> geometry_expression_ptr = nullptr;

  /// Global reference to solution vector
  gsMatrix<> solVector{};

  /// Boundary conditions pointer
  gsBoundaryConditions<> boundary_conditions;

  /// Source function
  gsFunctionExpr<> source_function{};

  /// Solution function
  gsFunctionExpr<> analytical_solution{};

  /// Target Solution function for objective function
  gsFunctionExpr<> target_solution{};

  /// Solution function flag
  bool has_solution{false};

  // Flag for source function
  bool has_source_id{false};

  // Flag target id
  bool has_target_id{false};
  /// Function basis
  gsMultiBasis<> function_basis{};

  // Linear System Matrix
  std::shared_ptr<const gsSparseMatrix<>> system_matrix = nullptr;

  // Linear System Matrix
  std::shared_ptr<gsMatrix<>> ctps_sensitivities_matrix_ptr = nullptr;

  // Linear System RHS
  std::shared_ptr<gsMatrix<>> system_rhs = nullptr;

  // Solution of the adjoint problem
  std::shared_ptr<gsMatrix<>> lagrange_multipliers_ptr = nullptr;

  // DOF-Mapper
  std::shared_ptr<gsDofMapper> dof_mapper_ptr = nullptr;

  // Number of refinements in the current iteration
  int n_refinements{};

  // Number of degree elevations
  int n_degree_elevations{};

  // Number of refinements in the current iteration
  int dimensionality_{};

#ifdef PYGADJOINTS_USE_OPENMP
  int n_omp_threads{1};
#endif
};

} // namespace pygadjoints
