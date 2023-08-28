#include <gismo.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#ifdef PYGADJOINTS_USE_OPENMP
#include <omp.h>
#endif

#include "pygadjoints/custom_expression.hpp"

namespace pygadjoints {

using namespace gismo;

namespace py = pybind11;

/// @brief
class LinearElasticityProblem {
  // Typedefs
  typedef gsExprAssembler<>::geometryMap geometryMap;
  typedef gsExprAssembler<>::variable variable;
  typedef gsExprAssembler<>::space space;
  typedef gsExprAssembler<>::solution solution;

  using SolverType = gsSparseSolver<>::CGDiagonal;

 public:
  LinearElasticityProblem() : expr_assembler_pde(1, 1) {
#ifdef PYGADJOINTS_USE_OPENMP
    omp_set_num_threads(std::min(omp_get_max_threads(), n_omp_threads));
#endif
  };

  gsStopwatch timer;

  /**
   * @brief Set the Material Constants
   *
   * @param lambda first lame constant
   * @param mu second lame constant
   * @param rho density
   */
  void SetMaterialConstants(const real_t& lambda, const real_t& mu,
                            const real_t& rho) {
    lame_lambda_ = lambda;
    lame_mu_ = mu;
    rho_ = rho;
  }

  /**
   * @brief Export the results as Paraview vtu file
   *
   * @param fname Filename
   * @param plot_elements Plot patch borders
   * @param sample_rate Samplerate (samples per element)
   * @return void
   */
  void ExportParaview(const std::string& fname, const bool& plot_elements,
                      const int& sample_rate, const bool& export_b64) {
    // Generate Paraview File
    gsExprEvaluator<> expression_evaluator(expr_assembler_pde);
    gsParaviewCollection collection("ParaviewOutput/" + fname,
                                    &expression_evaluator);
    collection.options().setSwitch("plotElements", true);
    collection.options().setSwitch("base64", export_b64);
    collection.options().setInt("numPoints", plot_elements);
    collection.options().setInt("plotElements.resolution", sample_rate);
    collection.options().setInt("numPoints", sample_rate);
    collection.newTimeStep(&mp_pde);
    collection.addField(*solution_expression_ptr, "displacement");
    collection.saveTimeStep();
    collection.save();
  }

  /**
   * @brief Export the results in xml file
   *
   * @param fname Filename
   * @return double Elapsed time
   */
  double ExportXML(const std::string& fname) {
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

  void print(const int i) {
    for (int j{}; j < i; j++) {
      gsWarn << "TEST IF GISMO IS THERE SUCCEEDED, I AM YOUR\n";
    }
  }

#ifdef PYGADJOINTS_USE_OPENMP
  /**
   * @brief Set the Number Of Threads for OpenMP
   *
   * Somehow does not compile
   * @param n_threads
   */
  void SetNumberOfThreads(const int& n_threads) {
    n_omp_threads = n_threads;
    omp_set_num_threads(n_threads);
    // gsInfo << "Available threads: " << omp_get_max_threads() << "\n";
  }
#endif

  void ReadInputFromFile(const std::string& filename) {
    // IDs in the text file (might change later)
    const int mp_id{0}, source_id{1}, bc_id{2}, ass_opt_id{3};

    // Import mesh and load relevant information
    gsFileData<> fd(filename);
    fd.getId(mp_id, mp_pde);
    fd.getId(source_id, neumann_function);
    fd.getId(bc_id, boundary_conditions);
    boundary_conditions.setGeoMap(mp_pde);
    gsOptionList Aopt;
    fd.getId(ass_opt_id, Aopt);

    // Set Options in expression assembler
    expr_assembler_pde.setOptions(Aopt);
  }

  void Init(const int numRefine) {
    n_refinements = numRefine;

    //! [Refinement]
    function_basis = gsMultiBasis<>(mp_pde, true);

    // h-refine each basis
    for (int r = 0; r < n_refinements; ++r) {
      function_basis.uniformRefine();
    }

    // Elements used for numerical integration
    expr_assembler_pde.setIntegrationElements(function_basis);

    // Set the dimension
    dimensionality_ = mp_pde.geoDim();

    // Set the discretization space
    basis_function_ptr = std::make_shared<space>(
        expr_assembler_pde.getSpace(function_basis, dimensionality_));

    // Solution vector and solution variable
    solution_expression_ptr = std::make_shared<solution>(
        expr_assembler_pde.getSolution(*basis_function_ptr, solVector));

    // Retrieve expression that represents the geometry mapping
    geometry_expression_ptr =
        std::make_shared<geometryMap>(expr_assembler_pde.getMap(mp_pde));

    basis_function_ptr->setup(boundary_conditions, dirichlet::l2Projection, 0);

    // Initialize the system
    expr_assembler_pde.initSystem();

    // Assign a Dof Mapper
    dof_mapper_ptr =
        std::make_shared<gsDofMapper>(basis_function_ptr->mapper());
  }

  void Assemble() {
    if (!basis_function_ptr) {
      std::cerr << "ERROR";
      return;
    }

    // Auxiliary variables for readability
    geometryMap& geometric_mapping = *geometry_expression_ptr;
    const space& basis_function = *basis_function_ptr;

    // Compute the system matrix and right-hand side
    auto phys_jacobian = ijac(basis_function, geometric_mapping);
    auto bilin_lambda = lame_lambda_ * idiv(basis_function, geometric_mapping) *
                        idiv(basis_function, geometric_mapping).tr() *
                        meas(geometric_mapping);
    auto bilin_mu_1 = lame_mu_ *
                      (phys_jacobian.cwisetr() % phys_jacobian.tr()) *
                      meas(geometric_mapping);
    auto bilin_mu_2 = lame_mu_ * (phys_jacobian % phys_jacobian.tr()) *
                      meas(geometric_mapping);

    // Set the boundary_conditions term
    auto neumann_function_expression =
        expr_assembler_pde.getCoeff(neumann_function, geometric_mapping);
    auto lin_form = rho_ * basis_function * neumann_function_expression *
                    meas(geometric_mapping);

    auto bilin_combined = (bilin_lambda + bilin_mu_1 + bilin_mu_2);

    // Assemble
    expr_assembler_pde.assemble(bilin_combined);
    expr_assembler_pde.assemble(lin_form);

    // Compute the Neumann terms defined on physical space
    auto g_N = expr_assembler_pde.getBdrFunction(geometric_mapping);

    // Neumann conditions
    expr_assembler_pde.assembleBdr(
        boundary_conditions.get("Neumann"),
        basis_function * g_N * nv(geometric_mapping).norm());

    system_matrix =
        std::make_shared<const gsSparseMatrix<>>(expr_assembler_pde.matrix());
    system_rhs = std::make_shared<gsMatrix<>>(expr_assembler_pde.rhs());

    // Clear for future evaluations
    expr_assembler_pde.clearMatrix();
    expr_assembler_pde.clearRhs();
  }

  void SolveLinearSystem() {
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
    // Compute volume of domain
    gsExprEvaluator<> expression_evaluator(expr_assembler_pde);
    return expression_evaluator.integral(meas(*geometry_expression_ptr));
  }

  double ComputeObjectiveFunction() {
    // Compute volume of domain
    gsExprEvaluator<> expression_evaluator(expr_assembler_pde);
    return expression_evaluator.integral(meas(*geometry_expression_ptr));
  }

  py::array_t<double> ComputeVolumeDerivativeToCTPS() {
    // Compute the derivative of the volume of the domain with respect to the
    // control points
    // Auxiliary expressions
    const space& basis_function = *basis_function_ptr;
    auto jacobian = jac(*geometry_expression_ptr);       // validated
    auto inv_jacs = jacobian.ginv();                     // validated
    auto meas_expr = meas(*geometry_expression_ptr);     // validated
    auto djacdc = jac(basis_function);                   // validated
    auto aux_expr = (djacdc * inv_jacs).tr();            // validated
    auto meas_expr_dx = meas_expr * (aux_expr).trace();  // validated
    expr_assembler_pde.assemble(meas_expr_dx.tr());
    const auto& volume_deriv = expr_assembler_pde.rhs();

    py::array_t<double> derivative(volume_deriv.size());
    double* derivative_ptr = static_cast<double*>(derivative.request().ptr);
    for (int i{}; i < volume_deriv.size(); i++) {
      derivative_ptr[i] = volume_deriv(i, 0);
    }
    return derivative;
  }

  py::array_t<double> ComputeObjectiveFunctionDerivativeWrtCTPS() {
    // Auxiliary references
    const geometryMap& geometric_mapping = *geometry_expression_ptr;
    const space& basis_function = *basis_function_ptr;
    const solution& solution_expression = *solution_expression_ptr;

    //////////////////////////////////////
    // Derivative of Objective Function //
    //////////////////////////////////////
    expr_assembler_pde.clearRhs();
    expr_assembler_pde.assembleBdr(boundary_conditions.get("Neumann"),
                                   2 * basis_function * solution_expression *
                                       nv(geometric_mapping).norm());
    const auto objective_function_derivative = expr_assembler_pde.rhs();

    /////////////////////////////////
    // Solving the adjoint problem //
    /////////////////////////////////
    const gsSparseMatrix<> matrix_in_initial_configuration(
        system_matrix->transpose().eval());
    auto rhs_vector = expr_assembler_pde.rhs();

    // Initialize linear solver
    gsSparseSolver<>::CGDiagonal solverAdjoint;
    gsMatrix<> lagrange_multipliers;
    solverAdjoint.compute(matrix_in_initial_configuration);
    lagrange_multipliers = -solverAdjoint.solve(expr_assembler_pde.rhs());

    ////////////////////////////////
    // Derivative of the LHS Form //
    ////////////////////////////////
    expr_assembler_pde.clearMatrix();
    expr_assembler_pde.clearRhs();

    // Auxiliary expressions
    auto jacobian = jac(geometric_mapping);              // validated
    auto inv_jacs = jacobian.ginv();                     // validated
    auto meas_expr = meas(geometric_mapping);            // validated
    auto djacdc = jac(basis_function);                   // validated
    auto aux_expr = (djacdc * inv_jacs).tr();            // validated
    auto meas_expr_dx = meas_expr * (aux_expr).trace();  // validated

    // Start to assemble the bilinear form with the known solution field
    // 1. Bilinear form of lambda expression seperated into 3 individual
    // sections
    auto BL_lambda_1 =
        idiv(solution_expression, geometric_mapping).val();      // validated
    auto BL_lambda_2 = idiv(basis_function, geometric_mapping);  // validated
    auto BL_lambda =
        lame_lambda_ * BL_lambda_2 * BL_lambda_1 * meas_expr;  // validated

    // trace(A * B) = A:B^T
    auto BL_lambda_1_dx = frobenius(
        aux_expr, ijac(solution_expression, geometric_mapping));  // validated
    auto BL_lambda_2_dx =
        (ijac(basis_function, geometric_mapping) % aux_expr);  // validated

    auto BL_lambda_dx =
        lame_lambda_ * BL_lambda_2 * BL_lambda_1 * meas_expr_dx -
        lame_lambda_ * BL_lambda_2_dx * BL_lambda_1 * meas_expr -
        lame_lambda_ * BL_lambda_2 * BL_lambda_1_dx * meas_expr;  // validated

    // 2. Bilinear form of mu (first part)
    // BL_mu1_2 seems to be in a weird order with [jac0, jac2] leading
    // to [2x(2nctps)]
    auto BL_mu1_1 = ijac(solution_expression, geometric_mapping);  // validated
    auto BL_mu1_2 = ijac(basis_function, geometric_mapping);       // validated
    auto BL_mu1 = lame_mu_ * (BL_mu1_2 % BL_mu1_1) * meas_expr;    // validated

    auto BL_mu1_1_dx = -(ijac(solution_expression, geometric_mapping) *
                         aux_expr.cwisetr());  //          validated
    auto BL_mu1_2_dx =
        -(jac(basis_function) * inv_jacs * aux_expr.cwisetr());  // validated

    auto BL_mu1_dx0 =
        lame_mu_ * BL_mu1_2 % BL_mu1_1_dx * meas_expr;  // validated
    auto BL_mu1_dx1 =
        lame_mu_ * frobenius(BL_mu1_2_dx, BL_mu1_1) * meas_expr;  // validated
    auto BL_mu1_dx2 = lame_mu_ * frobenius(BL_mu1_2, BL_mu1_1).cwisetr() *
                      meas_expr_dx;  // validated

    // 2. Bilinear form of mu (first part)
    auto BL_mu2_1 =
        ijac(solution_expression, geometric_mapping).cwisetr();  // validated
    auto& BL_mu2_2 = BL_mu1_2;                                   // validated
    auto BL_mu2 = lame_mu_ * (BL_mu2_2 % BL_mu2_1) * meas_expr;  // validated

    auto inv_jac_T = inv_jacs.tr();
    auto BL_mu2_1_dx = -inv_jac_T * jac(basis_function).tr() * inv_jac_T *
                       jac(solution_expression).cwisetr();  // validated
    auto& BL_mu2_2_dx = BL_mu1_2_dx;                        // validated

    auto BL_mu2_dx0 =
        lame_mu_ * BL_mu2_2 % BL_mu2_1_dx * meas_expr;  // validated
    auto BL_mu2_dx1 =
        lame_mu_ * frobenius(BL_mu2_2_dx, BL_mu2_1) * meas_expr;  // validated
    auto BL_mu2_dx2 = lame_mu_ * frobenius(BL_mu2_2, BL_mu2_1).cwisetr() *
                      meas_expr_dx;  // validated
    // Linear Form Part
    // auto LF_1 = -rho_ * basis_function * ff * meas_expr;
    // auto LF_1_dx = -rho_ * basis_function * ff * meas_expr_dx;

    // Assemble
    expr_assembler_pde.assemble(
        BL_lambda_dx + BL_mu1_dx0 + BL_mu1_dx2 + BL_mu2_dx0 + BL_mu2_dx2,
        // + LF_1_dx,
        BL_mu1_dx1, BL_mu2_dx1);

    ///////////////////////////
    // Compute sensitivities //
    ///////////////////////////
    const auto sensitivities =
        lagrange_multipliers.transpose() * expr_assembler_pde.matrix();

    // Write eigen matrix into a py::array
    py::array_t<double> sensitivities_py(sensitivities.size());
    double* sensitivities_py_ptr =
        static_cast<double*>(sensitivities_py.request().ptr);
    for (int i{}; i < sensitivities.size(); i++) {
      sensitivities_py_ptr[i] = sensitivities(i, 0);
    }
    return sensitivities_py;
  }

  py::tuple GetParameterSensitivities(
      std::string filename  // Filename for parametrization
  ) {
    gsFileData<> fd(filename);
    gsMultiPatch<> mp;
    fd.getId(0, mp);
    gsMatrix<index_t> patch_supports;
    fd.getId(10, patch_supports);

    const int design_dimension = patch_supports.col(1).maxCoeff() + 1;
    // h-refine each basis
    for (int r = 0; r < n_refinements; ++r) {
      mp.uniformRefine();
    }

    // Start the assignement
    if (!dof_mapper_ptr) {
      throw std::runtime_error("System has not been initialized");
    }
    const size_t totalSz = dof_mapper_ptr->freeSize();

    // Create an estimate for the number of entries
    const std::size_t n_entries{patch_supports.rows() * dimensionality_ *
                                dof_mapper_ptr->patchSize(0, 0)};
    py::array_t<double> values(n_entries);
    py::array_t<int> rows(n_entries);
    py::array_t<int> cols(n_entries);

    // Retrieve pointers to data
    double* values_ptr = static_cast<double*>(values.request().ptr);
    int* rows_ptr = static_cast<int*>(rows.request().ptr);
    int* cols_ptr = static_cast<int*>(cols.request().ptr);
    std::size_t counter{};
    for (int patch_support{}; patch_support < patch_supports.rows();
         patch_support++) {
      const int j_patch = patch_supports(patch_support, 0);
      const int i_design = patch_supports(patch_support, 1);
      const int k_index_offset = patch_supports(patch_support, 2);
      for (index_t k_dim = 0; k_dim != dimensionality_; k_dim++) {
        for (size_t l_dof = 0;
             l_dof != dof_mapper_ptr->patchSize(j_patch, k_dim); l_dof++) {
          if (dof_mapper_ptr->is_free(l_dof, j_patch, k_dim)) {
            const int global_id = dof_mapper_ptr->index(l_dof, j_patch, k_dim);
            rows_ptr[counter] = global_id;
            cols_ptr[counter] = i_design;
            values_ptr[counter] = mp.patch(j_patch).coef(
                l_dof, k_dim + k_index_offset * dimensionality_);
            counter++;
            if (counter == n_entries) {
              throw std::runtime_error("Underestimated number of entries");
            }
          }
        }
      }
    }

    // Resize to actual size
    rows.resize({counter});
    cols.resize({counter});
    values.resize({counter});

    // Create a Matrix on python side in scipy using triplets
    return py::make_tuple(py::make_tuple(values, py::make_tuple(rows, cols)),
                          py::make_tuple(static_cast<int>(totalSz),
                                         static_cast<int>(design_dimension)));
  }

 private:
  // -------------------------
  /// First Lame constant
  real_t lame_lambda_{2000000};
  /// Second Lame constant
  real_t lame_mu_{500000};
  /// Density
  real_t rho_{1000};

  // -------------------------
  /// Expression assembler related to the forward problem
  gsExprAssembler<> expr_assembler_pde;

  /// Multipatch object of the forward problem
  gsMultiPatch<> mp_pde;

  /// Expression that describes the last calculated solution
  std::shared_ptr<solution> solution_expression_ptr = nullptr;

  /// Expression that describes the last calculated solution
  std::shared_ptr<space> basis_function_ptr = nullptr;

  /// Expression that describes the last calculated solution
  std::shared_ptr<geometryMap> geometry_expression_ptr = nullptr;

  /// Global reference to solution vector
  gsMatrix<> solVector{};

  /// Boundary conditions pointer
  gsBoundaryConditions<> boundary_conditions;

  /// Neumann function
  gsFunctionExpr<> neumann_function{};

  /// Function basis
  gsMultiBasis<> function_basis{};

  // Linear System Matrixn_refinements
  std::shared_ptr<const gsSparseMatrix<>> system_matrix = nullptr;

  // Linear System RHS
  std::shared_ptr<gsMatrix<>> system_rhs = nullptr;

  // DOF-Mapper
  std::shared_ptr<gsDofMapper> dof_mapper_ptr = nullptr;

  // Number of refinements in the current iteration
  int n_refinements{};

  // Number of refinements in the current iteration
  int dimensionality_{};

#ifdef PYGADJOINTS_USE_OPENMP
  int n_omp_threads{1};
#endif
};

}  // namespace pygadjoints
