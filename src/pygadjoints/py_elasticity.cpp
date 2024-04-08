#include "pygadjoints/py_elasticity.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

void add_elasticity_problem(py::module_ &m) {
  py::class_<pygadjoints::LinearElasticityProblem> klasse(
      m, "LinearElasticityProblem");
  klasse
      .def(py::init<>())
      // Manipulating the problem and initialization
      .def("init", &pygadjoints::LinearElasticityProblem::Init,
           py::arg("fname"), py::arg("refinements"),
           py::arg("degree_elevations"), py::arg("print_summary") = false)
      .def("set_material_constants",
           &pygadjoints::LinearElasticityProblem::SetMaterialConstants,
           py::arg("lame_lambda"), py::arg("lame_mu"))
      .def("set_objective_function",
           &pygadjoints::LinearElasticityProblem::SetObjectiveFunction,
           py::arg("objective_function"))

      // IO routines
      .def("read_control_point_sensitivities",
           &pygadjoints::LinearElasticityProblem::ReadParameterSensitivities,
           py::arg("fname"))
      .def("get_control_point_sensitivities",
           &pygadjoints::LinearElasticityProblem::GetParameterSensitivities)
      .def("update_geometry",
           &pygadjoints::LinearElasticityProblem::UpdateGeometry,
           py::arg("fname"), py::arg("topology_changes"))
      .def("export_paraview",
           &pygadjoints::LinearElasticityProblem::ExportParaview,
           py::arg("fname"), py::arg("plot_elements"), py::arg("sample_rate"),
           py::arg("binary"))
      .def("export_xml", &pygadjoints::LinearElasticityProblem::ExportXML,
           py::arg("fname"))
      .def("export_multipatch_object",
           &pygadjoints::LinearElasticityProblem::ExportMultipatchSolution,
           py::arg("fname"))

      // Assembly and Solving the linear problem
      .def("assemble", &pygadjoints::LinearElasticityProblem::Assemble)
      .def("solve_linear_system",
           &pygadjoints::LinearElasticityProblem::SolveLinearSystem)
      .def("solve_adjoint_system",
           &pygadjoints::LinearElasticityProblem::SolveAdjointProblem)

      // Scalar measures and their derivatives
      .def("volume", &pygadjoints::LinearElasticityProblem::ComputeVolume)
      .def("volume_deris_wrt_ctps",
           &pygadjoints::LinearElasticityProblem::ComputeVolumeDerivativeToCTPS)
      .def("objective_function",
           &pygadjoints::LinearElasticityProblem::ComputeObjectiveFunction)
      .def("objective_function_deris_wrt_ctps",
           &pygadjoints::LinearElasticityProblem::
               ComputeObjectiveFunctionDerivativeWrtCTPS)

  // OpenMP specifics
#ifdef PYGADJOINTS_USE_OPENMP
      .def("set_number_of_threads",
           &pygadjoints::LinearElasticityProblem::SetNumberOfThreads,
           py::arg("nthreads"))
#endif
      ;
}
