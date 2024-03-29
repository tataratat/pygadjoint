"""
This script creates a Quarter of an extrusion die with a Slitprofile.
The resulting geometry is used in the paper as a second numerical example.

The first part of the script, defines the contour geometry parameters, such as
the height and length of the Slit as well as the outer radius of the die.
Thereafter the geometry is created based on this approach. Other than for the
test case used at the SIAM conference, no parts of the geometry are
determined via spline approximation.
"""
import numpy as np
import splinepy as spp


def create_volumetric_die(
    center_knot=0.4,
    die_outer_radius=0.1,
    total_slit_width=0.1,
    slit_height=0.004,
    total_slit_width_m=0.11,
    slit_height_m=0.0075,
    total_depth=0.1,
    inlet_radius=0.025,
):
    """
    Create an extrusion die as given in the paper

    "Shape Optimization for Temperature Regulation in Extrusion Dies Using
    Microstructures", J. Zwar, G. Elber and S. Elgeti
    with DOI: 10.1115/1.4056075

    Contrary to the paper, we will create a quadratic (not cubic) approximation

    The resulting geometry is bicubic-linear

    To achieve the best possible approximation of the cylindric outer geometry,
    we will first insert the knots before stripping the weights (all weights
    will reduce to C0 for the same reason, this will not affect the
    construction process).

    Parameters
    ----------
    center knot : float
      Center knot is the divider on the outer surface that corresponds to the
      connection point between the quarter circle and the straight edge of the
      inner surface
    die_outer_radius : float
      Outer radius of the geometry
    total_slit_length : float
      Length of the slit including the quarter circle
    slit_height : float
      height of the slit (total)
    total_slit_width_m : float
      Length of the slit including the quarter circle in the intermediate layer
    slit_height_m : float
      height of the slit (total) in the intermediate layer
    total_depths : float
      Total depths of the extrusion die
    inlet_radius : float
      Radius of the flow-channel's inlet

    Returns
    -------
    extrusion_die : BSpline
    """

    def conjoin_splines(start_spline, end_spline):
        # Make sure that the control points and knots match
        assert start_spline.para_dim == end_spline.para_dim
        assert start_spline.para_dim == 1
        assert start_spline.degrees[0] == end_spline.degrees[0]
        assert np.allclose(start_spline.cps[-1, :], end_spline.cps[0, :])
        assert np.allclose(
            start_spline.kvs[0][-start_spline.degrees[0] :],
            end_spline.kvs[0][: end_spline.degrees[0]],
        )

        # Join the splines
        return spp.BSpline(
            degrees=start_spline.degrees,
            knot_vectors=[
                [
                    *start_spline.kvs[0][: -start_spline.degrees[0]],
                    *end_spline.kvs[0][end_spline.degrees[0] :],
                ]
            ],
            control_points=np.vstack(
                (start_spline.cps[:-1, :], end_spline.cps)
            ),
        )

    # Create the outer geometry
    outer_arc_nurbs = spp.helpme.create.arc(
        radius=die_outer_radius, degree=True, angle=90, n_knot_spans=1
    ).nurbs.create.embedded(3)

    # Insert the knots and refine once with double knots for C0 continuity
    outer_arc_nurbs.insert_knots(
        0, [center_knot / 2, center_knot, (1 + center_knot) / 2]
    )
    outer_arc_nurbs.insert_knots(
        0, [center_knot / 2, center_knot, (1 + center_knot) / 2]
    )

    # Strip it of the weights
    outer_arc_nurbs_dict = outer_arc_nurbs.todict()
    outer_arc_nurbs_dict.pop("weights")
    outer_arc = spp.BSpline(**outer_arc_nurbs_dict)

    # Create the three lines that define the inner geometry
    straight_line_length = (total_slit_width - slit_height) * 0.5
    half_slit_height = slit_height * 0.5
    straight_line_outlet = spp.Bezier(
        degrees=[1],
        control_points=[
            [straight_line_length, half_slit_height],
            [0, half_slit_height],
        ],
    ).bspline.create.embedded(3)
    straight_line_outlet.elevate_degrees([0])
    straight_line_outlet.insert_knots(0, [0.5, 0.5])
    straight_line_outlet.kvs[0].scale(center_knot, 1)

    # Create the inner arc geometry
    inner_arc_outlet_nurbs = spp.helpme.create.arc(
        radius=half_slit_height, degree=True, angle=90, n_knot_spans=1
    ).nurbs.create.embedded(3)
    inner_arc_outlet_nurbs.cps[:, 0] += straight_line_length

    # Insert the knots and refine once with double knots for C0 continuity
    inner_arc_outlet_nurbs.insert_knots(0, [0.5, 0.5])
    inner_arc_outlet_nurbs.kvs[0].scale(0, center_knot)

    # Strip it of the weights
    inner_arc_outlet_dict = inner_arc_outlet_nurbs.todict()
    inner_arc_outlet_dict.pop("weights")
    inner_arc_outlet = spp.BSpline(**inner_arc_outlet_dict)

    # inner boundary
    inner_curve_outlet = conjoin_splines(
        inner_arc_outlet, straight_line_outlet
    )

    # Second Layer
    # Create the three lines that define the inner geometry
    straight_line_length_m = (total_slit_width_m - slit_height_m) * 0.5
    half_slit_height_m = slit_height_m * 0.5
    half_depth = 0.5 * total_depth
    straight_line_m = spp.Bezier(
        degrees=[1],
        control_points=[
            [straight_line_length_m, half_slit_height_m, half_depth],
            [0, half_slit_height_m, half_depth],
        ],
    ).bspline
    straight_line_m.elevate_degrees([0])
    straight_line_m.insert_knots(0, [0.5, 0.5])
    straight_line_m.kvs[0].scale(center_knot, 1)

    # Create the inner arc geometry
    inner_arc_m_nurbs = spp.helpme.create.arc(
        radius=half_slit_height_m, degree=True, angle=90, n_knot_spans=1
    ).nurbs.create.embedded(3)
    inner_arc_m_nurbs.cps[:] += [straight_line_length_m, 0, half_depth]

    # Insert the knots and refine once with double knots for C0 continuity
    inner_arc_m_nurbs.insert_knots(0, [0.5, 0.5])
    inner_arc_m_nurbs.kvs[0].scale(0, center_knot)

    # Strip it of the weights
    inner_arc_m_dict = inner_arc_m_nurbs.todict()
    inner_arc_m_dict.pop("weights")
    inner_arc_m = spp.BSpline(**inner_arc_m_dict)

    # inner boundary
    inner_curve_m = conjoin_splines(inner_arc_m, straight_line_m)

    # Outer arc in the center
    outer_arc_m = outer_arc.copy()
    outer_arc_m.cps[:, 2] += half_depth

    # Outer arc at the inlet
    outer_arc_inlet = outer_arc.copy()
    outer_arc_inlet.cps[:, 2] += total_depth

    # Inlet arc
    inner_arc_inlet = outer_arc.copy()
    inner_arc_inlet.cps[:, 2] += total_depth
    inner_arc_inlet.cps[:, :2] *= inlet_radius / die_outer_radius

    # Combine all spline curves into a trivariate
    slit_profile = spp.BSpline(
        degrees=[2, 1, 2],
        knot_vectors=[
            outer_arc_inlet.kvs[0],
            [0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1],
        ],
        control_points=np.vstack(
            (
                inner_curve_outlet.cps,
                outer_arc.cps,
                inner_curve_m.cps,
                outer_arc_m.cps,
                inner_arc_inlet.cps,
                outer_arc_inlet.cps,
            )
        ),
    )
    spp.helpme.reparametrize.permute_parametric_axes(slit_profile, [0, 2, 1])

    return slit_profile


def create_volumetric_die_lin(
    center_knot=0.4,
    die_outer_radius=0.1,
    total_slit_width=0.1,
    slit_height=0.004,
    total_depth=0.1,
    inlet_radius=0.025,
):
    """
    Create an extrusion die as given in the paper

    "Shape Optimization for Temperature Regulation in Extrusion Dies Using
    Microstructures", J. Zwar, G. Elber and S. Elgeti
    with DOI: 10.1115/1.4056075

    Contrary to the paper, we will create a quadratic (not cubic) approximation

    The resulting geometry is bicubic-linear

    To achieve the best possible approximation of the cylindric outer geometry,
    we will first insert the knots before stripping the weights (all weights
    will reduce to C0 for the same reason, this will not affect the
    construction process).

    Parameters
    ----------
    center knot : float
      Center knot is the divider on the outer surface that corresponds to the
      connection point between the quarter circle and the straight edge of the
      inner surface
    die_outer_radius : float
      Outer radius of the geometry
    total_slit_length : float
      Length of the slit including the quarter circle
    slit_height : float
      height of the slit (total)
    total_slit_width_m : float
      Length of the slit including the quarter circle in the intermediate layer
    slit_height_m : float
      height of the slit (total) in the intermediate layer
    total_depths : float
      Total depths of the extrusion die
    inlet_radius : float
      Radius of the flow-channel's inlet

    Returns
    -------
    extrusion_die : BSpline
    """

    def conjoin_splines(start_spline, end_spline):
        # Make sure that the control points and knots match
        assert start_spline.para_dim == end_spline.para_dim
        assert start_spline.para_dim == 1
        assert start_spline.degrees[0] == end_spline.degrees[0]
        assert np.allclose(start_spline.cps[-1, :], end_spline.cps[0, :])
        assert np.allclose(
            start_spline.kvs[0][-start_spline.degrees[0] :],
            end_spline.kvs[0][: end_spline.degrees[0]],
        )

        # Join the splines
        return spp.BSpline(
            degrees=start_spline.degrees,
            knot_vectors=[
                [
                    *start_spline.kvs[0][: -start_spline.degrees[0]],
                    *end_spline.kvs[0][end_spline.degrees[0] :],
                ]
            ],
            control_points=np.vstack(
                (start_spline.cps[:-1, :], end_spline.cps)
            ),
        )

    # Create the outer geometry
    outer_arc_nurbs = spp.helpme.create.arc(
        radius=die_outer_radius, degree=True, angle=90, n_knot_spans=1
    ).nurbs.create.embedded(3)

    # Insert the knots and refine once with double knots for C0 continuity
    outer_arc_nurbs.insert_knots(
        0,
        [
            # center_knot / 2,
            # center_knot / 2,
            center_knot,
            center_knot,
            # (1 + center_knot) / 2
            # (1 + center_knot) / 2,
        ],
    )
    # Strip it of the weights
    outer_arc_nurbs_dict = outer_arc_nurbs.todict()
    outer_arc_nurbs_dict.pop("weights")
    outer_arc = spp.BSpline(**outer_arc_nurbs_dict)

    # Create the three lines that define the inner geometry
    straight_line_length = (total_slit_width - slit_height) * 0.5
    half_slit_height = slit_height * 0.5
    straight_line_outlet = spp.Bezier(
        degrees=[1],
        control_points=[
            [straight_line_length, half_slit_height],
            [0, half_slit_height],
        ],
    ).bspline.create.embedded(3)
    straight_line_outlet.elevate_degrees([0])
    # straight_line_outlet.insert_knots(0, [0.5, 0.5])
    straight_line_outlet.kvs[0].scale(center_knot, 1)

    # Create the inner arc geometry
    inner_arc_outlet_nurbs = spp.helpme.create.arc(
        radius=half_slit_height, degree=True, angle=90, n_knot_spans=1
    ).nurbs.create.embedded(3)
    inner_arc_outlet_nurbs.cps[:, 0] += straight_line_length

    # Insert the knots and refine once with double knots for C0 continuity
    # inner_arc_outlet_nurbs.insert_knots(0, [0.5, 0.5])
    inner_arc_outlet_nurbs.kvs[0].scale(0, center_knot)

    # Strip it of the weights
    inner_arc_outlet_dict = inner_arc_outlet_nurbs.todict()
    inner_arc_outlet_dict.pop("weights")
    inner_arc_outlet = spp.BSpline(**inner_arc_outlet_dict)

    # inner boundary
    inner_curve_outlet = conjoin_splines(
        inner_arc_outlet, straight_line_outlet
    )

    # Outer arc at the inlet
    outer_arc_inlet = outer_arc.copy()
    outer_arc_inlet.cps[:, 2] += total_depth

    # Inlet arc
    inner_arc_inlet = outer_arc.copy()
    inner_arc_inlet.cps[:, 2] += total_depth
    inner_arc_inlet.cps[:, :2] *= inlet_radius / die_outer_radius

    # Combine all spline curves into a trivariate
    slit_profile = spp.BSpline(
        degrees=[2, 1, 1],
        knot_vectors=[
            outer_arc_inlet.kvs[0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ],
        control_points=np.vstack(
            (
                inner_curve_outlet.cps,
                outer_arc.cps,
                inner_arc_inlet.cps,
                outer_arc_inlet.cps,
            )
        ),
    )
    spp.helpme.reparametrize.permute_parametric_axes(slit_profile, [0, 2, 1])

    return slit_profile


if __name__ == "__main__":
    test = create_volumetric_die()
    print(test.integrate.volume())
    # test.show()
    test = create_volumetric_die_lin(total_slit_width=0.11, slit_height=0.05)
    print(test.integrate.volume())
    test.show()
    mp = spp.Multipatch(test.extract.beziers())
    mp.boundaries_from_continuity()
    bb = []
    for i_color, boundary in enumerate(mp.boundaries):
        for i, j in zip(*boundary):
            bb.append(mp.patches[i].extract.boundaries()[j])
            bb[-1].show_options["c"] = i_color
    spp.show(bb, control_points=False, knots=True)

    a = spp.Bezier(
        degrees=[1, 1],
        control_points=[
            [0, 0],
            [5, 0],
            [4, 6],
            [7, 3],
        ],
    ).create.extruded([0, 0, 1])
    b = spp.Bezier(
        degrees=[1, 1], control_points=[[4, 6], [7, 3], [8, 3], [9, 2]]
    ).create.extruded([0, 0, 1])
