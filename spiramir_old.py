from math import cos, sin, log, exp, atan, pi

import random
import time

import bpy
from bpy_extras.object_utils import object_data_add
from mathutils import Vector, Matrix

from . import utils

class CURVE_OT_spiramir_old(bpy.types.Operator):
    bl_idname = "curve.spiramir_old"
    bl_label = "Spiramir"
    bl_description = "Create a spiramir (old)"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_options = {'REGISTER', 'UNDO'}

    direction: bpy.props.EnumProperty(
        items=[('COUNTER_CLOCKWISE', "Counter Clockwise",
                "Wind in a counter clockwise direction"),
               ("CLOCKWISE", "Clockwise",
                "Wind in a clockwise direction")],
        default='COUNTER_CLOCKWISE',
        name="Direction",
        description="Direction of winding"
    )
    radius: bpy.props.FloatProperty(
        default=1.0,
        min=0.00001, max=1000.0,
        description="Radius of the spiral"
    )
    winding_factor: bpy.props.FloatProperty(
        default=0.2,
        min=0.0001, max=2,
        description="Spiral Winding Factor"
    )
    starting_angle: bpy.props.FloatProperty(
        default=-20.0,
        min=-100.0, max=100.0,
        description="Angle to start drawing the spiral"
    )
    curvature_error: bpy.props.FloatProperty(
        default=0.001,
        min=0.0000001, max=1000.0,
        description="Maximum curvature error"
    )

    min_growth_ratio: bpy.props.FloatProperty(
        default=0.6,
        min=0.0, max=1.0,
        description="Minimum growth ratio"
    )
    max_growth_ratio: bpy.props.FloatProperty(
        default=0.95,
        min=0.0, max=1.0,
        description="Maximum growth ratio"
    )
    iterations: bpy.props.IntProperty(
        default=2,
        min=1, max=500,
        description="Number of recursive iterations"
    )
    max_attempts: bpy.props.IntProperty(
        default=2,
        min=1, max=10000,
        description="Maximum number of placement attempts during recursion"
    )
    min_distance: bpy.props.FloatProperty(
        default=0.001,
        min=0.0001, max=1.0,
        description="The minimum distance between spirals"
    )
    offset: bpy.props.IntProperty(
        default=25,
        min=1, max=1000,
        description="Number of vertices forbidden from spawning children at beginning at end of spiral"
    )
    
    edit_mode: bpy.props.BoolProperty(
        name="Show in edit mode",
        default=False,
        description="Show in edit mode"
    )
    verbose: bpy.props.BoolProperty(
        name="Verbose logging",
        default=False,
        description="Print verbose logging in the debug console"
    )

    def draw(self, context):
        layout = self.layout
        col = layout.column_flow(align=True)

        col.prop(self, "direction")

        col = layout.column(align=True)
        col.label(text="Spiral Parameters:")
        col.prop(self, "radius", text="Radius")
        col.prop(self, "winding_factor", text="Winding Factor")
        col.prop(self, "starting_angle", text="Starting Angle")
        col.prop(self, "curvature_error", text="Curvature Error")

        col = layout.column(align=True)
        col.label(text="Construction Flags:")
        col.row().prop(self, "edit_mode", expand=True)
        col.row().prop(self, "verbose", expand=True)

        col = layout.column(align=True)
        col.label(text="Recursion Parameters:")
        col.prop(self, "iterations", text="Recursion Iterations")
        col.prop(self, "max_attempts", text="Max Attempts")
        col.prop(self, "offset", text="Growth Offset")
        col.prop(self, "min_growth_ratio", text="Min Growth Ratio")
        col.prop(self, "max_growth_ratio", text="Max Growth Ratio")
        col.prop(self, "min_distance", text="Minimum Distance")

    def log(self, message, *args):
        if self.verbose:
            if args:
                print(message, args)
            else:
                print(message)

    def get_splines_window(self):
        return 2

    def get_selected_point(self, curve):
        selected_points = []
        for splines in utils.windowed(curve.data.splines, self.get_splines_window()):
            spiral = splines[0]
            for points in utils.windowed(spiral.points, 2):
                if points[1].select and points[0]:
                    selected_points.append(points)
        if selected_points:
            return selected_points[-1]


    def execute(self, context):
        self.log("\n\n=========================== Execute ==================================")
        time_start = time.time()

        origin = Vector((0, 0, 0))
        tangent_angle = 0.0
        radius = self.radius
        direction = self.direction
        spirals = []
        spirals_centripetals = []
        fertile_spirals = []
        fertile_spirals_weights = []
        attempts = 1
        aborted = 0

        if bpy.context.mode == 'EDIT_CURVE':
            curve = context.active_object
            previous_point, selected_point = self.get_selected_point(curve)
            if selected_point:
                origin = selected_point.co[0:3]
                length = abs(selected_point.weight)
                direction = utils.invert_direction(
                    utils.direction_from_sign(selected_point.weight))
                tangent_angle = utils.angle_between_points(
                    previous_point.co, selected_point.co)
                radius = self.max_growth_ratio * \
                    utils.spiral_radius_at_length(length, self.winding_factor)

                for spiral, centripetal in utils.windowed(curve.data.splines, self.get_splines_window()):
                    if len(spiral.points[0].co) == 4 and spiral.points[0].co[4] != 0 and len(centripetal.points) == 2:
                        spirals.append(spiral)
                        spirals_centripetals.append(centripetal)
                        if len(spiral.points) > 2 * self.offset:
                            fertile_spirals.append(spiral)
                            fertile_spirals_weights.append(len(spiral.points))
        else:
            data_curve = bpy.data.curves.new(name='Spiramir', type='CURVE')

            curve = object_data_add(context, data_curve)
            curve.matrix_world = utils.get_align_matrix(context, origin)
            curve.select_set(True)

            curve.data.dimensions = '2D'
            curve.data.use_path = True
            curve.data.fill_mode = 'BOTH'

            curve['spiramir_winding_factor'] = self.winding_factor
            curve['spiramir_curvature_error'] = self.curvature_error
            curve['spiramir_starting_angle'] = self.starting_angle

        spiral, spiral_radius, spiral_mass_center = utils.make_spiral(
            radius, self.winding_factor, self.curvature_error, self.starting_angle, direction)

        for _ in range(self.max_attempts):
            if spiral:
                spline = utils.add_spline_to_curve(curve, spiral, origin)

                spirals.append(spline)

                centripetal_vector = utils.make_centripetal_vector(
                    spiral_mass_center)
                centripetal_spline = utils.add_spline_to_curve(
                    curve, centripetal_vector, origin)
                spirals_centripetals.append(centripetal_spline)

                if len(spline.points) > 2 * self.offset:
                    fertile_spirals.append(spline)
                    fertile_spirals_weights.append(len(spline.points))

                if len(spirals) >= self.iterations:
                    break

            self.log("fertile spirals: ", fertile_spirals)

            mother_spiral = random.choices(
                fertile_spirals, fertile_spirals_weights, k=1)[0]
            self.log('mother spiral: ', mother_spiral)

            contact_point_index = random.randrange(
                self.offset, len(mother_spiral.points) - self.offset)
            contact_point = mother_spiral.points[contact_point_index]
            previous_point = mother_spiral.points[contact_point_index - 1]
            origin = contact_point.co[0:3]

            self.log('contact point: ', origin, contact_point_index)

            length = abs(contact_point.weight)
            direction = utils.invert_direction(
                utils.direction_from_sign(contact_point.weight))
            tangent_angle = utils.angle_between_points(
                previous_point.co, contact_point.co)
            mother_radius = utils.spiral_radius_at_length(
                length, self.winding_factor)
            radius = random.uniform(
                self.min_growth_ratio * mother_radius, self.max_growth_ratio * mother_radius)

            spiral, spiral_radius, spiral_mass_center = utils.make_spiral(
                radius, self.winding_factor, self.curvature_error, self.starting_angle, direction, tangent_angle)

            attempts += 1

            absolute_spiral_mass_center = utils.translate(
                origin, spiral_mass_center)
            for spiral_under_test, centripetal in zip(spirals, spirals_centripetals):
                self.log("spiral under test: ", spiral_under_test)
                if spiral_under_test != mother_spiral:
                    self.log(" testing spiral!")
                    mass_center = centripetal.points[1]
                    radius = (
                        centripetal.points[1].co - centripetal.points[0].co).length
                    d = utils.distance(mass_center.co[0:3],
                                       absolute_spiral_mass_center)
                    if d < radius + spiral_radius:
                        self.log(" circles intersect! testing spiral distance")
                        min_distance = utils.minimum_distance(
                            spiral, spiral_under_test)
                        if min_distance > self.min_distance:
                            self.log(
                                "  spirals were too close, aborting!", min_distance)
                            aborted += 1
                            spiral = None
                            break
                        else:
                            self.log(
                                "  spiral was far enough, drawing! ", min_distance)
                    else:
                        self.log("  spiral circle didn't intersect!")

        self.report({'INFO'}, "Drawing took %.4f sec. Spirals: %i, Attempts: %i, Aborted: %i" %
                    ((time.time() - time_start), len(spirals), attempts, aborted))

        if self.edit_mode:
            bpy.ops.object.mode_set(mode='EDIT')
        else:
            bpy.ops.object.mode_set(mode='OBJECT')

        self.log(">>>>>>>>>>>>>>>>>>>>>>>>>>>> done <<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

        return {'FINISHED'}
