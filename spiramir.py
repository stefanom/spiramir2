from math import cos, sin, log, exp, atan, pi

import random, time

import bpy
from bpy_extras.object_utils import object_data_add
from mathutils import Vector, Matrix

from . import utils


class CURVE_PT_spiramir(bpy.types.Panel):
    bl_idname = "CURVE_PT_spiramir"
    bl_label = "Spiramir Grower"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Create"

    def draw(self, context):
        parent_curve = context.active_object
        # If the selected object is a spiramir, the props defaults
        # should change to align with growth coming from that parent.
        if parent_curve and parent_curve.type == 'CURVE':
            props = self.layout.operator('curve.spiramir')
            props.position = 0.5
            props.radius = parent_curve['spiramir_radius'] * 0.9
            props.direction = utils.invert_direction(
                parent_curve['spiramir_direction'])
            props.winding_factor = parent_curve['spiramir_winding_factor']
            props.curvature_error = parent_curve['spiramir_curvature_error']
            props.starting_angle = parent_curve['spiramir_starting_angle']


class CURVE_OT_spiramir(bpy.types.Operator):
    bl_idname = "curve.spiramir"
    bl_label = "Spiramir"
    bl_options = {'REGISTER', 'UNDO'}

    verbose = False

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

    tube_radius: bpy.props.FloatProperty(
        default=0.01,
        min=0.0, max=1000.0,
        description="Radius of the tube around the curve"
    )
    position: bpy.props.FloatProperty(
        default=0.0,
        min=0.0, max=100.0,
        description="Relative position of the growth point"
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

    def draw(self, context):
        layout = self.layout
        col = layout.column_flow(align=True)

        col.prop(self, "direction")

        col = layout.column(align=True)
        col.label(text="Spiral Parameters:")
        col.prop(self, "radius", text="Radius")
        col.prop(self, "position", text="Relative Position")
        col.prop(self, "winding_factor", text="Winding Factor")
        col.prop(self, "starting_angle", text="Starting Angle")
        col.prop(self, "curvature_error", text="Curvature Error")
        col.prop(self, "tube_radius", text="Tube Radius")

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

    def execute(self, context):
        self.log("\n\n=========================== Execute ==================================")
        time_start = time.time()

        parent_curve = context.object

        if parent_curve:
          if bpy.context.mode == 'EDIT_CURVE':
            selected_point_position, selected_point_weight = utils.get_selected_point_position_and_weight(
                parent_curve)
            if selected_point_position:
              self.position = selected_point_position
              self.radius = self.max_growth_ratio * utils.spiral_radius_at_length(
                  selected_point_weight, self.winding_factor)
            bpy.ops.object.mode_set(mode='OBJECT')
          elif (parent_curve.type != 'CURVE' or not parent_curve.select_get()):
            parent_curve = None

        origin = Vector((0, 0, 0))
        bpy.ops.object.empty_add(type='ARROWS', location=origin)
        empty = context.object

        if parent_curve:
            constraint = empty.constraints.new('FOLLOW_PATH')
            constraint.target = parent_curve
            constraint.offset_factor = self.position
            constraint.use_curve_follow = True
            constraint.use_fixed_location = True
            constraint.forward_axis = 'FORWARD_X'
            constraint.up_axis = 'UP_Z'

            if not parent_curve.data.animation_data:
                override = {'constraint': constraint}
                bpy.ops.constraint.followpath_path_animate(
                    override, constraint='Follow Path')

        data_curve = bpy.data.curves.new(name='Spiramir', type='CURVE')
        data_curve.dimensions = '2D'
        data_curve.bevel_depth = self.tube_radius

        curve = object_data_add(context, data_curve)

        curve['spiramir'] = True
        curve['spiramir_radius'] = self.radius
        curve['spiramir_direction'] = self.direction
        curve['spiramir_winding_factor'] = self.winding_factor
        curve['spiramir_curvature_error'] = self.curvature_error
        curve['spiramir_starting_angle'] = self.starting_angle

        curve.parent = empty

        spiral, _, _ = utils.make_spiral(
            self.radius, self.winding_factor, self.curvature_error, self.starting_angle, self.direction)

        utils.add_spline_to_curve(curve, spiral)

        self.log("Drawing took %.4f sec." % (time.time() - time_start))

        context.view_layer.update()

        return {'FINISHED'}

# TODO
# - grow from selected vertex
# - iterative growth
