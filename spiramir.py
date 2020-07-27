import random
import time

import bpy

from . import utils


class CURVE_PT_spiramir(bpy.types.Panel):
    bl_idname = "CURVE_PT_spiramir"
    bl_label = "Spiramir Grower"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Create"

    def draw(self, context):
        parent_curve = context.active_object
        if parent_curve and parent_curve.type == 'CURVE':
            self.layout.operator('curve.spiramir')


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
        default=10,
        min=1, max=500,
        description="Number of recursive iterations"
    )
    max_attempts: bpy.props.IntProperty(
        default=100,
        min=1, max=10000,
        description="Maximum number of placement attempts during recursion"
    )
    min_distance: bpy.props.FloatProperty(
        default=0.001,
        min=0.0001, max=1.0,
        description="The minimum distance between spirals"
    )
    offset: bpy.props.FloatProperty(
        default=0.15,
        min=0.0, max=1.0,
        description="Fraction of the beginning and end of the parent spiramir to avoid growting from"
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

    def get_random_growth_point(self):
        parent_curve = utils.get_random_curve()
        position = random.uniform(self.offset, 1 - self.offset)
        return parent_curve, position

    def add_recursive_spirals(self, context):
        drawn_spirals = 0

        for _ in range(self.max_attempts):
            radius = self.radius
            direction = self.direction

            parent_curve, position = self.get_random_growth_point()
            print(parent_curve, position)
            if 'spiramir' in parent_curve:
                direction = utils.invert_direction(
                    parent_curve['spiramir_direction'])
                weight = utils.get_weight_at_position(
                    parent_curve, position)
                parent_radius = utils.spiral_radius_at_length(
                    weight, self.winding_factor)
                radius = random.uniform(
                    self.min_growth_ratio * parent_radius, self.max_growth_ratio * parent_radius)
                print(weight, parent_radius, radius)
            else:
                print('non spiramir parent')

            spiral = utils.Spiral(radius=radius, direction=direction, winding_factor=self.winding_factor,
                                  curvature_error=self.curvature_error, starting_angle=self.starting_angle)

            if spiral.is_viable():
                spiral.add_to_scene(context, parent_curve=parent_curve,
                                    position=position, tube_radius=self.tube_radius)
                drawn_spirals += 1
                if drawn_spirals >= self.iterations:
                    break

    def execute(self, context):
        self.log(
            "\n\n=========================== Execute ==================================")
        time_start = time.time()

        single = False

        parent_curve = context.object

        position = self.position
        radius = self.radius
        direction = self.direction

        if parent_curve:
            if bpy.context.mode == 'EDIT_CURVE':
                selected_point_position, selected_point_weight = utils.get_selected_point(
                    parent_curve)
                if selected_point_position:
                    position = selected_point_position
                    radius = self.max_growth_ratio * \
                        utils.spiral_radius_at_length(
                            selected_point_weight, self.winding_factor)
                    if 'spiramir' in parent_curve:
                        direction = utils.invert_direction(
                            parent_curve['spiramir_direction'])
                    single = True
                bpy.ops.object.mode_set(mode='OBJECT')
            elif (parent_curve.type != 'CURVE' or not parent_curve.select_get()):
                parent_curve = None

        if single:
            spiral = utils.Spiral(radius=radius, direction=direction, winding_factor=self.winding_factor,
                                  curvature_error=self.curvature_error, starting_angle=self.starting_angle)
            spiral.add_to_scene(context, parent_curve=parent_curve,
                                position=position, tube_radius=self.tube_radius)
        else:
            self.add_recursive_spirals(context)

        self.log("Drawing took %.4f sec." % (time.time() - time_start))

        context.view_layer.update()

        return {'FINISHED'}
