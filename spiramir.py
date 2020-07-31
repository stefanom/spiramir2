import random
import time

import bpy

from . import utils


class CURVE_OT_spiramir(bpy.types.Operator):
    bl_idname = "curve.spiramir"
    bl_label = "Spiramir"
    bl_options = {'REGISTER', 'UNDO'}

    verbose = False

    direction: bpy.props.EnumProperty(
        items=[(utils.COUNTER_CLOCKWISE, "Counter Clockwise",
                "Wind in a counter clockwise direction"),
               (utils.CLOCKWISE, "Clockwise",
                "Wind in a clockwise direction")],
        default=utils.COUNTER_CLOCKWISE,
        name="Direction",
        description="Direction of winding"
    )
    radius: bpy.props.FloatProperty(
        default=10.0,
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
        default=0.01,
        min=0.0000001, max=1000.0,
        description="Maximum curvature error"
    )

    tube_radius: bpy.props.FloatProperty(
        default=0.05,
        min=0.0, max=1000.0,
        description="Radius of the tube around the curve"
    )
    position: bpy.props.FloatProperty(
        default=0.0,
        min=0.0, max=100.0,
        description="Relative position of the growth point"
    )
    draw_supports: bpy.props.BoolProperty(
        default=False,
        description="Whether to draw the supports or not"
    )

    iterations: bpy.props.IntProperty(
        default=1,
        min=1, max=500,
        description="Number of recursive iterations"
    )
    max_attempts: bpy.props.IntProperty(
        default=100,
        min=1, max=10000,
        description="Maximum number of placement attempts during recursion"
    )
    min_growth_ratio: bpy.props.FloatProperty(
        default=0.4,
        min=0.0, max=1.0,
        description="Minimum growth ratio"
    )
    max_growth_ratio: bpy.props.FloatProperty(
        default=0.95,
        min=0.0, max=1.0,
        description="Maximum growth ratio"
    )
    min_radius: bpy.props.FloatProperty(
        default=0.5,
        min=0.0001, max=100.0,
        description="The minimum radius for recursively grown spirals"
    )
    offset: bpy.props.FloatProperty(
        default=0.1,
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
        col.prop(self, "draw_supports", text="Draw Supports")

        col = layout.column(align=True)
        col.label(text="Recursion Parameters:")
        col.prop(self, "iterations", text="Recursion Iterations")
        col.prop(self, "max_attempts", text="Max Attempts")
        col.prop(self, "offset", text="Growth Offset")
        col.prop(self, "min_growth_ratio", text="Min Growth Ratio")
        col.prop(self, "max_growth_ratio", text="Max Growth Ratio")
        col.prop(self, "min_radius", text="Minimum Spiral Radius")

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
            growth_point = None
            radius = self.radius
            direction = self.direction

            parent_curve, position = self.get_random_growth_point()

            if parent_curve:
                growth_point, _ = utils.get_constrainted_empty(
                    parent_curve, position)
                grow_left = 'spiramir' not in parent_curve or (
                    'spiramir' in parent_curve and parent_curve['spiramir_direction'] == utils.CLOCKWISE)
                available_radius, contact_point = utils.get_available_radius(
                    growth_point, grow_left=grow_left)
                if self.draw_supports and contact_point:
                    utils.add_circle(
                        available_radius, parent=growth_point, contact_point=contact_point)

                available_radius = abs(available_radius)

                if 'spiramir' in parent_curve:
                    direction = utils.invert_direction(
                        parent_curve['spiramir_direction'])
                    weight = utils.get_weight_at_position(
                        parent_curve, position)
                    available_radius = min(utils.spiral_radius_at_length(
                        weight, self.winding_factor), available_radius)

                radius = random.uniform(
                    self.min_growth_ratio * available_radius, self.max_growth_ratio * available_radius)

            if radius >= self.min_radius:
                spiral = utils.Spiral(radius=radius, direction=direction, winding_factor=self.winding_factor,
                                      curvature_error=self.curvature_error, starting_angle=self.starting_angle)

                spiral.add_to_scene(
                    context, parent=growth_point, tube_radius=self.tube_radius, draw_circle=self.draw_supports)
                drawn_spirals += 1
                if drawn_spirals >= self.iterations:
                    break
            else:
                utils.remove(growth_point)

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
                position, weight = utils.get_selected_point(parent_curve)
                if position:
                    growth_point, _ = utils.get_constrainted_empty(
                        parent_curve, position)
                    grow_left = 'spiramir' not in parent_curve or (
                        'spiramir' in parent_curve and parent_curve['spiramir_direction'] == utils.CLOCKWISE)
                    available_radius, contact_point = utils.get_available_radius(
                        growth_point, grow_left=grow_left)
                    if self.draw_supports and contact_point:
                        utils.add_circle(
                            available_radius, parent=growth_point, contact_point=contact_point)
                    radius = self.max_growth_ratio * \
                        min(utils.spiral_radius_at_length(
                            weight, self.winding_factor), abs(available_radius))
                    if 'spiramir' in parent_curve:
                        direction = utils.invert_direction(
                            parent_curve['spiramir_direction'])
                    single = True
            elif (parent_curve.type != 'CURVE' or not parent_curve.select_get()):
                parent_curve = None

        if single:
            spiral = utils.Spiral(radius=radius, direction=direction, winding_factor=self.winding_factor,
                                  curvature_error=self.curvature_error, starting_angle=self.starting_angle)
            spiral.add_to_scene(context, parent=growth_point,
                                tube_radius=self.tube_radius, draw_circle=self.draw_supports)
        else:
            self.add_recursive_spirals(context)

        self.log("Drawing took %.4f sec." % (time.time() - time_start))

        context.view_layer.update()

        return {'FINISHED'}
