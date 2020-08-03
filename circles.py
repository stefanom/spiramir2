import random
import time

import bpy

from . import utils


class CURVE_OT_spiramir_circles(bpy.types.Operator):
    bl_idname = "curve.spiramir_circles"
    bl_label = "Circles"
    bl_description = "Enclose with circles"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_options = {'REGISTER', 'UNDO'}

    iterations: bpy.props.IntProperty(
        default=1,
        min=1, max=500,
        description="Number of circles to inscribe"
    )
    max_attempts: bpy.props.IntProperty(
        default=100,
        min=1, max=10000,
        description="Maximum number of attempts before giving up"
    )
    min_radius: bpy.props.FloatProperty(
        default=0.0,
        min=0.0, max=1000.0,
        description="Smallest radius allowed for embedded circles"
    )
    max_radius: bpy.props.FloatProperty(
        default=10,
        min=0.0, max=1000.0,
        description="Biggest radius allowed for embedded circles"
    )
    bevel_depth: bpy.props.FloatProperty(
        default=0.1,
        min=0.0, max=1000.0,
        description="Bevel depth for the embedded circles"
    )

    def draw(self, context):
        col = self.layout.column(align=True)
        col.prop(self, "iterations", text="Iterations")
        col.prop(self, "max_attempts", text="Max Attempts")
        col.prop(self, "min_radius", text="Min Radius")
        col.prop(self, "max_radius", text="Max Radius")
        col.prop(self, "bevel_depth", text="Bevel Depth")

    def execute(self, context):
        parent_curve = context.object
        if parent_curve and bpy.context.mode == 'EDIT_CURVE':
            position, _ = utils.get_selected_point(parent_curve)
            growth_point, _ = utils.get_constrainted_empty(
                parent_curve, position, name="Growth Point")
            grow_left = 'spiramir' not in parent_curve or (
                'spiramir' in parent_curve and parent_curve['spiramir_direction'] == utils.CLOCKWISE)
            r, cp = utils.get_available_radius(
                growth_point, grow_left=grow_left)
            utils.add_circle(r, parent=growth_point,
                             contact_point=cp, bevel_depth=self.bevel_depth, name="Circle")
        else:
            drawn = 0
            for _ in range(self.max_attempts):
                curve = utils.get_random_curve()
                if not curve:
                    raise ValueError('No curve to grow from.')
                position = random.uniform(0.0, 1.0)

                growth_point, _ = utils.get_constrainted_empty(
                    curve, position, name="Growth Point")
                r, cp = utils.get_available_radius(growth_point)
                if r > self.min_radius:
                    r = min(r, self.max_radius)
                    utils.add_circle(
                        r, parent=growth_point, contact_point=cp, bevel_depth=self.bevel_depth, name="Circle")
                    drawn += 1
                    if drawn == self.iterations:
                        break
                else:
                    utils.remove(growth_point)

        return {'FINISHED'}
