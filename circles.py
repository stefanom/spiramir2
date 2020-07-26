import random, time

import bpy

from . import utils

class CURVE_OT_spiramir_circles(bpy.types.Operator):
    bl_idname = "curve.spiramir_circles"
    bl_label = "Spiramir Circles"
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
        default=0.1,
        min=0.0, max=1000.0,
        description="Smallest radius allowed for embedded circles"
    )

    max_radius: bpy.props.FloatProperty(
        default=0.7,
        min=0.0, max=1000.0,
        description="Biggest radius allowed for embedded circles"
    )

    bevel_depth: bpy.props.FloatProperty(
        default=0.02,
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

    def inscribe_circle(self):
        curve = utils.get_random_curve()
        spline = utils.get_random_spline(curve)
        step = random.uniform(0.0, 1.0)

        if not spline.bezier_points:
            return False

        p, t, n = utils.bezier_multi_seg(
            spline.bezier_points, step, closed_loop=True, matrix=curve.matrix_world)
        r, cp = utils.get_available_radius(curve, p, t, n)

        if r > self.min_radius:
            r = min(r, self.max_radius)

            bpy.ops.object.empty_add(type='ARROWS', location=p)
            empty = bpy.context.object
            empty.rotation_mode = 'QUATERNION'
            empty.rotation_quaternion = t.to_track_quat('X', 'Z')

            bpy.ops.curve.primitive_bezier_circle_add(
                radius=r, location=(0, r, 0))
            circle = bpy.context.object
            circle.parent = empty
            circle.data.bevel_depth = self.bevel_depth

            bpy.ops.object.empty_add(type='SINGLE_ARROW', location=cp)
            return True
        else:
            return False

    def execute(self, context):
        drawn = 0
        for _ in range(self.max_attempts):
            if self.inscribe_circle():
                print('[*] Successful inscription.')
                drawn += 1
                if drawn == self.iterations:
                    break
            else:
                print('[ ] Failed inscription.')

        return {'FINISHED'}
