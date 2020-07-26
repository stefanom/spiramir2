import bpy

bl_info = {
    "name": "Spiramir2",
    "author": "Stefano Mazzocchi",
    "description": "",
    "blender": (2, 80, 0),
    "version": (0, 0, 1),
    "location": "View3D > Add > Curve",
    "warning": "",
    "wiki_url": "",
    "category": "Add Curve"
}


from . import circles
from . import sprues
from . import spiramir
from . import utils

classes = [
    spiramir.CURVE_PT_spiramir,
    spiramir.CURVE_OT_spiramir,
    sprues.CURVE_OT_spiramir_sprues,
    circles.CURVE_OT_spiramir_circles,
]

addon_keymaps = []


def menu_func(self, context):
    self.layout.operator(spiramir.CURVE_OT_spiramir.bl_idname)
    self.layout.operator(sprues.CURVE_OT_spiramir_sprues.bl_idname)
    self.layout.operator(circles.CURVE_OT_spiramir_circles.bl_idname)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.VIEW3D_MT_curve_add.append(menu_func)

    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    km = kc.keymaps.new(name="3D View Generic",
                        space_type='VIEW_3D', region_type='WINDOW')
    kmi1 = km.keymap_items.new(
        spiramir.CURVE_OT_spiramir.bl_idname, 'S', 'PRESS', ctrl=True, shift=True)
    kmi2 = km.keymap_items.new(
        sprues.CURVE_OT_spiramir_sprues.bl_idname, 'R', 'PRESS', ctrl=True, shift=True)
    kmi3 = km.keymap_items.new(
        circles.CURVE_OT_spiramir_circles.bl_idname, 'C', 'PRESS', ctrl=True, shift=True)

    addon_keymaps.append((km, kmi1))
    addon_keymaps.append((km, kmi2))
    addon_keymaps.append((km, kmi3))


def unregister():
    for km, kmi in addon_keymaps:
        km.keymap_items.remove(kmi)
    addon_keymaps.clear()

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

    bpy.types.VIEW3D_MT_curve_add.remove(menu_func)


# =========================== attic ====================================

            # if len(curve.data.splines) == 1:
            #     spline = curve.data.splines[0]
            #     if spline.type == 'BEZIER':
            #         origin, tangent, _ = bezier_multi_seg(spline.bezier_points, self.position, closed_loop=True, matrix=curve.matrix_world)
            #     elif spline.type == 'POLY':
            #         points = len(spline.points)
            #         i = round(self.position * points)
            #         origin = curve.matrix_world @ Vector(spline.points[i].co[0:3])
            #         tangent = curve.matrix_world @ Vector(spline.points[i + 1 % points].co[0:3]) - origin

        
        #empty.rotation_mode = 'QUATERNION'
        #empty.rotation_quaternion = tangent.to_track_quat('X', 'Z')

