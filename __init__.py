from bpy.types import (
    Operator,
    Menu,
)
import time
from mathutils import Vector, Matrix
from math import cos, sin, pi
from bpy_extras.object_utils import object_data_add
from bpy.props import (
    EnumProperty,
    BoolProperty,
    FloatProperty,
    IntProperty,
    FloatVectorProperty
)
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


def get_align_matrix(context, location):
    loc = Matrix.Translation(location)
    obj_align = context.preferences.edit.object_align
    if (context.space_data.type == 'VIEW_3D' and obj_align == 'VIEW'):
        rot = context.space_data.region_3d.view_matrix.to_3x3().inverted().to_4x4()
    else:
        rot = Matrix()
    align_matrix = loc @ rot
    return align_matrix


def vertsToPoints(verts):
    vertArray = []
    for v in verts:
        vertArray += v
        vertArray.append(0)
    return vertArray


def make_spiral(radius, b, turns, direction, steps):
    max_phi = 2 * pi * turns
    # angle in radians between two vertices
    step_phi = max_phi / (steps * turns)

    if direction == 'CLOCKWISE':
        step_phi *= -1  # flip direction
        max_phi *= -1

    verts = []
    verts.append([radius, 0, 0])

    cur_phi = 0
    while abs(cur_phi) <= abs(max_phi):
        cur_phi += step_phi
        cur_rad = radius * pow(b, abs(cur_phi))
        px = cur_rad * cos(cur_phi)
        py = cur_rad * sin(cur_phi)
        verts.append([px, py, 0])

    return verts


def draw_spiral(context, radius=0.1, b=1.3, turns=3, direction='CLOCKWISE', steps=63, location=[0, 0, 0], rotation=[0, 0, 0]):
    verts = make_spiral(radius, b, turns, direction, steps)
    align_matrix = get_align_matrix(context, location)
    splineType = 'POLY'

    if bpy.context.mode == 'EDIT_CURVE':
        Curve = context.active_object
        newSpline = Curve.data.splines.new(type=splineType)          # spline
    else:
        # create curve
        dataCurve = bpy.data.curves.new(
            name='Spiral', type='CURVE')  # curvedatablock
        newSpline = dataCurve.splines.new(type=splineType)          # spline

        # create object with newCurve
        Curve = object_data_add(context, dataCurve)  # place in active scene
        Curve.matrix_world = align_matrix  # apply matrix
        Curve.rotation_euler = rotation
        Curve.select_set(True)

    # set curveOptions
    Curve.data.dimensions = '2D'
    Curve.data.use_path = True
    Curve.data.fill_mode = 'BOTH'

    # turn verts into array
    vertArray = vertsToPoints(verts)

    for spline in Curve.data.splines:
        for point in spline.points:
            point.select = False

    # create newSpline from vertarray
    newSpline.points.add(int(len(vertArray) / 4 - 1))
    newSpline.points.foreach_set('co', vertArray)
    newSpline.use_endpoint_u = False
    for point in newSpline.points:
        point.select = True

    # move and rotate spline in edit mode
    if bpy.context.mode == 'EDIT_CURVE':
        bpy.ops.transform.translate(value=location)
        bpy.ops.transform.rotate(value=rotation[0], orient_axis='X')
        bpy.ops.transform.rotate(value=rotation[1], orient_axis='Y')
        bpy.ops.transform.rotate(value=rotation[2], orient_axis='Z')


class CURVE_OT_spiramir(Operator):
    bl_idname = "curve.spiramir"
    bl_label = "Spiramir"
    bl_description = "Create a spiramir"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_options = {'REGISTER', 'UNDO'}

    direction: EnumProperty(
        items=[('COUNTER_CLOCKWISE', "Counter Clockwise",
                "Wind in a counter clockwise direction"),
               ("CLOCKWISE", "Clockwise",
                "Wind in a clockwise direction")],
        default='COUNTER_CLOCKWISE',
        name="Direction",
        description="Direction of winding"
    )
    turns: IntProperty(
        default=1,
        min=1, max=1000,
        description="Length of Spiral in 360 deg"
    )
    steps: IntProperty(
        default=64,
        min=2, max=1000,
        description="Number of Vertices per turn"
    )
    radius: FloatProperty(
        default=1.00,
        min=0.00, max=100.00,
        description="Radius for first turn"
    )
    b: FloatProperty(
        default=1.2,
        min=0.00, max=30.00,
        description="Factor of exponent"
    )
    edit_mode: BoolProperty(
        name="Show in edit mode",
        default=False,
        description="Show in edit mode"
    )
    location: FloatVectorProperty(
        name="",
        description="Start location",
        default=(0.0, 0.0, 0.0),
        subtype='TRANSLATION'
    )
    rotation: FloatVectorProperty(
        name="",
        description="Rotation",
        default=(0.0, 0.0, 0.0),
        subtype='EULER'
    )

    def draw(self, context):
        layout = self.layout
        col = layout.column_flow(align=True)

        col.prop(self, "direction")

        col = layout.column(align=True)
        col.label(text="Spiral Parameters:")
        col.prop(self, "turns", text="Turns")
        col.prop(self, "steps", text="Steps")
        col.prop(self, "radius", text="Radius")
        col.prop(self, "b", text="Expansion Rate")

        col = layout.column()
        col.row().prop(self, "edit_mode", expand=True)

        box = layout.box()
        box.label(text="Location:")
        box.prop(self, "location")

        box = layout.box()
        box.label(text="Rotation:")
        box.prop(self, "rotation")

    def execute(self, context):
        # turn off 'Enter Edit Mode'
        use_enter_edit_mode = bpy.context.preferences.edit.use_enter_edit_mode
        bpy.context.preferences.edit.use_enter_edit_mode = False

        time_start = time.time()

        draw_spiral(context, radius=self.radius, b=self.b, turns=self.turns, direction=self.direction,
                    steps=self.steps, location=self.location, rotation=self.rotation)

        if use_enter_edit_mode:
            bpy.ops.object.mode_set(mode='EDIT')

        # restore pre operator state
        bpy.context.preferences.edit.use_enter_edit_mode = use_enter_edit_mode

        if self.edit_mode:
            bpy.ops.object.mode_set(mode='EDIT')
        else:
            bpy.ops.object.mode_set(mode='OBJECT')

        self.report({'INFO'}, "Drawing Spiral Finished: %.4f sec" %
                    (time.time() - time_start))

        return {'FINISHED'}


def menu_func(self, context):
    self.layout.operator(CURVE_OT_spiramir.bl_idname)


def register():
    bpy.utils.register_class(CURVE_OT_spiramir)
    bpy.types.VIEW3D_MT_add.append(menu_func)


def unregister():
    bpy.utils.unregister_class(CURVE_OT_spiramir)
    bpy.types.VIEW3D_MT_add.remove(menu_func)
