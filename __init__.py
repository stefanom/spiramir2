import time
from mathutils import Vector, Matrix
from math import cos, sin, sqrt, log, exp, atan, pi

import bpy
from bpy_extras.object_utils import object_data_add
from bpy.props import (
    EnumProperty,
    BoolProperty,
    FloatProperty,
    IntProperty,
    FloatVectorProperty
)

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


def translate(p, t):
    return (p[0] + t[0], p[1] + t[1])

def rotate(p, angle):
    cosA = cos(angle)
    sinA = sin(angle)
    return (p[0] * cosA - p[1] * sinA, p[0] * sinA + p[1] * cosA)

def spiral_polar(t, b):
    return exp(b * t)

def spiral_cartesian(t, b, direction):
    r = spiral_polar(b, t)
    sign = 1 if direction == 'CLOCKWISE' else -1
    return (r * cos(t), sign * r * sin(t))

def spiral_length_at_angle(t, b):
    return sqrt(1 + b*b) * exp(b * t) / b

def spiral_angle_at_length(l, b):
    return log(l * b / sqrt(1 + b*b)) / b


def make_spiral(radius, b, segments, fraction, direction):
    verts = []

    t = log(radius) / b
    length = spiral_length_at_angle(t, b)
    rot = atan(1 / b)
    origin = spiral_cartesian(t, b, direction)

    step = (fraction * length) / segments

    for i in range(segments):
        l = length - i * step
        angle = spiral_angle_at_length(l, b)
        p = spiral_cartesian(angle, b, direction)
        p = translate(p, (-origin[0], -origin[1]))
        if direction == 'CLOCKWISE':
            p = rotate(p, - t - rot - pi)
        else:
            p = rotate(p, t + rot + pi)
        verts.append([p[0], p[1], 0])

    return verts


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


def draw_spiral(context, radius, b, direction, segments, fraction, location, rotation):
    verts = make_spiral(radius, b, segments, fraction, direction)
    align_matrix = get_align_matrix(context, location)
    splineType = 'POLY'

    if bpy.context.mode == 'EDIT_CURVE':
        Curve = context.active_object
        newSpline = Curve.data.splines.new(type=splineType)
    else:
        dataCurve = bpy.data.curves.new(name='Spiral', type='CURVE')
        newSpline = dataCurve.splines.new(type=splineType)

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


class CURVE_OT_spiramir(bpy.types.Operator):
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
    segments: IntProperty(
        default=128,
        min=1, max=1024,
        description="Total Number of Vertices"
    )
    radius: FloatProperty(
        default=1.0,
        min=0.00001, max=1000.00,
        description="Radius of the spiral"
    )
    winding_factor: FloatProperty(
        default=100,
        min=1, max=10000,
        description="Spiral Winding Factor"
    )
    fraction: FloatProperty(
        default=0.92,
        min=0.001, max=1.0,
        description="Fraction of the spiral drawn"
    )
    edit_mode: BoolProperty(
        name="Show in edit mode",
        default=True,
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
        col.prop(self, "segments", text="Segments")
        col.prop(self, "radius", text="Radius")
        col.prop(self, "winding_factor", text="Winding Factor")
        col.prop(self, "fraction", text="Drawn Fraction")

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

        b = 1 / (1 + log(self.winding_factor))

        draw_spiral(context, radius=self.radius, b=b, direction=self.direction,
                    segments=self.segments, fraction=self.fraction, location=self.location,
                    rotation=self.rotation)

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
