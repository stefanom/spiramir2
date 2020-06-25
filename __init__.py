from collections import deque
from itertools import chain, repeat
import time
from mathutils import Vector, Matrix
from math import cos, sin, sqrt, log, exp, atan, atan2, pi, acos

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
    if len(p) == 3 and len(t) == 3:
        return (p[0] + t[0], p[1] + t[1], p[2] + t[2])
    else:
        return (p[0] + t[0], p[1] + t[1])


def rotate(p, angle):
    cosA = cos(angle)
    sinA = sin(angle)
    return (p[0] * cosA - p[1] * sinA, p[0] * sinA + p[1] * cosA)


def direction_sign(direction):
    return 1 if direction == 'CLOCKWISE' else -1


def direction_from_sign(value):
    return 'CLOCKWISE' if value >= 0.0 else 'COUNTER_CLOCKWISE'


def invert_direction(direction):
    if direction == 'CLOCKWISE':
        return 'COUNTER_CLOCKWISE'
    else:
        return 'CLOCKWISE'


def cartesian(t, r, direction='CLOCKWISE'):
    sign = direction_sign(direction)
    return (r * cos(t), sign * r * sin(t))


def spiral_polar(t, b):
    return exp(b * t)


def spiral_cartesian(t, b, direction='CLOCKWISE'):
    return cartesian(t, spiral_polar(b, t), direction)


def spiral_length_at_angle(t, b):
    return sqrt(1 + b*b) * exp(b * t) / b


def spiral_angle_at_length(l, b):
    return log(l * b / sqrt(1 + b*b)) / b


def spiral_radius_at_length(l, b):
    return l * b / sqrt(1 + b*b)


def obsculating_circle_radius_at_angle(t, b):
    return b * spiral_length_at_angle(t, b)


def angle_increment(t, b, curvature_error):
    r = obsculating_circle_radius_at_angle(t, b)
    return acos(r / (r + curvature_error))


def distance(a, b):
    return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def between(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def angle_between_points(a, b):
    return atan2(a[1] - b[1], a[0] - b[0])


def make_spiral(radius, b, curvature_error, starting_angle, direction, rotation):
    verts = []

    t = log(radius) / b
    length = spiral_length_at_angle(t, b)
    rot = atan(1 / b)
    diameter = 0
    mass_center = None
    sign = direction_sign(direction)
    end = spiral_cartesian(t, b, direction)

    angle = starting_angle

    while True:
        l = spiral_length_at_angle(angle, b)
        if l >= length:
            break
        p = spiral_cartesian(angle, b, direction)
        p = translate(p, (-end[0], -end[1]))
        p = rotate(p, rotation - sign * (t + rot + pi))
        verts.append((p[0], p[1], 0, sign * l))
        d = distance((0, 0), p)
        if d > diameter:
            diameter = d
            mass_center = between((0, 0), p)
        angle += angle_increment(angle, b, curvature_error)

    verts.append((0, 0, 0, sign * length))

    return verts, diameter / 2, (mass_center[0], mass_center[1], 0)


def make_vector(angle, radius):
    v = cartesian(angle, radius)
    return [(0, 0, 0, 0), (v[0], v[1], 0, 0)]


def make_circle(radius, segments):
    verts = []
    angle = 2 * pi / segments
    for i in range(segments):
        p = cartesian(angle * i, radius)
        verts.append((p[0], p[1], 0, 0))
    verts.append((radius, 0, 0, 0))
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


def verts_to_points(verts, location):
    vert_array = []
    length_array = []
    for v in verts:
        vert_array.extend(translate(v[0:3], location))
        vert_array.append(0)
        length_array.append(v[3])
    return vert_array, length_array


def windowed(seq, n, fillvalue=None, step=1):
    window = deque(maxlen=n)
    i = n
    for _ in map(window.append, seq):
        i -= 1
        if not i:
            i = step
            yield tuple(window)

    size = len(window)
    if size < n:
        yield tuple(chain(window, repeat(fillvalue, n - size)))
    elif 0 < i < min(step, n):
        window += (fillvalue,) * i
        yield tuple(window)


def get_selected_vertex(curve):
    selected_points = []
    for spline in curve.data.splines:
        for points in windowed(spline.points, 2):
            if points[1].select and points[0]:
                selected_points.append(points)
    return selected_points[-1]


def draw_spiral(props, context):
    origin = [0, 0, 0]
    length_contraction = 0.8
    tangent_angle = 0.0
    b = props.winding_factor
    radius = props.radius
    direction = props.direction

    if bpy.context.mode == 'EDIT_CURVE':
        curve = context.active_object
        selected_vertex, previous_vertex = get_selected_vertex(curve)
        if selected_vertex:
            origin = selected_vertex.co[0:3]
            length = abs(selected_vertex.weight)
            direction = invert_direction(direction_from_sign(selected_vertex.weight))
            radius = length_contraction * spiral_radius_at_length(length, b)
            tangent_angle = angle_between_points(previous_vertex.co, selected_vertex.co) + pi
    else:
        data_curve = bpy.data.curves.new(name='Spiral', type='CURVE')

        curve = object_data_add(context, data_curve)  # Place in active scene
        curve.matrix_world = get_align_matrix(context, origin)
        curve.select_set(True)

        curve.data.dimensions = '2D'
        curve.data.use_path = True
        curve.data.fill_mode = 'BOTH'

        curve['spiramir_b'] = b
        curve['spiramir_curvature_error'] = props.curvature_error
        curve['spiramir_starting_angle'] = props.starting_angle

    spiral, spiral_radius, spiral_mass_center = make_spiral(radius, b, 
        props.curvature_error, props.starting_angle, direction, tangent_angle)

    if props.draw_circle:
        circle = make_circle(spiral_radius, 64)
        add_spline_to_curve(
            curve, circle, translate(origin, spiral_mass_center))

    if props.draw_tangent:
        tangent_vector = make_vector(tangent_angle, 0.4 * radius)
        add_spline_to_curve(curve, tangent_vector, origin)

    add_spline_to_curve(curve, spiral, origin)


def add_spline_to_curve(curve, verts, location):
    vert_array, length_array = verts_to_points(verts, location)

    new_spline = curve.data.splines.new(type='POLY')
    new_spline.points.add(int(len(vert_array) / 4 - 1))
    new_spline.points.foreach_set('co', vert_array)
    new_spline.points.foreach_set('weight', length_array)
    new_spline.points.foreach_set(
        'radius', [abs(l)**(1./3) for l in length_array])


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
    radius: FloatProperty(
        default=1.0,
        min=0.00001, max=1000.0,
        description="Radius of the spiral"
    )
    winding_factor: FloatProperty(
        default=0.2,
        min=0.0001, max=2,
        description="Spiral Winding Factor"
    )
    starting_angle: FloatProperty(
        default=-20.0,
        min=-100.0, max=100.0,
        description="Angle to start drawing the spiral"
    )
    curvature_error: FloatProperty(
        default=0.001,
        min=0.0000001, max=1000.0,
        description="Maximum curvature error"
    )
    draw_circle: BoolProperty(
        name="Draw circle",
        default=False,
        description="Draw the occupancy circle"
    )
    draw_tangent: BoolProperty(
        name="Draw tangent",
        default=False,
        description="Draw the tangent vector"
    )
    edit_mode: BoolProperty(
        name="Show in edit mode",
        default=True,
        description="Show in edit mode"
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

        col = layout.column()
        col.row().prop(self, "draw_circle", expand=True)
        col.row().prop(self, "draw_tangent", expand=True)
        col.row().prop(self, "edit_mode", expand=True)

    def execute(self, context):
        time_start = time.time()

        draw_spiral(self, context)

        if self.edit_mode:
            bpy.ops.object.mode_set(mode='EDIT')
        else:
            bpy.ops.object.mode_set(mode='OBJECT')

        self.report({'INFO'}, "Drawing Spiral Finished: %.4f sec" % (time.time() - time_start))

        return {'FINISHED'}


def menu_func(self, context):
    self.layout.operator(CURVE_OT_spiramir.bl_idname)


addon_keymaps = []

def register():
    bpy.utils.register_class(CURVE_OT_spiramir)
    bpy.types.VIEW3D_MT_curve_add.append(menu_func)

    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    km = kc.keymaps.new(name="3D View Generic",
                        space_type='VIEW_3D', region_type='WINDOW')
    kmi = km.keymap_items.new(
        CURVE_OT_spiramir.bl_idname, 'S', 'PRESS', ctrl=True, shift=True)

    addon_keymaps.append((km, kmi))


def unregister():
    for km, kmi in addon_keymaps:
        km.keymap_items.remove(kmi)
    addon_keymaps.clear()

    bpy.utils.unregister_class(CURVE_OT_spiramir)
    bpy.types.VIEW3D_MT_curve_add.remove(menu_func)
