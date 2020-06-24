import time
from mathutils import Vector, Matrix
from math import cos, sin, sqrt, log, exp, atan, pi, acos

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


def cartesian(t, r, direction):
    sign = 1 if direction == 'CLOCKWISE' else -1
    return (r * cos(t), sign * r * sin(t))


def spiral_polar(t, b):
    return exp(b * t)


def spiral_cartesian(t, b, direction):
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


def make_spiral(radius, b, curvature_error, starting_angle, direction, rotation_angle):
    verts = []

    t = log(radius) / b
    length = spiral_length_at_angle(t, b)
    rot = atan(1 / b)
    diameter = 0
    mass_center = None

    end = spiral_cartesian(t, b, direction)

    angle = starting_angle

    while True:
        l = spiral_length_at_angle(angle, b)
        if l >= length:
            break
        p = spiral_cartesian(angle, b, direction)
        p = translate(p, (-end[0], -end[1]))
        if direction == 'CLOCKWISE':
            p = rotate(p, - rotation_angle - t - rot - pi)
        else:
            p = rotate(p, rotation_angle + t + rot + pi)
        verts.append([p[0], p[1], 0, l])
        d = distance((0, 0), p)
        if d > diameter:
            diameter = d
            mass_center = between((0, 0), p)
        angle += angle_increment(angle, b, curvature_error)

    verts.append([0, 0, 0, length])

    return verts, diameter, mass_center


def make_vector(angle, radius, direction):
    v = cartesian(angle, radius, direction)
    return [[0, 0, 0, 0], [v[0], v[1], 0, 0]]


def get_align_matrix(context, location):
    loc = Matrix.Translation(location)
    obj_align = context.preferences.edit.object_align
    if (context.space_data.type == 'VIEW_3D' and obj_align == 'VIEW'):
        rot = context.space_data.region_3d.view_matrix.to_3x3().inverted().to_4x4()
    else:
        rot = Matrix()
    align_matrix = loc @ rot
    return align_matrix


def verts_to_points(verts):
    vert_array = []
    length_array = []
    for v in verts:
        vert_array.extend(v[0:3])
        vert_array.append(0)
        length_array.append(v[3])
    return vert_array, length_array


def get_selected_vertices(curve):
    points = []
    for spline in curve.data.splines:
        for point in spline.points:
            if point.select:
                points.append(point)
    return points[-1]


def invert_direction(direction):
    if direction == 'CLOCKWISE':
        return 'COUNTER_CLOCKWISE'
    else:
        return 'CLOCKWISE'


def add_spline_to_curve(curve, spline_type, verts, location, rotation):
    vert_array, length_array = verts_to_points(verts)

    # Deselect all points.
    for spline in curve.data.splines:
        for point in spline.points:
            point.select = False

    # Create new spline from vert_array.
    new_spline = curve.data.splines.new(type=spline_type)
    new_spline.points.add(int(len(vert_array) / 4 - 1))
    new_spline.points.foreach_set('co', vert_array)
    new_spline.points.foreach_set('weight', length_array)
    new_spline.points.foreach_set('radius', [l**(1./3) for l in length_array])
    new_spline.use_endpoint_u = False

    # Select all points.
    for point in new_spline.points:
        point.select = True

    # Translate and rotate.
    bpy.ops.transform.translate(value=location)
    bpy.ops.transform.rotate(value=rotation[0], orient_axis='X')
    bpy.ops.transform.rotate(value=rotation[1], orient_axis='Y')
    bpy.ops.transform.rotate(value=rotation[2], orient_axis='Z')


def draw_spiral(context, radius, b, direction, curvature_error, starting_angle, location, rotation):
    align_matrix = get_align_matrix(context, location)
    spline_type = 'POLY'
    length_contraction = 0.8

    if bpy.context.mode == 'EDIT_CURVE':
        curve = context.active_object
        selected_vertex = get_selected_vertices(curve)
        if selected_vertex:
            origin = selected_vertex.co[0:3]
            length = selected_vertex.weight
            radius = spiral_radius_at_length(length, b)

            # This is weird but turns out the rotation of the spiral
            # cancels out the changes we have to make to obtain the 
            # tangent angle and the actual angle at length is the tangential
            # angle in the frame of reference of the spiral with the origin
            # at the end.
            tangent_angle = spiral_angle_at_length(length, b)

            # tangent_vector = make_vector(
            #     tangent_angle, 0.3 * radius, direction)
            # add_spline_to_curve(curve, spline_type, tangent_vector, origin, rotation)

            direction = invert_direction(curve['spiramir_direction'])
            verts, _, _ = make_spiral(length_contraction * radius, b, curvature_error,
                                 starting_angle, direction, tangent_angle)
            add_spline_to_curve(curve, spline_type, verts, origin, rotation)

    else:
        data_curve = bpy.data.curves.new(name='Spiral', type='CURVE')
        curve = object_data_add(context, data_curve)  # Place in active scene
        curve.matrix_world = align_matrix  # Apply matrix
        curve.rotation_euler = rotation
        curve.select_set(True)

        curve.data.dimensions = '2D'

        curve['spiramir_radius'] = radius
        curve['spiramir_b'] = b
        curve['spiramir_direction'] = direction
        curve['spiramir_curvature_error'] = curvature_error
        curve['spiramir_starting_angle'] = starting_angle

        verts, _, _ = make_spiral(radius, b, curvature_error, starting_angle, direction, 0)
        add_spline_to_curve(curve, spline_type, verts, location, rotation)


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
        col.prop(self, "radius", text="Radius")
        col.prop(self, "winding_factor", text="Winding Factor")
        col.prop(self, "starting_angle", text="Starting Angle")
        col.prop(self, "curvature_error", text="Curvature Error")

        col = layout.column()
        col.row().prop(self, "edit_mode", expand=True)

        box = layout.box()
        box.label(text="Location:")
        box.prop(self, "location")

        box = layout.box()
        box.label(text="Rotation:")
        box.prop(self, "rotation")

    def execute(self, context):
        time_start = time.time()

        draw_spiral(context, radius=self.radius, b=self.winding_factor, direction=self.direction,
                    curvature_error=self.curvature_error, starting_angle=self.starting_angle,
                    location=self.location, rotation=self.rotation)

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
