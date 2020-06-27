from collections import deque
from itertools import chain, repeat
import time, random
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


def to_4d(v):
    if type(v) in [list, tuple]:
        if len(v) == 1:
            return (v[0], 0, 0, 0)
        if len(v) == 2:
            return (v[0], v[1], 0, 0)
        if len(v) == 3:
            return (v[0], v[1], v[2], 0)
        return v
    else:
        return (v, 0, 0, 0)


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
    mass_center = (0,0)
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
    return [(0, 0, 0, 0), to_4d(v)]


def make_centripetal_vector(center):
    return [(0, 0, 0, 0), to_4d(center)]


def make_circle(radius, segments):
    verts = []
    angle = 2 * pi / segments
    for i in range(segments):
        p = cartesian(angle * i, radius)
        verts.append(to_4d(p))
    verts.append(to_4d(radius))
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


def get_mass_center(points):
    mass_center = (0, 0, 0)
    for point in points:
        mass_center = translate(mass_center, point.co[0:3])
    return (mass_center[0] / len(points), mass_center[1] / len(points), 0)


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


def verts_to_points(verts, location):
    vert_array = []
    length_array = []
    for v in verts:
        vert_array.extend(translate(v[0:3], location))
        vert_array.append(0)
        length_array.append(v[3])
    return vert_array, length_array


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
    contraction: FloatProperty(
        default=0.8,
        min=0.0001, max=1.0,
        description="Radius reduction for recursive growth"
    )
    iterations: IntProperty(
        default=64,
        min=1, max=500,
        description="Number of recursive iterations"
    )
    max_attempts: IntProperty(
        default=128,
        min=1, max=10000,
        description="Maximum number of placement attempts during recursion"
    )
    padding: FloatProperty(
        default=0.05,
        min=0.0001, max=1.0,
        description="Amount of encroaching padding for recursive growth"
    )
    offset: IntProperty(
        default=25,
        min=1, max=1000,
        description="Number of vertices forbidden from spawning children at beginning at end of spiral"
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
        default=False,
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

        col = layout.column(align=True)
        col.label(text="Construction Flags:")
        col.row().prop(self, "draw_circle", expand=True)
        col.row().prop(self, "draw_tangent", expand=True)
        col.row().prop(self, "edit_mode", expand=True)

        col = layout.column(align=True)
        col.label(text="Recursion Parameters:")
        col.prop(self, "iterations", text="Recursion Iterations")
        col.prop(self, "max_attempts", text="Max Attempts")
        col.prop(self, "offset", text="Growth Offset")
        col.prop(self, "contraction", text="Growth Contraction")
        col.prop(self, "padding", text="Encroaching Padding")

    def get_splines_window(self):
        window = 2
        if self.draw_tangent:
            window += 1
        if self.draw_circle:
            window += 1
        return window

    def get_selected_point(self, curve):
        selected_points = []
        for splines in windowed(curve.data.splines, self.get_splines_window()):
            spiral = splines[0]
            for points in windowed(spiral.points, 2):
                if points[1].select and points[0]:
                    selected_points.append(points)
        if selected_points:
            return selected_points[-1]

    def execute(self, context):
        time_start = time.time()

        origin = [0, 0, 0]
        tangent_angle = 0.0
        radius = self.radius
        direction = self.direction
        vertices = []
        splines_window = self.get_splines_window()
        attempts = 1
        aborted = 0

        if bpy.context.mode == 'EDIT_CURVE':
            curve = context.active_object
            previous_point, selected_point = self.get_selected_point(curve)
            if selected_point:
                origin = selected_point.co[0:3]
                length = abs(selected_point.weight)
                direction = invert_direction(
                    direction_from_sign(selected_point.weight))
                tangent_angle = angle_between_points(
                    previous_point.co, selected_point.co)
                radius = self.contraction * \
                    spiral_radius_at_length(length, self.winding_factor)
        else:
            data_curve = bpy.data.curves.new(name='Spiramir', type='CURVE')

            # Place in active scene
            curve = object_data_add(context, data_curve)
            curve.matrix_world = get_align_matrix(context, origin)
            curve.select_set(True)

            curve.data.dimensions = '2D'
            curve.data.use_path = True
            curve.data.fill_mode = 'BOTH'

            curve['spiramir_winding_factor'] = self.winding_factor
            curve['spiramir_curvature_error'] = self.curvature_error
            curve['spiramir_starting_angle'] = self.starting_angle

        spiral, spiral_radius, spiral_mass_center = make_spiral(
            radius, self.winding_factor, self.curvature_error, self.starting_angle, direction, tangent_angle)

        for _ in range(self.max_attempts):
            if spiral:
                if self.draw_circle:
                    circle = make_circle(spiral_radius, 64)
                    add_spline_to_curve(
                        curve, circle, translate(origin, spiral_mass_center))

                if self.draw_tangent:
                    tangent_vector = make_vector(tangent_angle, 0.4 * radius)
                    add_spline_to_curve(curve, tangent_vector, origin)

                add_spline_to_curve(curve, spiral, origin)
                vertices.append(max(len(spiral) - 2 * self.offset, 0))

                centripetal_vector = make_centripetal_vector(
                    spiral_mass_center)
                add_spline_to_curve(curve, centripetal_vector, origin)

                if len(vertices) >= self.iterations:
                    break

            # Random choice of mother spiral is weighted by their length
            # proxied by the number of vertices, but we need to make sure
            # the spline has more than 'self.offset' vertices so that we
            # have something to pick from as contact point.
            while True:
                mother_spiral_index = random.choices(
                    range(len(vertices)), vertices, k=1)[0]
                mother_spiral = curve.data.splines[mother_spiral_index]
                if len(mother_spiral.points) > 2 * self.offset:
                    break

            contact_point_index = random.randrange(
                self.offset, len(mother_spiral.points) - self.offset)
            contact_point = mother_spiral.points[contact_point_index]
            previous_point = mother_spiral.points[contact_point_index - 1]

            length = abs(contact_point.weight)
            direction = invert_direction(
                direction_from_sign(contact_point.weight))
            tangent_angle = angle_between_points(
                previous_point.co, contact_point.co)
            radius = self.contraction * \
                spiral_radius_at_length(length, self.winding_factor)

            origin = contact_point.co[0:3]

            spiral, spiral_radius, spiral_mass_center = make_spiral(
                radius, self.winding_factor, self.curvature_error, self.starting_angle, direction, tangent_angle)

            attempts += 1

            absolute_spiral_mass_center = translate(origin, spiral_mass_center)
            for splines in windowed(curve.data.splines, splines_window):
                spiral_under_test = splines[0]
                centripetal = splines[1]
                if spiral_under_test != mother_spiral and len(centripetal.points) == 2:
                    mass_center = centripetal.points[1]
                    radius = (
                        centripetal.points[1].co - centripetal.points[0].co).length
                    d = distance(mass_center.co[0:3],
                                 absolute_spiral_mass_center)
                    min_d = (radius + spiral_radius) * (1 + self.padding)
                    if d < min_d:
                        aborted += 1
                        spiral = None
                        break

        self.report({'INFO'}, "Drawing took %.4f sec. Spirals: %i, Attempts: %i, Aborted: %i" %
                    ((time.time() - time_start), len(vertices), attempts, aborted))

        if self.edit_mode:
            bpy.ops.object.mode_set(mode='EDIT')
        else:
            bpy.ops.object.mode_set(mode='OBJECT')

        return {'FINISHED'}


class CURVE_OT_spiramir_sprues(bpy.types.Operator):
    bl_idname = "curve.spiramir_sprues"
    bl_label = "Spiramir Sprues"
    bl_description = "Create sprues for a spiramir"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_options = {'REGISTER', 'UNDO'}

    distance: FloatProperty(
        default=0.4,
        min=0.00001, max=1000.0,
        description="Distance on the curve between sprues"
    )

    offset: FloatProperty(
        default=0.03,
        min=0.00001, max=1000.0,
        description="Offset to start from the beginning of the curve"
    )

    height: FloatProperty(
        default=1.0,
        min=0.00001, max=1000.0,
        description="Height of the end point of the sprues (on the Z axis)"
    )

    radius: FloatProperty(
        default=0.1,
        min=0.00001, max=1000.0,
        description="Radius of the top of the sprue"
    )

    curvature: FloatProperty(
        default=0.5,
        min=0.0, max=1.0,
        description="Curvature of the sprue"
    )

    def get_contact_points_for_spline(self, spline):
        contacts = []
        travel = 0.0
        previous_length = abs(spline.points[0].weight)
        first = True

        for point in spline.points:
            length = abs(point.weight)
            travel += length - previous_length
            previous_length = length
            if first and travel > self.offset:
                contacts.append(point)
                travel = 0.0
                first = False
            if travel > self.distance:
                contacts.append(point)
                travel %= self.distance

        return contacts

    def draw_sprue(self, curve, contact, contact_radius, mass_center):
        sprue = curve.data.splines.new(type='BEZIER')
        sprue.bezier_points.add(1)
        points = sprue.bezier_points

        points[0].co = contact
        points[0].radius = contact_radius
        points[0].handle_right = translate(
            contact, (0, 0, self.curvature * self.height))
        points[0].handle_right_type = 'FREE'
        points[0].handle_left = contact
        points[0].handle_left_type = 'FREE'

        top = translate(mass_center, (0, 0, self.height))
        points[1].co = top
        points[1].radius = self.radius
        points[1].handle_left = translate(
            top, (0, 0, - self.curvature * self.height))
        points[1].handle_left_type = 'FREE'
        points[1].handle_right = top
        points[1].handle_right_type = 'FREE'

    def draw_sprues(self, spiramir, context):
        origin = [0, 0, 0]
        data_curve = bpy.data.curves.new(name='Sprues', type='CURVE')

        curve = object_data_add(context, data_curve)  # Place in active scene
        curve.matrix_world = get_align_matrix(context, origin)
        curve.select_set(True)

        curve.data.dimensions = '3D'
        curve.data.resolution_u = 32
        curve.data.use_path = True
        curve.data.fill_mode = 'FULL'

        curve['spiramir_sprues'] = True

        contact_points = []

        for spline in spiramir.data.splines:
            if spline.points[0].weight != 0:
                contact_points.extend(
                    self.get_contact_points_for_spline(spline))

        mass_center = get_mass_center(contact_points)

        for point in contact_points:
            self.draw_sprue(curve, point.co[0:3], point.radius, mass_center)

    def draw(self, context):
        col = self.layout.column(align=True)
        col.prop(self, "distance", text="Distance")
        col.prop(self, "offset", text="Offset")
        col.prop(self, "height", text="Height")
        col.prop(self, "radius", text="Radius")
        col.prop(self, "curvature", text="Curvature")

    def execute(self, context):
        curve = context.active_object

        if not curve:
            self.report({'ERROR'}, "Drawing Spiramir sprues requires an object to be selected.")
            return {'FINISHED'}

        if not 'spiramir_b' in curve:
            self.report({'ERROR'}, "Drawing Spiramir sprues requires a spiramir to be selected.")
            return {'FINISHED'}

        if curve.mode != 'OBJECT':
            self.report({'ERROR'}, "Drawing Spiramir sprues requires to be in object mode.")
            return {'FINISHED'}

        self.report({'INFO'}, "Drawing sprues for spiramir: {}".format(curve))

        time_start = time.time()

        self.draw_sprues(curve, context)

        self.report({'INFO'}, "Drawing Sprues Finished: %.4f sec" %
                    (time.time() - time_start))

        return {'FINISHED'}


def menu_func(self, context):
    self.layout.operator(CURVE_OT_spiramir.bl_idname)
    self.layout.operator(CURVE_OT_spiramir_sprues.bl_idname)


addon_keymaps = []

def register():
    bpy.utils.register_class(CURVE_OT_spiramir)
    bpy.utils.register_class(CURVE_OT_spiramir_sprues)
    bpy.types.VIEW3D_MT_curve_add.append(menu_func)

    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    km = kc.keymaps.new(name="3D View Generic",
                        space_type='VIEW_3D', region_type='WINDOW')
    kmi1 = km.keymap_items.new(
        CURVE_OT_spiramir.bl_idname, 'S', 'PRESS', ctrl=True, shift=True)
    kmi2 = km.keymap_items.new(
        CURVE_OT_spiramir_sprues.bl_idname, 'T', 'PRESS', ctrl=True, shift=True)

    addon_keymaps.append((km, kmi1))
    addon_keymaps.append((km, kmi2))


def unregister():
    for km, kmi in addon_keymaps:
        km.keymap_items.remove(kmi)
    addon_keymaps.clear()

    bpy.utils.unregister_class(CURVE_OT_spiramir_sprues)
    bpy.utils.unregister_class(CURVE_OT_spiramir)
    bpy.types.VIEW3D_MT_curve_add.remove(menu_func)
