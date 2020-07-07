from collections import deque
from itertools import chain, repeat
import time, random, sys
from mathutils import Vector, Matrix, geometry
from math import cos, sin, sqrt, log, exp, atan, atan2, pi, acos, radians

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
    return new_spline


def minimum_distance(new_spiral, existing_spiral):
    min_distance = sys.float_info.max
    for p in new_spiral:
        for sp in existing_spiral.points:
            d = (Vector(p) - sp.co).length
            if d < min_distance:
                min_distance = d
    return min_distance


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
    min_growth_ratio: FloatProperty(
        default=0.6,
        min=0.0, max=1.0,
        description="Minimum growth ratio"
    )
    max_growth_ratio: FloatProperty(
        default=0.95,
        min=0.0, max=1.0,
        description="Maximum growth ratio"
    )
    iterations: IntProperty(
        default=64,
        min=1, max=500,
        description="Number of recursive iterations"
    )
    max_attempts: IntProperty(
        default=512,
        min=1, max=10000,
        description="Maximum number of placement attempts during recursion"
    )
    min_distance: FloatProperty(
        default=0.001,
        min=0.0001, max=1.0,
        description="The minimum distance between spirals"
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
    verbose: BoolProperty(
        name="Verbose logging",
        default=False,
        description="Print verbose logging in the debug console"
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
        col.row().prop(self, "verbose", expand=True)

        col = layout.column(align=True)
        col.label(text="Recursion Parameters:")
        col.prop(self, "iterations", text="Recursion Iterations")
        col.prop(self, "max_attempts", text="Max Attempts")
        col.prop(self, "offset", text="Growth Offset")
        col.prop(self, "min_growth_ratio", text="Min Growth Ratio")
        col.prop(self, "max_growth_ratio", text="Max Growth Ratio")
        col.prop(self, "min_distance", text="Minimum Distance")

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

    def log(self, message, *args):
        if self.verbose:
            if args:
                print(message, args)
            else:
                print(message)

    def execute(self, context):
        self.log("\n\n=========================== Execute ==================================")
        time_start = time.time()

        origin = (0, 0, 0)
        tangent_angle = 0.0
        radius = self.radius
        direction = self.direction
        spirals = []
        spirals_centripetals = []
        fertile_spirals = []
        fertile_spirals_weights = []
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
                radius = self.max_growth_ratio * spiral_radius_at_length(length, self.winding_factor)

                for spiral, centripetal in windowed(curve.data.splines, self.get_splines_window()):
                    if len(spiral.points[0].co) == 4 and spiral.points[0].co[4] != 0 and len(centripetal.points) == 2:
                        spirals.append(spiral)
                        spirals_centripetals.append(centripetal)
                        if len(spiral.points) > 2 * self.offset:
                            fertile_spirals.append(spiral)
                            fertile_spirals_weights.append(len(spiral.points))
        else:
            data_curve = bpy.data.curves.new(name='Spiramir', type='CURVE')

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

                spline = add_spline_to_curve(curve, spiral, origin)

                spirals.append(spline)

                centripetal_vector = make_centripetal_vector(
                    spiral_mass_center)
                centripetal_spline = add_spline_to_curve(curve, centripetal_vector, origin)
                spirals_centripetals.append(centripetal_spline)

                if len(spline.points) > 2 * self.offset:
                    fertile_spirals.append(spline)
                    fertile_spirals_weights.append(len(spline.points))

                if len(spirals) >= self.iterations:
                    break

            self.log("fertile spirals: ", fertile_spirals)

            mother_spiral = random.choices(
                fertile_spirals, fertile_spirals_weights, k=1)[0]
            self.log('mother spiral: ', mother_spiral)

            contact_point_index = random.randrange(
                self.offset, len(mother_spiral.points) - self.offset)
            contact_point = mother_spiral.points[contact_point_index]
            previous_point = mother_spiral.points[contact_point_index - 1]
            origin = contact_point.co[0:3]

            self.log('contact point: ', origin, contact_point_index)

            length = abs(contact_point.weight)
            direction = invert_direction(
                direction_from_sign(contact_point.weight))
            tangent_angle = angle_between_points(
                previous_point.co, contact_point.co)
            mother_radius = spiral_radius_at_length(
                length, self.winding_factor)
            radius = random.uniform(self.min_growth_ratio * mother_radius, self.max_growth_ratio * mother_radius)

            spiral, spiral_radius, spiral_mass_center = make_spiral(
                radius, self.winding_factor, self.curvature_error, self.starting_angle, direction, tangent_angle)

            attempts += 1

            absolute_spiral_mass_center = translate(origin, spiral_mass_center)
            for spiral_under_test, centripetal in zip(spirals, spirals_centripetals):
                self.log("spiral under test: ", spiral_under_test)
                if spiral_under_test != mother_spiral:
                    self.log(" testing spiral!")
                    mass_center = centripetal.points[1]
                    radius = (centripetal.points[1].co - centripetal.points[0].co).length
                    d = distance(mass_center.co[0:3], absolute_spiral_mass_center)
                    if d < radius + spiral_radius:
                        self.log(" circles intersect! testing spiral distance")
                        min_distance = minimum_distance(spiral, spiral_under_test)
                        if min_distance > self.min_distance:
                            self.log("  spirals were too close, aborting!", min_distance)
                            aborted += 1
                            spiral = None
                            break
                        else:
                            self.log("  spiral was far enough, drawing! ", min_distance)
                    else:
                        self.log("  spiral circle didn't intersect!")

        self.report({'INFO'}, "Drawing took %.4f sec. Spirals: %i, Attempts: %i, Aborted: %i" %
                    ((time.time() - time_start), len(spirals), attempts, aborted))

        if self.edit_mode:
            bpy.ops.object.mode_set(mode='EDIT')
        else:
            bpy.ops.object.mode_set(mode='OBJECT')

        self.log(">>>>>>>>>>>>>>>>>>>>>>>>>>>> done <<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

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


# Rotation for the tangent to the normal.
normal_rot = Matrix.Rotation(radians(90.0), 4, 'Z')


def get_curve(name):
    return bpy.context.scene.objects[name]


def get_random_curve():
    scene = bpy.context.scene
    curves = []
    for obj in scene.objects:
        if obj.type == 'CURVE' and obj.visible_get():
            curves.append(obj)
    return random.choice(curves)


def get_random_spline(curve):
    return random.choice(curve.data.splines)


def bezier_step(pt0, pt1, pt2, pt3, step=0.5):
    if step <= 0.0:
        return pt0.copy()
    if step >= 1.0:
        return pt3.copy()

    u = 1.0 - step
    tcb = step * step
    ucb = u * u
    tsq3u = tcb * 3.0 * u
    usq3t = ucb * 3.0 * step
    tcb *= step
    ucb *= u

    return pt0 * ucb + pt1 * usq3t + pt2 * tsq3u + pt3 * tcb


def bezier_tangent(pt0, pt1, pt2, pt3, step=0.5):
    if step <= 0.0:
        return pt1 - pt0
    if step >= 1.0:
        return pt3 - pt2

    u = 1.0 - step
    ut6 = u * step * 6.0
    tsq3 = step * step * 3.0
    usq3 = u * u * 3.0

    return (pt1 - pt0) * usq3 + (pt2 - pt1) * ut6 + (pt3 - pt2) * tsq3


def bezier_multi_seg(knots=[], step=0.0, closed_loop=True, matrix=Matrix()):
    knots_len = len(knots)

    if knots_len == 1:
        knot = knots[0]
        point = knot.co.copy()
        tangent = knot.handle_right - point
        tangent.normalize()
        normal = normal_rot @ tangent
        return point, tangent, normal

    if closed_loop:
        scaled_t = (step % 1.0) * knots_len
        index = int(scaled_t)
        a = knots[index]
        b = knots[(index + 1) % knots_len]
    else:
        if step <= 0.0:
            knot = knots[0]
            point = knot.co.copy()
            tangent = knot.handle_right - point
            tangent.normalize()
            normal = normal_rot @ tangent
            return point, tangent, normal
        if step >= 1.0:
            knot = knots[-1]
            tangent = knot.handle_right - point
            tangent.normalize()
            normal = normal_rot @ tangent
            return point, tangent, normal

        scaled_t = step * (knots_len - 1)
        index = int(scaled_t)
        a = knots[index]
        b = knots[index + 1]

    pt0 = a.co
    pt1 = a.handle_right
    pt2 = b.handle_left
    pt3 = b.co
    u = scaled_t - index

    # Obtain the point in local coordinates.
    point = bezier_step(pt0, pt1, pt2, pt3, step=u)
    # Obtain the tanget of the same step in local coordinates.
    tangent = bezier_tangent(pt0, pt1, pt2, pt3, step=u)
    # Normalize the tangential vector.
    tangent.normalize()
    # Rotate it 90' to obtain the outside normal.
    normal = normal_rot @ tangent

    # Apply the matrix transformation to the point.
    point = matrix @ point
    # Obtain the rotational part of the matrix.
    rot = matrix.to_quaternion()
    # And apply it to the tangent and normal. These are 1-unit directional vectors
    # so we just rotate them instead of applying the full matrix, if not translation
    # and scale can mess things up.
    tangent = rot @ tangent
    normal = rot @ normal

    return point, tangent, normal


def get_radius(p1, t1, n1, p2):
    p = p2 - p1
    x = p.dot(t1)
    y = p.dot(n1)
    return (x*x + y*y) / (2 * y) if y != 0.0 else 0.0


def get_available_radius(curve, point, tangent, normal, steps=1000):
    radius = sys.float_info.max
    contact_point = None

    for obj in bpy.context.scene.objects:
        if obj.type == 'CURVE' and obj.visible_get():
            for spline in obj.data.splines:
                if spline.type != 'BEZIER':
                    continue
                for i in range(steps):
                    p, _, _ = bezier_multi_seg(spline.bezier_points, i / steps, closed_loop=True, matrix=obj.matrix_world)
                    r = get_radius(point, tangent, normal, p)
                    if r > 0.0 and r < radius:
                        radius = r
                        contact_point = p

    if radius != sys.float_info.max:
        return radius, contact_point
    else:
        None, None


class CURVE_OT_spiramir_circles(bpy.types.Operator):
    bl_idname = "curve.spiramir_circles"
    bl_label = "Spiramir Circles"
    bl_description = "Enclose with circles"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_options = {'REGISTER', 'UNDO'}

    iterations: IntProperty(
        default=1,
        min=1, max=500,
        description="Number of circles to inscribe"
    )

    max_attempts: IntProperty(
        default=100,
        min=1, max=10000,
        description="Maximum number of attempts before giving up"
    )

    min_radius: FloatProperty(
        default=0.1,
        min=0.0, max=1000.0,
        description="Smallest radius allowed for embedded circles"
    )

    max_radius: FloatProperty(
        default=0.7,
        min=0.0, max=1000.0,
        description="Biggest radius allowed for embedded circles"
    )

    bevel_depth: FloatProperty(
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
        curve = get_random_curve()
        spline = get_random_spline(curve)
        step = random.uniform(0.0, 1.0)

        p, t, n = bezier_multi_seg(spline.bezier_points, step, closed_loop=True, matrix=curve.matrix_world)
        r, cp = get_available_radius(curve, p, t, n)

        if r > self.min_radius:
            r = min(r, self.max_radius)

            bpy.ops.object.empty_add(type='ARROWS', location=p)
            empty = bpy.context.object
            empty.rotation_mode = 'QUATERNION'
            empty.rotation_quaternion = t.to_track_quat('X', 'Z')

            bpy.ops.curve.primitive_bezier_circle_add(radius=r, location=(0, r, 0))
            circle = bpy.context.object
            circle.parent = empty
            circle.data.bevel_depth = self.bevel_depth

            bpy.ops.object.empty_add(type='SINGLE_ARROW', location=cp)
            return True
        else:
            return False

    def execute(self, context):
        print('>>>>>>>>>> start >>>>>>>>>>>>>>>>')
        time_start = time.time()

        drawn = 0
        for _ in range(self.max_attempts):
            if self.inscribe_circle():
                print('[*] Successful inscription.')
                drawn += 1
                if drawn == self.iterations:
                    break
            else:
                print('[ ] Failed inscription.')

        self.report({'INFO'}, "Drawn %d circles in %.4f sec" % (drawn, time.time() - time_start))

        print('<<<<<<<<<<<<<<< end <<<<<<<<<<<<<<')

        return {'FINISHED'}


def menu_func(self, context):
    self.layout.operator(CURVE_OT_spiramir.bl_idname)
    self.layout.operator(CURVE_OT_spiramir_sprues.bl_idname)
    self.layout.operator(CURVE_OT_spiramir_circles.bl_idname)


addon_keymaps = []

def register():
    bpy.utils.register_class(CURVE_OT_spiramir)
    bpy.utils.register_class(CURVE_OT_spiramir_sprues)
    bpy.utils.register_class(CURVE_OT_spiramir_circles)
    bpy.types.VIEW3D_MT_curve_add.append(menu_func)

    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    km = kc.keymaps.new(name="3D View Generic",
                        space_type='VIEW_3D', region_type='WINDOW')
    kmi1 = km.keymap_items.new(
        CURVE_OT_spiramir.bl_idname, 'S', 'PRESS', ctrl=True, shift=True)
    kmi2 = km.keymap_items.new(
        CURVE_OT_spiramir_sprues.bl_idname, 'R', 'PRESS', ctrl=True, shift=True)
    kmi3 = km.keymap_items.new(
        CURVE_OT_spiramir_circles.bl_idname, 'C', 'PRESS', ctrl=True, shift=True)

    addon_keymaps.append((km, kmi1))
    addon_keymaps.append((km, kmi2))
    addon_keymaps.append((km, kmi3))


def unregister():
    for km, kmi in addon_keymaps:
        km.keymap_items.remove(kmi)
    addon_keymaps.clear()

    bpy.utils.unregister_class(CURVE_OT_spiramir_circles)
    bpy.utils.unregister_class(CURVE_OT_spiramir_sprues)
    bpy.utils.unregister_class(CURVE_OT_spiramir)
    bpy.types.VIEW3D_MT_curve_add.remove(menu_func)
