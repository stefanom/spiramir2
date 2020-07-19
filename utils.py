from collections import deque
from math import cos, sin, sqrt, log, exp, atan, atan2, pi, acos, radians
from itertools import chain, repeat
import random, sys

import bpy
from mathutils import Vector, Matrix


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
    mass_center = (0, 0)
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
    return loc @ rot


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
                    p, _, _ = bezier_multi_seg(
                        spline.bezier_points, i / steps, closed_loop=True, matrix=obj.matrix_world)
                    r = get_radius(point, tangent, normal, p)
                    if r > 0.0 and r < radius:
                        radius = r
                        contact_point = p

    if radius != sys.float_info.max:
        return radius, contact_point
    else:
        None, None
