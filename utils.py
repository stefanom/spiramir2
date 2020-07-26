from collections import deque
from math import cos, sin, sqrt, log, exp, atan, atan2, pi, acos, radians
from itertools import chain, repeat
import random
import sys

import bpy
from mathutils import Vector, Matrix

# -------------------------- good ------------------------


def get_align_matrix(context, location):
    loc = Matrix.Translation(location)
    obj_align = context.preferences.edit.object_align
    if (context.space_data.type == 'VIEW_3D' and obj_align == 'VIEW'):
        rot = context.space_data.region_3d.view_matrix.to_3x3().inverted().to_4x4()
    else:
        rot = Matrix()
    return loc @ rot


def invert_direction(direction):
    if direction == 'CLOCKWISE':
        return 'COUNTER_CLOCKWISE'
    else:
        return 'CLOCKWISE'


def direction_sign(direction):
    return 1 if direction == 'CLOCKWISE' else -1


def spiral_cartesian(t, b, direction):
    sign = direction_sign(direction)
    r = exp(b * t)
    return Vector((r * cos(t), sign * r * sin(t), 0, 0))


def spiral_angle_at_length(l, b):
    return log(l * b / sqrt(1 + b*b)) / b


def spiral_radius_at_length(l, b):
    return l * b / sqrt(1 + b*b)


def spiral_length_at_angle(t, b):
    return sqrt(1 + b*b) * exp(b * t) / b


def obsculating_circle_radius_at_angle(t, b):
    return b * spiral_length_at_angle(t, b)


def angle_increment(t, b, curvature_error):
    r = obsculating_circle_radius_at_angle(t, b)
    return acos(r / (r + curvature_error))


def make_spiral(radius, winding_factor, curvature_error, starting_angle, direction):
    verts = []

    sign = direction_sign(direction)
    t = log(radius) / winding_factor
    length = spiral_length_at_angle(t, winding_factor)
    rot = atan(1 / winding_factor)
    rotation = Matrix.Rotation(-sign * (t + rot + pi), 4, 'Z')
    end = spiral_cartesian(t, winding_factor, direction)
    angle = starting_angle

    diameter = 0
    mass_center = Vector((0, 0, 0))

    while True:
        l = spiral_length_at_angle(angle, winding_factor)
        if l >= length:
            break
        p = spiral_cartesian(angle, winding_factor, direction) - end
        p[3] = l
        p = rotation @ p
        verts.append(p)
        d = p.length
        if d > diameter:
            diameter = d
            mass_center = p / 2
        angle += angle_increment(angle, winding_factor, curvature_error)

    verts.append(Vector((0, 0, 0, length)))

    return verts[::-1], diameter / 2, mass_center


def verts_to_points(verts):
    vert_array = []
    length_array = []
    for v in verts:
        vert_array.extend(v[0:3])
        vert_array.append(0.0)
        length_array.append(v[3])
    return vert_array, length_array


def add_spline_to_curve(curve, verts):
    verts_array, lengths_array = verts_to_points(verts)

    new_spline = curve.data.splines.new(type='POLY')
    new_spline.points.add(int(len(verts_array) / 4 - 1))
    new_spline.points.foreach_set('co', verts_array)
    new_spline.points.foreach_set('weight', lengths_array)
    new_spline.points.foreach_set(
        'radius', [abs(l)**(1./3) for l in lengths_array])
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


def get_selected_point_position_and_weight(curve):
    for spline in curve.data.splines:
      selected_point = None
      selected_point_length = 0.0
      selected_point_weight = 0.0

      if spline.type == 'POLY':
        length = 0.0
        for point, next_point in windowed(spline.points, 2):
          length += (point.co - next_point.co).length
          if point.select:
            selected_point = point
            selected_point_length = length
            selected_point_weight = point.weight

      if selected_point:
        return selected_point_length / length, selected_point_weight

# -------------------------- bad ------------------------


def translate(p, t):
    if len(p) == 3 and len(t) == 3:
        return (p[0] + t[0], p[1] + t[1], p[2] + t[2])
    else:
        return (p[0] + t[0], p[1] + t[1])


def distance(a, b):
    return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def between(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


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
