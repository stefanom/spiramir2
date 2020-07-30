from collections import deque, OrderedDict
from math import cos, sin, sqrt, log, exp, atan, atan2, pi, acos, radians
from itertools import chain, repeat
import random
import sys

import bpy
import bpy_extras

from mathutils import Vector, Matrix

# ------------------------ Python Utils -----------------------------------


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

# --------------------- Blender Utils --------------------------------------


def get_align_matrix(context, location):
    loc = Matrix.Translation(location)
    obj_align = context.preferences.edit.object_align
    if (context.space_data.type == 'VIEW_3D' and obj_align == 'VIEW'):
        rot = context.space_data.region_3d.view_matrix.to_3x3().inverted().to_4x4()
    else:
        rot = Matrix()
    return loc @ rot


def get_curve(name):
    return bpy.context.scene.objects[name]


def get_scene_objects():
    return bpy.context.scene.objects


def get_visible_scene_curves():
    curves = []
    for obj in get_scene_objects():
        if obj.type == 'CURVE' and obj.visible_get() and obj.data.bevel_depth > 0:
            curves.append(obj)
    return curves


def get_random_curve():
    curves = get_visible_scene_curves()
    return random.choice(curves) if curves else None


def get_random_spline(curve):
    return random.choice(curve.data.splines)


# --------------------- Spiral Utils --------------------------------------

CLOCKWISE = 'CLOCKWISE'
COUNTER_CLOCKWISE = 'COUNTER_CLOCKWISE'


def invert_direction(direction):
    if direction == CLOCKWISE:
        return COUNTER_CLOCKWISE
    else:
        return CLOCKWISE


def direction_sign(direction):
    return 1 if direction == CLOCKWISE else -1


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


class Spiral:

    def __init__(self,
                 direction=CLOCKWISE,
                 radius=1.0,
                 winding_factor=0.2,
                 curvature_error=0.01,
                 starting_angle=10):

        self.direction = direction
        self.radius = radius
        self.winding_factor = winding_factor
        self.curvature_error = curvature_error
        self.starting_angle = starting_angle

        vertices = []

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
            d = p.length
            p[3] = l
            p = rotation @ p
            vertices.append(p)
            if d > diameter:
                diameter = d
                mass_center = p / 2
            angle += angle_increment(angle, winding_factor, curvature_error)

        vertices.append(Vector((0, 0, 0, length)))

        # We compute the spiral from the center out, but the growth happens
        # from the outside to the center, so we need to invert the order
        # of the vertices to represent that flow in the curve.
        self.vertices = vertices[::-1]

        self.bounding_circle_radius = diameter / 2
        self.bounding_circle_center = mass_center.to_3d()

    # def is_viable(self):
    #     for curve in get_visible_scene_curves():
    #         if 'spiramir' in curve:
    #             radius = self.bounding_circle_radius
    #             center = self.bounding_circle_center
    #             other_radius = curve['spiramir_bounding_circle_radius']
    #             other_center = Vector(curve['spiramir_bounding_circle_center'])

    #             # First we check the bounding circles, which are way faster to evaluate
    #             # for overlap: two circles overlap iif the distance between their
    #             # centers is smaller than the sum of their radii. If they don't overlap
    #             # we are guaranteed the spirals will not interset.
    #             if (center - other_center).length <= radius + other_radius:
    #                 # If the bounding circles intersect, it is still possible
    #                 # that the spirals don't intersect, but we need to check
    #                 # for intersection more directly.
    #                 if self.intersects(curve):
    #                     return False
    #         else:
    #             if self.intersects(curve):
    #                 return False
    #     return True

    # def intersects(self, curve, steps=100):
    #     vertices = []
    #     for spline in curve.data.splines:
    #         if spline.type == 'POLY':
    #             vertices.extend(spline.points)
    #         if spline.type == 'BEZIER':
    #             for i in range(steps):
    #                 p, _, _ = bezier_multi_seg(
    #                     spline.bezier_points, i / steps, matrix=curve.matrix_world)
    #                 vertices.append(point)
    #       return False

    def add_to_scene(self, context, parent_curve=None, position=0.0, tube_radius=0.0, draw_circle=False):
        if parent_curve:
            empty = create_constrainted_empty(parent_curve, position)
            empty.name = "Spiramir Connector"

        data_curve = bpy.data.curves.new(name='Spiramir', type='CURVE')
        data_curve.dimensions = '2D'
        data_curve.bevel_depth = tube_radius

        curve = bpy_extras.object_utils.object_data_add(context, data_curve)

        curve['spiramir'] = True
        curve['spiramir_radius'] = self.radius
        curve['spiramir_direction'] = self.direction
        curve['spiramir_winding_factor'] = self.winding_factor
        curve['spiramir_curvature_error'] = self.curvature_error
        curve['spiramir_starting_angle'] = self.starting_angle
        curve['spiramir_bounding_circle_radius'] = self.bounding_circle_radius
        curve['spiramir_bounding_circle_center'] = self.bounding_circle_center

        if parent_curve:
            curve.parent = empty

        verts_array = []
        lengths_array = []

        for v in self.vertices:
            verts_array.extend(v[0:3])
            verts_array.append(0.0)
            lengths_array.append(v[3])

        radii_array = [abs(l)**(1./3) for l in lengths_array]

        spline = curve.data.splines.new(type='POLY')
        spline.points.add(int(len(verts_array) / 4 - 1))
        spline.points.foreach_set('co', verts_array)
        spline.points.foreach_set('weight', lengths_array)
        spline.points.foreach_set('radius', radii_array)

        if draw_circle:
            bpy.ops.curve.primitive_bezier_circle_add(
                radius=self.bounding_circle_radius, location=self.bounding_circle_center)
            circle = bpy.context.object
            circle.name = 'Bounding Circle'
            if parent_curve:
                circle.parent = empty


# ------------------------------------ Bezier Utils -------------------------------


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


# Rotation for the tangent to the normal.
normal_rot = Matrix.Rotation(radians(90.0), 4, 'Z')
normal_rot_neg = Matrix.Rotation(radians(-90.0), 4, 'Z')


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


# ----------------------------- Circles Utils -----------------------------------


def get_radius(p1, t1, n1, p2):
    p = p2 - p1
    y = p.dot(n1)
    if y:
        x = p.dot(t1)
        return (x*x + y*y) / (2 * y)
    else:
        return 0.0


def get_all_visible_curve_points(steps=100):
    for obj in get_visible_scene_curves():
        for spline in obj.data.splines:
            if spline.type == 'POLY':
                for p in spline.points:
                    p = obj.matrix_world @ p.co.to_3d()
                    yield p
            elif spline.type == 'BEZIER':
                for i in range(steps):
                    p, _, _ = bezier_multi_seg(
                        spline.bezier_points, i / steps, matrix=obj.matrix_world)
                    yield p


def get_available_radius(point, tangent, normal, grow_left=True, steps=100):
    radius = sys.float_info.max if grow_left else -sys.float_info.max
    contact_point = None

    for p in get_all_visible_curve_points(steps=steps):
        r = get_radius(point, tangent, normal, p)
        if grow_left and r > 0.0 and r < radius:
            radius = r
            contact_point = p
        if not grow_left and r < 0.0 and r > radius:
            radius = r
            contact_point = p

    return radius, contact_point

def add_circle(curve, position, radius, contact_point=None, bevel_depth=0.0):
    if not radius:
        print('warning: tried to draw a zero circle')
        return

    empty = create_constrainted_empty(curve, position)
    empty.name = 'Circle Connector'
    #empty.parent = curve
    #empty.matrix_parent_inverse = curve.matrix_world.inverted()

    # CAREFUL: radius can be negative, signaling that we would be growing
    # on the outside of the curve.
    bpy.ops.curve.primitive_bezier_circle_add(radius=abs(radius), location=(0, radius, 0))
    circle = bpy.context.object
    #circle.name = "Availability Circle"
    circle.parent = empty
    circle.data.bevel_depth = bevel_depth

    if contact_point:
        bpy.ops.object.empty_add(type='SINGLE_ARROW', location=contact_point)
        contact_point_empty = bpy.context.object
        contact_point_empty.name = "Contact Point"
        #contact_point_empty.parent = empty
        #contact_point_empty.matrix_parent_inverse = empty.matrix_world.inverted()

    return circle

# --------------------------- Curve Utils ------------------------------------------

def create_constrainted_empty(curve, position):
    if bpy.context.mode == 'EDIT_CURVE':
        bpy.ops.object.mode_set(mode='OBJECT')

    bpy.ops.object.empty_add(type='ARROWS')
    empty = bpy.context.object

    constraint = empty.constraints.new('FOLLOW_PATH')
    constraint.target = curve
    constraint.offset_factor = position
    constraint.use_curve_follow = True
    constraint.use_fixed_location = True
    constraint.forward_axis = 'FORWARD_X'
    constraint.up_axis = 'UP_Z'

    if not curve.data.animation_data:
        override = {'constraint': constraint}
        bpy.ops.constraint.followpath_path_animate(
            override, constraint='Follow Path')

    return empty


def get_selected_point(curve):
    for spline in curve.data.splines:
        selected_point = None
        selected_point_length = 0.0
        selected_point_weight = 0.0

        if spline.type == 'POLY':
            length = 0.0
            for point, next_point in windowed(spline.points, 2):
                if point.select:
                    selected_point = point
                    selected_point_length = length
                    selected_point_weight = point.weight
                p = curve.matrix_world @ point.co.to_3d()
                np = curve.matrix_world @ next_point.co.to_3d()
                length += (np - p).length

        if selected_point:
            return selected_point_length / length, selected_point_weight


def get_point_on_curve(curve, position):
    if len(curve.data.splines) > 1:
        raise ValueError('This function only works with curves with a single spline.')

    spline = curve.data.splines[0]

    if spline.type == 'BEZIER':
        return bezier_multi_seg(spline.bezier_points, position, matrix=curve.matrix_world)

    elif spline.type == 'POLY':
        points = OrderedDict()
        length = 0.0

        for previous_point, point, next_point in windowed(spline.points, 3):
            previous_point = curve.matrix_world @ previous_point.co.to_3d()
            point = curve.matrix_world @ point.co.to_3d()
            next_point = curve.matrix_world @ next_point.co.to_3d()

            length += (point - previous_point).length
            tangent = next_point - previous_point
            tangent.normalize()
            normal = normal_rot @ tangent

            points[length] = (point, tangent, normal)

        position *= length

        for l, v in points.items():
            if l >= position:
                return v


def get_weight_at_position(curve, position):
    if 'spiramir' not in curve:
        raise ValueError('This function only works with spiramir curves.')

    spline = curve.data.splines[0]

    if spline.type == 'POLY':
        weights = OrderedDict()
        length = 0.0
        for point, next_point in windowed(spline.points, 2):
            length += (next_point.co - point.co).length
            weights[length] = point.weight

        position *= length

        for l, weight in weights.items():
            if l >= position:
                return weight
