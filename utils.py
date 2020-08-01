from mathutils import Vector
from collections import deque, OrderedDict
from math import cos, sin, sqrt, log, exp, atan, atan2, pi, acos, radians
from itertools import chain, repeat
import random
import sys

import bpy
import bpy_extras

from mathutils import Vector, Matrix
from mathutils.geometry import interpolate_bezier

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


def remove(obj):
    if obj:
        bpy.data.objects.remove(obj, do_unlink=True)


def update(cursor, constraint):
    bpy.context.view_layer.update()


# --------------------- Empty Utils --------------------------------------


def add_empty(type='ARROWS', name=None, parent=None, location=None):
    # The empty_add operator only works in object mode.
    if bpy.context.mode == 'EDIT_CURVE':
        bpy.ops.object.mode_set(mode='OBJECT')

    bpy.ops.object.empty_add(type=type)
    empty = bpy.context.selected_objects[0]

    if name:
        empty.name = name

    if location:
        empty.location = location

    if parent:
        empty.parent = parent
        empty.matrix_parent_inverse = parent.matrix_world.inverted()

    return empty


def get_constrainted_empty(curve, position):
    empty = add_empty()

    constraint = empty.constraints.new('FOLLOW_PATH')
    constraint.target = curve
    constraint.offset_factor = position
    constraint.use_curve_follow = True
    constraint.use_fixed_location = True
    constraint.forward_axis = 'FORWARD_X'
    constraint.up_axis = 'UP_Z'

    if not curve.data or not curve.data.animation_data:
        override = {'constraint': constraint}
        bpy.ops.constraint.followpath_path_animate(
            override, constraint='Follow Path')

    update(empty, constraint)

    return empty, constraint


def get_empty_orientations(empty):
    p = empty.matrix_world.translation
    rotation = empty.matrix_world.to_quaternion()
    t = rotation @ Vector((1, 0, 0))
    n = rotation @ Vector((0, 1, 0))
    return p, t, n


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

    def add_to_scene(self, context, parent=None, tube_radius=0.0, draw_circle=False):

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

        if parent:
            curve.parent = parent

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
            if parent:
                circle.parent = parent


# --------------------------- Curve Utils ------------------------------------------


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


# ----------------------------- Circles Utils -----------------------------------


def get_touching_circle_radius(p, t, n, point):
    p = point - p
    # If the points are too close, we are likely 
    # measuring adjecent points from the same curve due to
    # interpolation errors in POLY curves, so not a useful signal.
    if p.length < 0.1:
        return 0.0
    y = p.dot(n)
    if y:
        x = p.dot(t)
        return (x*x + y*y) / (2 * y)
    else:
        return 0.0


def curve_points_iterator(curves, steps):
    for curve in curves:
        for spline in curve.data.splines:
            if spline.type == 'POLY':
                for p in spline.points:
                    yield curve.matrix_world @ p.co.to_3d()
            if spline.type == 'BEZIER':
                if len(spline.bezier_points) < 2:
                    break
                segments = len(spline.bezier_points)
                if not spline.use_cyclic_u:
                    segments -= 1

                for i in range(segments):
                    inext = (i + 1) % len(spline.bezier_points)

                    knot1 = spline.bezier_points[i].co
                    handle1 = spline.bezier_points[i].handle_right
                    handle2 = spline.bezier_points[inext].handle_left
                    knot2 = spline.bezier_points[inext].co

                    for p in interpolate_bezier(knot1, handle1, handle2, knot2, steps):
                        yield curve.matrix_world @ p


def get_intersectable_curves(empty, p, n, grow_left):
    intersectable_curves = []
    evaluated = 0
    for curve in get_visible_scene_curves():
        evaluated += 1
        # The bounding box is 3D. We only grow on the XY plane so we don't really
        # care about the z coordinate and so only need to check 4 out of 8
        # vertices of the bounding box.
        for i in range(0, 8, 2):
            c = curve.bound_box[i]
            corner = curve.matrix_world @ Vector((c[0], c[1], 0))
            y = (corner - p).dot(n)
            if (grow_left and y > 0.0) or (not grow_left and y < 0.0):
                intersectable_curves.append(curve)
                break
        
    print('kept %f%% (%i out of %i)' % (100 * len(intersectable_curves) / evaluated, len(intersectable_curves), evaluated))

    return intersectable_curves


def get_available_radius(empty, grow_left=True, steps=128):
    ep, et, en = get_empty_orientations(empty)
    radius = sys.float_info.max if grow_left else -sys.float_info.max
    contact_point = None

    intersectable_curves = get_intersectable_curves(empty, ep, en, grow_left)
    for p in curve_points_iterator(intersectable_curves, steps):
        r = get_touching_circle_radius(ep, et, en, p)
        if (grow_left and r > 0.0 and r < radius) or (not grow_left and r < 0.0 and r > radius):
            radius = r
            contact_point = p

    return radius, contact_point


def add_circle(radius, parent=None, contact_point=None, bevel_depth=0.0):
    if not radius:
        print('warning: tried to draw a zero circle')
        return

    # CAREFUL: radius can be negative, signaling that we would be growing
    # on the right of the curve (when following the vertices).
    bpy.ops.curve.primitive_bezier_circle_add(
        radius=abs(radius), location=(0, radius, 0))
    circle = bpy.context.object
    circle.data.bevel_depth = bevel_depth

    if parent:
        circle.parent = parent

    if contact_point:
        add_empty(type='SINGLE_ARROW', location=contact_point,
                  parent=parent, name="Contact Point")

    return circle
