from math import cos, sin, log, exp, atan, pi

import random, time

import bpy
from bpy_extras.object_utils import object_data_add
from mathutils import Vector, Matrix

from . import utils

class CURVE_OT_spiramir(bpy.types.Operator):
    bl_idname = "curve.spiramir"
    bl_label = "Spiramir"
    bl_description = "Create a spiramir"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_options = {'REGISTER', 'UNDO'}

    direction: bpy.props.EnumProperty(
        items=[('COUNTER_CLOCKWISE', "Counter Clockwise",
                "Wind in a counter clockwise direction"),
               ("CLOCKWISE", "Clockwise",
                "Wind in a clockwise direction")],
        default='COUNTER_CLOCKWISE',
        name="Direction",
        description="Direction of winding"
    )
    radius: bpy.props.FloatProperty(
        default=1.0,
        min=0.00001, max=1000.0,
        description="Radius of the spiral"
    )
    winding_factor: bpy.props.FloatProperty(
        default=0.2,
        min=0.0001, max=2,
        description="Spiral Winding Factor"
    )
    starting_angle: bpy.props.FloatProperty(
        default=-20.0,
        min=-100.0, max=100.0,
        description="Angle to start drawing the spiral"
    )
    curvature_error: bpy.props.FloatProperty(
        default=0.001,
        min=0.0000001, max=1000.0,
        description="Maximum curvature error"
    )
    min_growth_ratio: bpy.props.FloatProperty(
        default=0.6,
        min=0.0, max=1.0,
        description="Minimum growth ratio"
    )
    max_growth_ratio: bpy.props.FloatProperty(
        default=0.95,
        min=0.0, max=1.0,
        description="Maximum growth ratio"
    )
    iterations: bpy.props.IntProperty(
        default=64,
        min=1, max=500,
        description="Number of recursive iterations"
    )
    max_attempts: bpy.props.IntProperty(
        default=512,
        min=1, max=10000,
        description="Maximum number of placement attempts during recursion"
    )
    min_distance: bpy.props.FloatProperty(
        default=0.001,
        min=0.0001, max=1.0,
        description="The minimum distance between spirals"
    )
    offset: bpy.props.IntProperty(
        default=25,
        min=1, max=1000,
        description="Number of vertices forbidden from spawning children at beginning at end of spiral"
    )
    draw_circle: bpy.props.BoolProperty(
        name="Draw circle",
        default=False,
        description="Draw the occupancy circle"
    )
    draw_tangent: bpy.props.BoolProperty(
        name="Draw tangent",
        default=False,
        description="Draw the tangent vector"
    )
    edit_mode: bpy.props.BoolProperty(
        name="Show in edit mode",
        default=False,
        description="Show in edit mode"
    )
    verbose: bpy.props.BoolProperty(
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
        for splines in utils.windowed(curve.data.splines, self.get_splines_window()):
            spiral = splines[0]
            for points in utils.windowed(spiral.points, 2):
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
        self.log(
            "\n\n=========================== Execute ==================================")
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
                direction = utils.invert_direction(
                    utils.direction_from_sign(selected_point.weight))
                tangent_angle = utils.angle_between_points(
                    previous_point.co, selected_point.co)
                radius = self.max_growth_ratio * \
                    utils.spiral_radius_at_length(length, self.winding_factor)

                for spiral, centripetal in utils.windowed(curve.data.splines, self.get_splines_window()):
                    if len(spiral.points[0].co) == 4 and spiral.points[0].co[4] != 0 and len(centripetal.points) == 2:
                        spirals.append(spiral)
                        spirals_centripetals.append(centripetal)
                        if len(spiral.points) > 2 * self.offset:
                            fertile_spirals.append(spiral)
                            fertile_spirals_weights.append(len(spiral.points))
        else:
            data_curve = bpy.data.curves.new(name='Spiramir', type='CURVE')

            curve = object_data_add(context, data_curve)
            curve.matrix_world = utils.get_align_matrix(context, origin)
            curve.select_set(True)

            curve.data.dimensions = '2D'
            curve.data.use_path = True
            curve.data.fill_mode = 'BOTH'

            curve['spiramir_winding_factor'] = self.winding_factor
            curve['spiramir_curvature_error'] = self.curvature_error
            curve['spiramir_starting_angle'] = self.starting_angle

        spiral, spiral_radius, spiral_mass_center = utils.make_spiral(
            radius, self.winding_factor, self.curvature_error, self.starting_angle, direction, tangent_angle)

        for _ in range(self.max_attempts):
            if spiral:
                if self.draw_circle:
                    circle = utils.make_circle(spiral_radius, 64)
                    utils.add_spline_to_curve(
                        curve, circle, utils.translate(origin, spiral_mass_center))

                if self.draw_tangent:
                    tangent_vector = utils.make_vector(tangent_angle, 0.4 * radius)
                    utils.add_spline_to_curve(curve, tangent_vector, origin)

                spline = utils.add_spline_to_curve(curve, spiral, origin)

                spirals.append(spline)

                centripetal_vector = utils.make_centripetal_vector(
                    spiral_mass_center)
                centripetal_spline = utils.add_spline_to_curve(
                    curve, centripetal_vector, origin)
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
            direction = utils.invert_direction(
                utils.direction_from_sign(contact_point.weight))
            tangent_angle = utils.angle_between_points(
                previous_point.co, contact_point.co)
            mother_radius = utils.spiral_radius_at_length(
                length, self.winding_factor)
            radius = random.uniform(
                self.min_growth_ratio * mother_radius, self.max_growth_ratio * mother_radius)

            spiral, spiral_radius, spiral_mass_center = utils.make_spiral(
                radius, self.winding_factor, self.curvature_error, self.starting_angle, direction, tangent_angle)

            attempts += 1

            absolute_spiral_mass_center = utils.translate(origin, spiral_mass_center)
            for spiral_under_test, centripetal in zip(spirals, spirals_centripetals):
                self.log("spiral under test: ", spiral_under_test)
                if spiral_under_test != mother_spiral:
                    self.log(" testing spiral!")
                    mass_center = centripetal.points[1]
                    radius = (
                        centripetal.points[1].co - centripetal.points[0].co).length
                    d = utils.distance(mass_center.co[0:3],
                                 absolute_spiral_mass_center)
                    if d < radius + spiral_radius:
                        self.log(" circles intersect! testing spiral distance")
                        min_distance = utils.minimum_distance(
                            spiral, spiral_under_test)
                        if min_distance > self.min_distance:
                            self.log(
                                "  spirals were too close, aborting!", min_distance)
                            aborted += 1
                            spiral = None
                            break
                        else:
                            self.log(
                                "  spiral was far enough, drawing! ", min_distance)
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


class CURVE_PT_spiramir2(bpy.types.Panel):
    bl_idname = "CURVE_PT_spiramir2"
    bl_label = "Spiramir Grower"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Create"

    def draw(self, context):
        parent_curve = context.active_object
        # If the selected object is a spiramir, the props defaults
        # should change to align with growth coming from that parent.
        if parent_curve and parent_curve.type == 'CURVE':
            props = self.layout.operator('curve.spiramir2')
            props.position = 0.5
            props.radius = parent_curve['spiramir_radius'] * 0.9
            props.direction = utils.invert_direction(
                parent_curve['spiramir_direction'])
            props.winding_factor = parent_curve['spiramir_winding_factor']
            props.curvature_error = parent_curve['spiramir_curvature_error']
            props.starting_angle = parent_curve['spiramir_starting_angle']


class CURVE_OT_spiramir2(bpy.types.Operator):
    bl_idname = "curve.spiramir2"
    bl_label = "Add Spiramir"
    bl_options = {'REGISTER', 'UNDO'}

    verbose = True

    direction: bpy.props.EnumProperty(
        items=[('COUNTER_CLOCKWISE', "Counter Clockwise",
                "Wind in a counter clockwise direction"),
               ("CLOCKWISE", "Clockwise",
                "Wind in a clockwise direction")],
        default='COUNTER_CLOCKWISE',
        name="Direction",
        description="Direction of winding"
    )
    position: bpy.props.FloatProperty(
        default=0.0,
        min=0.0, max=100.0,
        description="Relative position of the growth point"
    )
    radius: bpy.props.FloatProperty(
        default=1.0,
        min=0.00001, max=1000.0,
        description="Radius of the spiral"
    )
    winding_factor: bpy.props.FloatProperty(
        default=0.2,
        min=0.0001, max=2,
        description="Spiral Winding Factor"
    )
    starting_angle: bpy.props.FloatProperty(
        default=-20.0,
        min=-100.0, max=100.0,
        description="Angle to start drawing the spiral"
    )
    curvature_error: bpy.props.FloatProperty(
        default=0.001,
        min=0.0000001, max=1000.0,
        description="Maximum curvature error"
    )
    tube_radius: bpy.props.FloatProperty(
        default=0.01,
        min=0.0, max=1000.0,
        description="Radius of the tube around the curve"
    )

    def log(self, message, *args):
        if self.verbose:
            if args:
                print(message, args)
            else:
                print(message)

    def direction_sign(self, direction):
        return 1 if direction == 'CLOCKWISE' else -1

    def spiral_cartesian(self, t, b, direction):
        sign = self.direction_sign(direction)
        r = exp(b * t)
        return Vector((r * cos(t), sign * r * sin(t), 0, 0))

    def make_spiral(self):
        verts = []

        sign = self.direction_sign(self.direction)
        t = log(self.radius) / self.winding_factor
        length = utils.spiral_length_at_angle(t, self.winding_factor)
        rot = atan(1 / self.winding_factor)
        rotation = Matrix.Rotation(-sign * (t + rot + pi), 4, 'Z')
        end = self.spiral_cartesian(t, self.winding_factor, self.direction)
        angle = self.starting_angle

        diameter = 0
        mass_center = Vector((0, 0, 0))

        while True:
            l = utils.spiral_length_at_angle(angle, self.winding_factor)
            if l >= length:
                break
            p = self.spiral_cartesian(
                angle, self.winding_factor, self.direction) - end
            p[3] = l
            p = rotation @ p
            verts.append(p)
            d = p.length
            if d > diameter:
                diameter = d
                mass_center = p / 2
            angle += utils.angle_increment(angle,
                                     self.winding_factor, self.curvature_error)

        verts.append(Vector((0, 0, 0, length)))

        return verts[::-1], diameter / 2, mass_center

    def verts_to_points(self, verts):
        vert_array = []
        length_array = []
        for v in verts:
            vert_array.extend(v[0:3])
            vert_array.append(0.0)
            length_array.append(v[3])
        return vert_array, length_array

    def add_spline_to_curve(self, curve, verts):
        verts_array, lengths_array = self.verts_to_points(verts)

        new_spline = curve.data.splines.new(type='POLY')
        new_spline.points.add(int(len(verts_array) / 4 - 1))
        new_spline.points.foreach_set('co', verts_array)
        new_spline.points.foreach_set('weight', lengths_array)
        new_spline.points.foreach_set(
            'radius', [abs(l)**(1./3) for l in lengths_array])
        return new_spline

    # def invoke(self, context, event):
    #     self.log("\n\n=========================== Invoke ==================================")
    #     #wm = context.window_manager
    #     #return wm.invoke_props_dialog(self)
    #     return {'FINISHED'}

    def draw(self, context):
        layout = self.layout
        col = layout.column_flow(align=True)

        col.prop(self, "direction")

        col = layout.column(align=True)
        col.prop(self, "radius", text="Radius")
        col.prop(self, "position", text="Relative Position")
        col.prop(self, "winding_factor", text="Winding Factor")
        col.prop(self, "starting_angle", text="Starting Angle")
        col.prop(self, "curvature_error", text="Curvature Error")
        col.prop(self, "tube_radius", text="Tube Radius")

    def execute(self, context):
        self.log(
            "\n\n=========================== Execute ==================================")
        time_start = time.time()

        parent_curve = context.object
        if parent_curve and (parent_curve.type != 'CURVE' or not parent_curve.select_get()):
            parent_curve = None

        origin = Vector((0, 0, 0))
        bpy.ops.object.empty_add(type='ARROWS', location=origin)
        empty = context.object

        if parent_curve:
            constraint = empty.constraints.new('FOLLOW_PATH')
            constraint.target = parent_curve
            constraint.offset_factor = self.position
            constraint.use_curve_follow = True
            constraint.use_fixed_location = True
            constraint.forward_axis = 'FORWARD_X'
            constraint.up_axis = 'UP_Z'

            if not parent_curve.data.animation_data:
                override = {'constraint': constraint}
                bpy.ops.constraint.followpath_path_animate(
                    override, constraint='Follow Path')

        data_curve = bpy.data.curves.new(name='Spiramir', type='CURVE')
        data_curve.dimensions = '2D'
        data_curve.bevel_depth = self.tube_radius

        curve = object_data_add(context, data_curve)

        curve['spiramir'] = True
        curve['spiramir_radius'] = self.radius
        curve['spiramir_direction'] = self.direction
        curve['spiramir_winding_factor'] = self.winding_factor
        curve['spiramir_curvature_error'] = self.curvature_error
        curve['spiramir_starting_angle'] = self.starting_angle

        curve.parent = empty

        spiral, _, _ = self.make_spiral()

        self.add_spline_to_curve(curve, spiral)

        self.log("Drawing took %.4f sec." % (time.time() - time_start))

        context.view_layer.update()

        return {'FINISHED'}
