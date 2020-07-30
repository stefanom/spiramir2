import bpy
from bpy_extras.object_utils import object_data_add
from mathutils import Vector

from . import utils

class CURVE_OT_spiramir_sprues(bpy.types.Operator):
    bl_idname = "curve.spiramir_sprues"
    bl_label = "Sprues"
    bl_description = "Create sprues for a spiramir"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Create"
    bl_options = {'REGISTER', 'UNDO'}

    distance: bpy.props.FloatProperty(
        default=0.4,
        min=0.00001, max=1000.0,
        description="Distance on the curve between sprues"
    )
    offset: bpy.props.FloatProperty(
        default=0.03,
        min=0.00001, max=1000.0,
        description="Offset to start from the beginning of the curve"
    )
    height: bpy.props.FloatProperty(
        default=1.0,
        min=0.00001, max=1000.0,
        description="Height of the end point of the sprues (on the Z axis)"
    )
    radius: bpy.props.FloatProperty(
        default=0.1,
        min=0.00001, max=1000.0,
        description="Radius of the top of the sprue"
    )
    curvature: bpy.props.FloatProperty(
        default=0.5,
        min=0.0, max=1.0,
        description="Curvature of the sprue"
    )

    def draw(self, context):
        col = self.layout.column(align=True)
        col.prop(self, "distance", text="Distance")
        col.prop(self, "offset", text="Offset")
        col.prop(self, "height", text="Height")
        col.prop(self, "radius", text="Radius")
        col.prop(self, "curvature", text="Curvature")

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'

    def get_contact_points_for_spline(self, spline):
        contacts = []
        travel = 0.0
        previous_length = abs(spline.points[0].weight)
        first = True

        for point in spline.points:
            length = abs(point.weight)
            travel += abs(length - previous_length)
            previous_length = length
            if first and travel > self.offset:
                contacts.append(point)
                travel = 0.0
                first = False
            if travel > self.distance:
                contacts.append(point)
                travel %= self.distance

        return contacts

    def draw_sprue(self, context, contact, contact_radius, mass_center):
        data_curve = bpy.data.curves.new(name='Sprue', type='CURVE')
        sprue = object_data_add(context, data_curve)

        sprue.data.dimensions = '3D'
        sprue.data.resolution_u = 32
        sprue.data.use_path = True
        sprue.data.fill_mode = 'FULL'

        sprue['spiramir_sprues'] = True

        spline = sprue.data.splines.new(type='BEZIER')
        spline.bezier_points.add(1)
        points = spline.bezier_points

        points[0].co = contact
        points[0].radius = contact_radius
        points[0].handle_right = contact + Vector((0, 0, self.curvature * self.height))
        points[0].handle_right_type = 'FREE'
        points[0].handle_left = contact
        points[0].handle_left_type = 'FREE'

        top = mass_center + Vector((0, 0, self.height))
        points[1].co = top
        points[1].radius = self.radius
        points[1].handle_left = top - Vector((0, 0, self.curvature * self.height))
        points[1].handle_left_type = 'FREE'
        points[1].handle_right = top
        points[1].handle_right_type = 'FREE'

        return sprue

    def execute(self, context):
        contacts = []
        mass_center = Vector((0, 0, 0))

        for o in context.view_layer.objects:
            if o.select_get() and 'spiramir' in o and o['spiramir']:
                for spline in o.data.splines:
                    for point in self.get_contact_points_for_spline(spline):
                        contact_point = o.matrix_world @ point.co.to_3d()
                        contacts.append((contact_point, point.radius))
                        mass_center += contact_point

        if contacts:
            mass_center /= len(contacts)

        for contact_point, contact_radius in contacts:
            self.draw_sprue(context, contact_point,
                            contact_radius, mass_center)

        return {'FINISHED'}
