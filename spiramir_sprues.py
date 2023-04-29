import bpy
from bpy_extras.object_utils import object_data_add

from mathutils import Vector

from . import utils


class CURVE_OT_spiramir_sprues(bpy.types.Operator):
    bl_idname = "curve.spiramir_sprues"
    bl_label = "Spiramir Sprues"
    bl_description = "Create sprues for a spiramir"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Spiramir"
    bl_options = {'REGISTER', 'UNDO'}

    distance: bpy.props.FloatProperty(
        default=1,
        min=0.00001, max=1000.0,
        description="Distance on the curve between sprues"
    )
    offset: bpy.props.FloatProperty(
        default=0.5,
        min=0.00001, max=1000.0,
        description="Offset to start from the beginning of the curve"
    )
    height: bpy.props.FloatProperty(
        default=10.0,
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

    def draw_sprue(self, context, contact, contact_radius, mass_center):
        curve = bpy.data.curves.new(name='Sprue', type='CURVE')
        sprue = object_data_add(context, curve)

        sprue.data.dimensions = '3D'
        sprue.data.resolution_u = 64
        sprue.data.use_path = True
        sprue.data.fill_mode = 'FULL'
        sprue.data.bevel_depth = self.radius

        sprue['spiramir_sprue'] = True

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

        for curve in utils.get_visible_scene_curves():
            for contact_point, radius in utils.get_equidistant_points(curve, self.offset, self.distance):
                contacts.append((contact_point, radius))
                mass_center += contact_point

        if contacts:
            mass_center /= len(contacts)

        for contact_point, contact_radius in contacts:
            self.draw_sprue(context, contact_point, contact_radius, mass_center)

        return {'FINISHED'}
