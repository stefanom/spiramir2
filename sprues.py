import bpy
import bmesh
from bpy_extras.object_utils import object_data_add

from mathutils import Vector


class CURVE_OT_sprues(bpy.types.Operator):
    bl_idname = "curve.sprues"
    bl_label = "Sprues"
    bl_description = "Create sprues"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Spiramir"
    bl_options = {'REGISTER', 'UNDO'}

    height: bpy.props.FloatProperty(
        default=-20.0,
        min=-1000.0, max=1000.0,
        description="Distance of the sprue origin (on the Z axis)"
    )
    radius: bpy.props.FloatProperty(
        default=0.8,
        min=0.00001, max=1000.0,
        description="Radius of the sprue"
    )
    curvature: bpy.props.FloatProperty(
        default=0.4,
        min=0.0, max=1.0,
        description="Curvature of the sprue"
    )

    def draw(self, context):
        layout = self.layout
        col = layout.column_flow(align=True)
        col.label(text="Sprues Parameters:")
        col.prop(self, "height", text="Height")
        col.prop(self, "radius", text="Radius")
        col.prop(self, "curvature", text="Curvature")

    def draw_sprue(self, context, origin, contact):
        curve = bpy.data.curves.new(name='Sprue', type='CURVE')
        sprue = object_data_add(context, curve)

        sprue.data.dimensions = '3D'
        sprue.data.resolution_u = 64
        sprue.data.use_path = True
        sprue.data.fill_mode = 'FULL'
        sprue.data.bevel_depth = self.radius

        sprue['sprue'] = True

        spline = sprue.data.splines.new(type='BEZIER')
        spline.bezier_points.add(1)
        points = spline.bezier_points

        points[0].co = contact
        points[0].radius = self.radius
        points[0].handle_right = contact + Vector((0, 0, self.curvature * self.height))
        points[0].handle_right_type = 'FREE'
        points[0].handle_left = contact
        points[0].handle_left_type = 'FREE'

        points[1].co = origin
        points[1].radius = self.radius
        points[1].handle_left = origin - Vector((0, 0, self.curvature * self.height))
        points[1].handle_left_type = 'FREE'
        points[1].handle_right = origin
        points[1].handle_right_type = 'FREE'

        return sprue
   
    def execute(self, context):

        # Check if there is an active object and it is selected
        if context.active_object is None or not context.active_object.select_get():
            self.report({'ERROR'}, "No object is selected. Please select an object.")
            return {'CANCELLED'}

        obj = context.active_object
        matrix_world = obj.matrix_world

        # Check if we are in edit mode
        if obj.mode != 'EDIT':
            self.report({'ERROR'}, "Please switch to EDIT mode and select some faces.")
            return {'CANCELLED'}

        mesh = bmesh.from_edit_mesh(obj.data)
        selected_faces = [f for f in mesh.faces if f.select]

        # Ensure at least one face is selected
        if len(selected_faces) == 0:
            self.report({'ERROR'}, "No face is selected. Please select at least one face.")
            return {'FINISHED'}

        # Get the centroids of selected faces as contact points
        contacts = []
        for face in selected_faces:
            face_centroid = face.calc_center_median()
            global_face_centroid = matrix_world @ face_centroid
            contacts.append(global_face_centroid)

        # Calculate the average position of all contact points
        origin = Vector((0, 0, 0))
        for contact in contacts:
            origin += contact
        origin /= len(contacts)

        # Offset the height of the centroid (this is our injection point)
        origin.z += self.height

        # Draw the sprues
        for contact in contacts:
            sprue = self.draw_sprue(context, origin, contact)
            # Link the sprue to the object and make sure they are in the same collection
            #sprue.parent = obj
            #obj.users_collection[0].objects.link(sprue)

        context.view_layer.update()

        return {'FINISHED'}
