
 SpiraMir 2
 ----------

 This folder contains a Blender add-on to draw or fill curves with recursive
 non-interative logarithmic spirals.
 
 Code Attic
 ----------

# if len(curve.data.splines) == 1:
#     spline = curve.data.splines[0]
#     if spline.type == 'BEZIER':
#         origin, tangent, _ = bezier_multi_seg(spline.bezier_points, self.position, closed_loop=True, matrix=curve.matrix_world)
#     elif spline.type == 'POLY':
#         points = len(spline.points)
#         i = round(self.position * points)
#         origin = curve.matrix_world @ Vector(spline.points[i].co[0:3])
#         tangent = curve.matrix_world @ Vector(spline.points[i + 1 % points].co[0:3]) - origin

#empty.rotation_mode = 'QUATERNION'
#empty.rotation_quaternion = tangent.to_track_quat('X', 'Z')

