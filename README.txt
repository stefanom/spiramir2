
 # SpiraMir 2
 
 This folder contains a Blender add-on to draw or fill curves with recursive
 non-interative logarithmic spirals.
 
## Attic

    # def get_contact_points(self, curves, steps):
    #     contacts = []
    #     for curve in curves:
    #         cursor, constraint = utils.get_constrainted_empty(curve, 0.0)
    #         for i in range(steps):
    #             constraint.offset_factor = i / steps
    #             # Need to update or changing the constrains
    #             # won't change the position.
    #             utils.update(cursor, constraint)
    #             # Need to make a copy of the position or it will
    #             # change when the constraint changes.
    #             yield Vector(cursor.matrix_world.translation)
    #         remove(cursor)
