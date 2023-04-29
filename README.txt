
# SpiraMir 2
 
This folder contains a Blender add-on to draw or fill curves with recursive
non-interative logarithmic spirals.

# How to install

* Configure Blender's "scripts" directory to be one of your choosing
* add an "addons" folder in that
* copy the "spiramir2" forlder in "addons"
* go to the Blender internal preferences and install the "spiramir2" addon (in the community tab)

# How to use

* in Object mode, select "Add > Curve > Spiramir"
* spiramir configurations are in the bottom left part of the screen

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
