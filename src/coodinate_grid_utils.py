# Convert world (meters) → grid index (pixels)
def world_to_grid(pt, origin, resolution, grid_height):
    xg = int((pt[0] - origin[0]) / resolution)
    yg_unflipped = int((pt[1] - origin[1]) / resolution)
    yg = grid_height - 1 - yg_unflipped  # Flip y to match image coordinates
    return (xg, yg)

# Convert grid index (pixels) → world (meters)
def grid_to_world(pt, origin, resolution, grid_height):
    px, py = pt
    py_flipped = grid_height - 1 - py  # Invert y index to world convention
    x_world = px * resolution + origin[0]
    y_world = py_flipped * resolution + origin[1]
    return (x_world, y_world)