
def plot_epm_roi(polygons: dict, image):
    fig, ax = plt.subplots()  # New Graph
    ax.set_axis_off()

    open_coordinates, closed_coordinates, center_coordinates = get_polygon_coordinates(polygons)

    open_poly = Polygon(open_coordinates, alpha=0.2, color='r')
    closed_poly = Polygon(closed_coordinates, alpha=0.2, color='b')
    center_poly = Polygon(center_coordinates, alpha=0.3, color=(0, 1, 0))

    patches = PatchCollection([open_poly, closed_poly, center_poly], match_original=True)

    ax.add_collection(patches)
    ax.legend([open_poly, closed_poly, center_poly], ['Open', 'Closed', 'Center'], loc='lower center', framealpha=1)
    _ = ax.imshow(image)

    return fig, ax


def get_polygon_coordinates(polygons: dict):
    coordinates = []

    for polygon in polygons['Polygon']:
        coordinates.append(list(polygon.exterior.coords)[:-1])  # Drop last point as it is a duplicate of the first

    return coordinates
