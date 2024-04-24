
def plot_epm_roi(polygons: tuple, image):
    fig, ax = plt.subplots()  # New Graph
    ax.set_axis_off()

    open_roi, closed_roi, center_roi = polygons

    open_poly = Polygon(open_roi.hull, alpha=0.2, color='r')
    closed_poly = Polygon(closed_roi.hull, alpha=0.2, color='b')
    center_poly = Polygon(center_roi.hull, alpha=0.3, color=(0, 1, 0))

    patches = PatchCollection([open_poly, closed_poly, center_poly], match_original=True)

    ax.add_collection(patches)
    ax.legend([open_poly, closed_poly, center_poly], ['Open', 'Closed', 'Center'], loc='lower center')
    _ = ax.imshow(image)

