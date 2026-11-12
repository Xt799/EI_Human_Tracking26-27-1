"""Top-down drawing of the room: the tile grid, the origin and one dot per person and object, with an arrow for the way a person faces."""
import cv2
import numpy as np

from common import ORIGIN_COLOUR, X_AXIS_COLOUR, Y_AXIS_COLOUR

PX_PER_M = 80
ARROW_M = 0.5        # length of a heading arrow
OUTLINE = (60, 60, 60)
MARGIN_PX = 30
BACKGROUND, GRID, GRID_BOLD, WALL = (245, 245, 245), (215, 215, 215), (175, 175, 175), (80, 80, 80)


class RoomMap:
    """Canvas over the room, y pointing down the screen. The background is drawn once; draw() adds the dots."""

    def __init__(self, room_m, tile_m):
        (self.x0, x1), (self.y0, y1) = room_m
        self.width = int((x1 - self.x0) * PX_PER_M) + 2 * MARGIN_PX
        self.height = int((y1 - self.y0) * PX_PER_M) + 2 * MARGIN_PX
        self.background = np.full((self.height, self.width, 3), BACKGROUND, np.uint8)
        for k in range(int(np.floor(self.x0 / tile_m)), int(np.ceil(x1 / tile_m)) + 1):
            cv2.line(self.background, self.to_px((k * tile_m, self.y0)), self.to_px((k * tile_m, y1)), GRID_BOLD if k % 5 == 0 else GRID, 1)
        for k in range(int(np.floor(self.y0 / tile_m)), int(np.ceil(y1 / tile_m)) + 1):
            cv2.line(self.background, self.to_px((self.x0, k * tile_m)), self.to_px((x1, k * tile_m)), GRID_BOLD if k % 5 == 0 else GRID, 1)
        cv2.line(self.background, self.to_px((0, 0)), self.to_px((3 * tile_m, 0)), X_AXIS_COLOUR, 3)
        cv2.line(self.background, self.to_px((0, 0)), self.to_px((0, 3 * tile_m)), Y_AXIS_COLOUR, 3)
        cv2.rectangle(self.background, self.to_px((self.x0, self.y0)), self.to_px((x1, y1)), WALL, 2)
        cv2.circle(self.background, self.to_px((0, 0)), 6, ORIGIN_COLOUR, -1)
        cv2.putText(self.background, "x", self.to_px((3.5 * tile_m, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, X_AXIS_COLOUR, 1, cv2.LINE_AA)
        cv2.putText(self.background, "y", self.to_px((0, 3.8 * tile_m)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, Y_AXIS_COLOUR, 1, cv2.LINE_AA)

    def to_px(self, xy):
        """Metres on the floor -> pixel on the canvas: x to the right, y down the screen."""
        return int(MARGIN_PX + (xy[0] - self.x0) * PX_PER_M), int(MARGIN_PX + (xy[1] - self.y0) * PX_PER_M)

    def draw(self, marks):
        """marks: (position in metres, colour, text, trail of positions, heading unit vector or None) per thing to show. Returns the canvas."""
        canvas = self.background.copy()
        for xy, colour, text, trail, heading in marks:
            if len(trail) > 1:
                cv2.polylines(canvas, [np.array([self.to_px(q) for q in trail], np.int32)], False, colour, 1, cv2.LINE_AA)
            p = self.to_px(xy)
            if heading is not None:
                tip = self.to_px(np.asarray(xy) + ARROW_M * np.asarray(heading))
                cv2.arrowedLine(canvas, p, tip, OUTLINE, 5, cv2.LINE_AA, tipLength=0.3)
                cv2.arrowedLine(canvas, p, tip, colour, 3, cv2.LINE_AA, tipLength=0.3)
            cv2.circle(canvas, p, 8, colour, -1, cv2.LINE_AA)
            cv2.circle(canvas, p, 8, OUTLINE, 1, cv2.LINE_AA)
            x = p[0] + 11
            if heading is not None and heading[0] > 0.5:   # the arrow points right: the text goes on the left of the dot
                x = p[0] - 11 - cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
            cv2.putText(canvas, text, (x, p[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
        return canvas
