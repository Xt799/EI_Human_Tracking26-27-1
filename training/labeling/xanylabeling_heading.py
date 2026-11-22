"""Start X-AnyLabeling with a live heading arrow drawn from the wheel points (run by label.ps1 with X-AnyLabeling's Python).

X-AnyLabeling itself is not changed: its canvas drawing is wrapped here, before the app starts, to draw one arrow over
the image from the wheel_1..wheel_4 points of the current image. It is redrawn on every change, so it follows the points
while they are dragged. The arrow is the heading of training/scripts/evaluate_pose.py, computed in the image:
- all four wheels: perpendicular to the line from the middle of the right side (wheel_1, wheel_2) to the middle of the
  left side (wheel_3, wheel_4), pointing to the front pair (wheel_1, wheel_3) rather than the rear pair (wheel_2, wheel_4);
- one side only: that side's rear -> front wheel (wheel_2 -> wheel_1 or wheel_4 -> wheel_3).
"""
import math
import sys

from PyQt6 import QtCore, QtGui

from anylabeling import app
from anylabeling.views.labeling.widgets import canvas

WHEELS = ("wheel_1", "wheel_2", "wheel_3", "wheel_4")
COLOUR = QtGui.QColor(0, 255, 0)
OUTLINE = QtGui.QColor(0, 0, 0)


def heading(p):
    """(centre, unit direction, robot size) in image px from {wheel index: (x, y)}, or None."""
    if len(p) == 4:
        sx = (p[2][0] + p[3][0] - p[0][0] - p[1][0]) / 2
        sy = (p[2][1] + p[3][1] - p[0][1] - p[1][1]) / 2
        vx, vy = sy, -sx
        fx = (p[0][0] + p[2][0] - p[1][0] - p[3][0]) / 2
        fy = (p[0][1] + p[2][1] - p[1][1] - p[3][1]) / 2
        if vx * fx + vy * fy < 0:
            vx, vy = -vx, -vy
        cx, cy = sum(q[0] for q in p.values()) / 4, sum(q[1] for q in p.values()) / 4
        size = math.hypot(sx, sy)
    else:
        side = next(((f, r) for f, r in ((0, 1), (2, 3)) if f in p and r in p), None)
        if side is None:
            return None
        f, r = p[side[0]], p[side[1]]
        vx, vy = f[0] - r[0], f[1] - r[1]
        cx, cy = (f[0] + r[0]) / 2, (f[1] + r[1]) / 2
        size = 2 * math.hypot(vx, vy)
    n = math.hypot(vx, vy)
    if n < 1e-6:
        return None
    return (cx, cy), (vx / n, vy / n), size


original_paint = canvas.Canvas.paintEvent


def paint_with_heading(self, event):
    original_paint(self, event)
    if self.pixmap is None or self.pixmap.width() == 0:
        return
    pts = {}
    for s in self.shapes:
        if s.label in WHEELS and s.points and self.is_visible(s):
            pts[WHEELS.index(s.label)] = (s.points[0].x(), s.points[0].y())
    h = heading(pts)
    if h is None:
        return
    (cx, cy), (ux, uy), size = h
    length = max(size, 30.0) * 1.3
    tail = QtCore.QPointF(cx - ux * length * 0.4, cy - uy * length * 0.4)
    tip = QtCore.QPointF(cx + ux * length, cy + uy * length)
    head = length * 0.3
    wings = [QtCore.QPointF(tip.x() - head * (ux * math.cos(a) - uy * math.sin(a)),
                            tip.y() - head * (uy * math.cos(a) + ux * math.sin(a))) for a in (0.45, -0.45)]
    p = QtGui.QPainter(self)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    p.scale(self.scale, self.scale)
    p.translate(self.offset_to_center())
    for colour, width in ((OUTLINE, 5.0), (COLOUR, 2.5)):          # width in screen px, whatever the zoom
        pen = QtGui.QPen(colour, width / self.scale)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.drawLine(tail, tip)
        for w in wings:
            p.drawLine(tip, w)
    p.end()


canvas.Canvas.paintEvent = paint_with_heading

if __name__ == "__main__":
    sys.exit(app.main())
