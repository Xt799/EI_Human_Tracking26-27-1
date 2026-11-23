"""Start X-AnyLabeling with a live heading arrow drawn from the front and rear points (run by label.ps1 with
X-AnyLabeling's Python).

X-AnyLabeling itself is not changed: its canvas drawing is wrapped here, before the app starts, to draw one arrow over
the image from the current image's rear point through its front point. It is redrawn on every change, so it follows the
points while they are dragged.
"""
import math
import sys

from PyQt6 import QtCore, QtGui

from anylabeling import app
from anylabeling.views.labeling.widgets import canvas

COLOUR = QtGui.QColor(0, 255, 0)
OUTLINE = QtGui.QColor(0, 0, 0)

original_paint = canvas.Canvas.paintEvent


def paint_with_heading(self, event):
    original_paint(self, event)
    if self.pixmap is None or self.pixmap.width() == 0:
        return
    pts = {s.label: s.points[0] for s in self.shapes if s.label in ("front", "rear") and s.points and self.is_visible(s)}
    if len(pts) < 2:
        return
    f, r = pts["front"], pts["rear"]
    dx, dy = f.x() - r.x(), f.y() - r.y()
    n = math.hypot(dx, dy)
    if n < 1e-6:
        return
    ux, uy = dx / n, dy / n
    tip = QtCore.QPointF(f.x() + ux * n * 0.6, f.y() + uy * n * 0.6)        # beyond the front, so the head is clear
    head = max(n * 0.35, 6.0)
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
        p.drawLine(r, tip)
        for w in wings:
            p.drawLine(tip, w)
    p.end()


canvas.Canvas.paintEvent = paint_with_heading

if __name__ == "__main__":
    sys.exit(app.main())
