from rendering.portal import PortalRenderer
from rendering.hand_mesh import draw_hand_mesh
from rendering.hud import HUD
from core.state import FrameContext

class Compositor:
    def __init__(self, h, w, device_name):
        self.portal = PortalRenderer()
        self.hud = HUD(h, w, device_name)

    def tick(self):
        self.hud.tick()

    def render(self, frame, ctx: FrameContext, alpha: float, is_active: bool, tl, br, invisible: bool):
        # 1. Background/Invisibility applied in effect, so frame is already updated with it.
        # 2. Portal visual elements
        self.portal.render(frame, ctx, alpha, is_active, tl, br, invisible)
        
        # 3. Hand Mesh
        if ctx.all_points:
            draw_hand_mesh(frame, ctx.all_points)
            
        # 4. HUD
        self.hud.draw(frame, alpha, is_active, ctx.hand_count, ctx.fingers_touching)
        
        return frame
