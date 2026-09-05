import cv2

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]

def draw_hand_mesh(frame, all_points):
    if not all_points:
        return frame
    for pts in all_points:
        if len(pts) < 21:
            continue
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (255, 255, 255), 1, cv2.LINE_AA)
        for i, p in enumerate(pts):
            r = 6 if i in (4, 8) else 3
            cv2.circle(frame, p, r, (0, 0, 220), -1, cv2.LINE_AA)
            cv2.circle(frame, p, r, (255, 255, 255), 1, cv2.LINE_AA)
    return frame
