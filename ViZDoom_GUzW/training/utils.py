import cv2


def display_buffers(state):
    depth = state.depth_buffer
    if depth is not None:
        cv2.imshow("ViZDoom Depth Buffer", depth)
        print("depth", depth.shape)

    screen_buffer = state.screen_buffer
    if screen_buffer is not None:
        cv2.imshow("ViZDoom screen_buffer Buffer", screen_buffer)
        print("screen_buffer", screen_buffer.shape)

    labels = state.labels_buffer
    if labels is not None:
        cv2.imshow("ViZDoom Labels Buffer", labels)
        print("labels", labels.shape)

    automap = state.automap_buffer
    if automap is not None:
        cv2.imshow("ViZDoom Map Buffer", automap)
        print("automap", automap.shape)
    sleep_time = 0.028
    cv2.waitKey(int(sleep_time * 1000))