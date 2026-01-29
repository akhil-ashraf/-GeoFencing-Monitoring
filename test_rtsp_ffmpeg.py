import subprocess
import numpy as np
import cv2

RTSP_URL = "rtsp://admin:Admin%4025123@192.168.31.200:554/Streaming/Channels/301"
WIDTH, HEIGHT = 640, 360

cmd = [
    "ffmpeg",
    "-rtsp_transport", "tcp",
    "-stimeout", "5000000",      # 5 sec timeout
    "-fflags", "nobuffer",
    "-flags", "low_delay",
    "-probesize", "32",
    "-analyzeduration", "0",
    "-i", RTSP_URL,
    "-an",
    "-vf", f"scale={WIDTH}:{HEIGHT}",
    "-pix_fmt", "bgr24",
    "-f", "rawvideo",
    "-"
]


pipe = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,   # silence logs
    bufsize=10**8
)

print("Connecting to camera... waiting for first frame")



while True:
    raw = pipe.stdout.read(WIDTH * HEIGHT * 3)
    if not raw:
        print("No more frames")
        break

    if not hasattr(globals(), "started"):
        print("Stream started")
        started = True

    frame = np.frombuffer(raw, np.uint8).reshape((HEIGHT, WIDTH, 3))
    cv2.imshow("RTSP via FFmpeg", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

pipe.terminate()
cv2.destroyAllWindows()
