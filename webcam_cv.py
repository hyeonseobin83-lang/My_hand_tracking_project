import cv2
import mediapipe as mp
import time

NAME_TAG = "BinHyeonseo(20221782)"   # ← 여기만 본인 이름/학번으로 수정
DURATION = 20
OUT_PATH = "result.mp4"

mp_face = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 20.0
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(OUT_PATH, fourcc, fps, (w, h))

face_det = mp_face.FaceDetection(min_detection_confidence=0.6)
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6)

start = time.time()
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    fres = face_det.process(rgb)
    if fres.detections:
        for d in fres.detections:
            bb = d.location_data.relative_bounding_box
            x, y = int(bb.xmin * w), int(bb.ymin * h)
            bw, bh = int(bb.width * w), int(bb.height * h)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(frame, f"Face {d.score[0]:.2f}", (x, max(y - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    hres = hands.process(rgb)
    if hres.multi_hand_landmarks:
        for hlm in hres.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS)

    cv2.rectangle(frame, (10, 10), (10 + 14 * len(NAME_TAG), 50), (0, 0, 0), -1)
    cv2.putText(frame, NAME_TAG, (18, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    elapsed = time.time() - start
    cv2.putText(frame, f"REC {elapsed:0.1f}/{DURATION}s",
                (w - 230, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    writer.write(frame)
    cv2.imshow("Webcam CV", frame)

    if elapsed >= DURATION or cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
print("Saved:", OUT_PATH)
