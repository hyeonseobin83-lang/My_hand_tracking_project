import cv2
import mediapipe as mp

# MediaPipe 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# 웹캠 열기
cap = cv2.VideoCapture(0)

print("프로젝트 실행 중... 'q'를 누르면 종료됩니다.")

while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("웹캠을 찾을 수 없습니다.")
        break

    # 좌우 반전 및 BGR을 RGB로 변환
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 손 검출 수행
    results = hands.process(img_rgb)

    # 검출된 손 마디 그리기
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Tracking Project", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
