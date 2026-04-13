import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
canvas = None
prev_x, prev_y = 0, 0
drawing = True
save_count = 0
finger_count = 0

def get_finger_count(hand_landmarks):
    tips = [8, 12, 16, 20]
    count = 0
    for tip in tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            count += 1
    return count

def get_color(finger_count):
    colors = {
        1: (255, 0, 0),
        2: (0, 0, 255),
        3: (0, 255, 0),
    }
    return colors.get(finger_count, (255, 255, 255))

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        if canvas is None:
            canvas = np.zeros_like(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                idx = hand.landmark[8]
                h, w, _ = frame.shape
                x, y = int(idx.x * w), int(idx.y * h)

                finger_count = get_finger_count(hand)
                color = get_color(finger_count)

                if drawing:
                    if prev_x == 0 and prev_y == 0:
                        prev_x, prev_y = x, y
                    cv2.line(canvas, (prev_x, prev_y), (x, y), color, 5)

                prev_x, prev_y = x, y
        else:
            prev_x, prev_y = 0, 0
            finger_count = 0

        combined = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)

        cv2.putText(combined, "Hand of Aspa", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)
        durum = "FIRCA ACIQ" if drawing else "FIRCA QAPALI"
        cv2.putText(combined, durum, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(combined, f"Barmaq sayi: {finger_count}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Hand of Aspa", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas = np.zeros_like(frame)
        elif key == ord('l'):
            drawing = not drawing
            prev_x, prev_y = 0, 0
        elif key == ord('s'):
            filename = f"cizim_{save_count}.png"
            cv2.imwrite(filename, canvas)
            save_count += 1
            print(f"Kaydedildi: {filename}")    

cap.release()
cv2.destroyAllWindows()