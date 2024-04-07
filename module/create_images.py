import os
import cv2

# дерристория в которую будем сохранять дата сет
DATA_DIR = './data'
# список классов которое нужно собрать
classes_list = ['', 'A', 'B'] # , 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'ILY'
# размер нашего дата сета
dataset_size = 200

cap = cv2.VideoCapture(0)

for class_item in classes_list:
    if not os.path.exists(os.path.join(DATA_DIR, class_item)):
        os.makedirs(os.path.join(DATA_DIR, class_item))

    print(f'Собираем дата сет для класса {class_item}')

    done = False
    while not done:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        cv2.putText(frame, f"Press \"Q\" to create data :) {class_item}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(25)

        if key == ord('q'):
            done = True
            break

    if done:
        counter = 0
        while counter < dataset_size:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame. Exiting...")
                break

            cv2.imshow('frame', frame)
            cv2.waitKey(25)
            cv2.imwrite(os.path.join(DATA_DIR, class_item, f'{counter}.jpg'), frame)

            counter += 1

cap.release()
cv2.destroyAllWindows()