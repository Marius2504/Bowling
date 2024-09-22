import cv2
import numpy as np
import os
from yolov5 import detect
import torch
import cv2

floor_coords = (1, 600, 1275, 713)

def Task1():
    for index in range(1,26):
        name = str(index) if index > 9 else "0" + str(index)
        output_file = open(f"submission_files/Dumitrescu_Marius_Cristian_407/Task1/{name}_predicted.txt", 'w')

        img_path = os.path.join("Task1",name +'.jpg')
        image = cv2.imread(img_path)

        query_file = os.path.join("Task1",name + "_query.txt")
        lines = []
        if os.path.exists(query_file):
            with open(query_file, 'r') as f:
                content = f.readlines()
                n = int(content[0])
                for i in range(n):
                    lines.append(int(content[i + 1]))

        output_file.write(f"{n}\n")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        height, width = gray.shape

        x1, y1, x2, y2 = floor_coords
        floor_roi = image[y1:y2, x1:x2]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Încarcă modelele pentru fiecare pistă (schimbă cu căile corecte)
        floor_model_1 = cv2.imread('Task1/full-configuration-templates/lane1.jpg')[y1:y2, x1:x2]
        floor_model_2 = cv2.imread('Task1/full-configuration-templates/lane2.jpg')[y1:y2, x1:x2]
        floor_model_3 = cv2.imread('Task1/full-configuration-templates/lane3.jpg')[y1:y2, x1:x2]
        floor_model_4 = cv2.imread('Task1/full-configuration-templates/lane4.jpg')[y1:y2, x1:x2]

        # Funcție pentru compararea ROI-ului cu modelele de referință
        def compare_floor(roi, model):
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            model_hsv = cv2.cvtColor(model, cv2.COLOR_BGR2HSV)

            # Calculam histograma pentru H, S
            hist_roi = cv2.calcHist([roi_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            hist_model = cv2.calcHist([model_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])

            cv2.normalize(hist_roi, hist_roi, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist_model, hist_model, 0, 1, cv2.NORM_MINMAX)

            similarity = cv2.compareHist(hist_roi, hist_model, cv2.HISTCMP_CORREL)

            return similarity


        similarity_1 = compare_floor(floor_roi, floor_model_1)
        similarity_2 = compare_floor(floor_roi, floor_model_2)
        similarity_3 = compare_floor(floor_roi, floor_model_3)
        similarity_4 = compare_floor(floor_roi, floor_model_4)

        similarities = [similarity_1, similarity_2, similarity_3, similarity_4]
        pista_index = np.argmax(similarities) + 1

        ppp = "Task1/full-configuration-templates/lane"+str(pista_index)+".jpg"
        reference = cv2.imread(ppp)

        if pista_index == 1:
            coords = [[575,105,688,221],
            [454,75,550,200],
            [706,73,785,189],
            [365,44,443,161],
            [591,39,665,121],
            [804,52,880,148],
            [279,34,352,137],
            [487,21,552,93],
            [687,22,755,92],
            [898,33,952,136]]
        elif pista_index == 2:
            coords = [[604, 112, 690, 190],
            [503, 79, 589, 164],
            [790, 75, 717, 165],
            [512, 57, 430, 138],
            [605, 52, 695, 121],
            [788, 55, 864, 126],
            [368, 49, 439, 120],
            [535, 47, 594, 114],
            [758, 49, 683, 116],
            [841, 38, 929, 109]]
        elif pista_index == 3:
            coords = [[595,216,708,383],
            [487,125,569,260],
            [754,132,831,265],
            [412,74,479,181],
            [627,69,693,182],
            [847,66,918,187],
            [351,27,412,129],
            [546,30,600,127],
            [728,26,787,125],
            [921,25,981,129]]
        elif pista_index == 4:
            coords = [[572,204,677,363],
            [455,129,540,252],
            [713,123,791,257],
            [380,79,446,174],
            [590,74,659,183],
            [803,75,867,179],
            [328,31,383,138],
            [507,28,568,131],
            [689,33,743,137],
            [867,31,923,127]]

        def validate_coords(x1, y1, x2, y2, img_width, img_height):

            x1 = max(0, min(x1, img_width - 1))
            y1 = max(0, min(y1, img_height - 1))
            x2 = max(0, min(x2, img_width - 1))
            y2 = max(0, min(y2, img_height - 1))

            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1

            return x1, y1, x2, y2

        for i, (x1, y1, x2, y2) in enumerate(coords):
            if i+1 in lines:
                x1, y1, x2, y2 = validate_coords(x1, y1, x2, y2, width, height)

                image_roi = image[y1:y2, x1:x2]
                ref_roi = reference[y1:y2, x1:x2]
                similarity_score = compare_floor(ref_roi, image_roi)
                simi = "1" if similarity_score > 0.3 else "0"

                roi = binary[y1:y2, x1:x2]
                roi2 = image[y1:y2, x1:x2]

                color = (0, 255, 0) if similarity_score > 0.3 else (0, 0, 255)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                output_file.write(f"{i + 1} {simi}\n")


def Task2():
    model_path = 'yolov5/runs/train/exp5/weights/best.pt'
    for index in range(1,16):
        name = str(index) if index > 9 else "0" + str(index)
        test_video_path = f'Task2/{name}.mp4'
        output_file = open(f"submission_files/Dumitrescu_Marius_Cristian_407/Task2/{name}_predicted.txt", 'w')

        last_frame = 0
        goodFrame = True
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)  # Încarcă modelul antrenat

        cap = cv2.VideoCapture(test_video_path)

        frame_count = 0
        listt = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)

            x_min = 0
            y_min = 0
            x_max = 0
            y_max = 0
            if frame_count > 11:
                for i in range(1,11):
                    x_min = x_min + listt[frame_count - i][1] - listt[frame_count - i-1][1]
                    y_min = y_min + listt[frame_count - i][2] - listt[frame_count - i-1][2]
                    x_max = x_max + listt[frame_count - i][3] - listt[frame_count - i-1][3]
                    y_max = y_max + listt[frame_count - i][4] - listt[frame_count - i-1][4]

                x_min = x_min/10 + listt[frame_count - 1][1]
                y_min = y_min/10 + listt[frame_count - 1][2]
                x_max = x_max/10 + listt[frame_count - 1][3]
                y_max = y_max/10 + listt[frame_count - 1][4]

            if len(results.xyxy[0]) == 0:
                if goodFrame == True:
                    goodFrame = False
                    last_frame = frame_count
                listt.append([frame_count, int(x_min), int(y_min), int(x_max), int(y_max)])
            else:
                goodFrame = True
                for det in results.xyxy[0]:
                    x_min, y_min, x_max, y_max, conf, cls = det.tolist()
                listt.append([frame_count, int(x_min), int(y_min), int(x_max), int(y_max)])
            annotated_frame = results.render()[0]

            # Afișează frame-ul cu chenarele desenate
            cv2.imshow('Detected Frame', annotated_frame)

            # Închide fereastra la apăsarea tastei 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_count = frame_count+ 1

        first_index = -1
        index = 0
        while first_index == -1 and index < listt[-1][0]:
            if listt[index][1] !=0 and listt[index][2] !=0 and listt[index][3] !=0 and listt[index][4] !=0:
                first_index = index
            else:
                index = index + 1


        first_index = first_index - 1
        while first_index > -1:
            x_min = 0
            y_min = 0
            x_max = 0
            y_max = 0
            for i in range(1, 11):
                x_min = x_min + listt[first_index + i][1] - listt[first_index + i + 1][1]
                y_min = y_min + listt[first_index + i][2] - listt[first_index + i + 1][2]
                x_max = x_max + listt[first_index + i][3] - listt[first_index + i + 1][3]
                y_max = y_max + listt[first_index + i][4] - listt[first_index + i + 1][4]

            x_min = x_min / 10 + listt[first_index + 1][1]
            y_min = y_min / 10 + listt[first_index + 1][2]
            x_max = x_max / 10 + listt[first_index + 1][3]
            y_max = y_max / 10 + listt[first_index + 1][4]
            listt[first_index] = [first_index, int(x_min), int(y_min), int(x_max), int(y_max)]

            first_index = first_index - 1

        output_file.write(f"{len(listt)} -1 -1 -1 -1\n")
        for frame in listt[0:last_frame-1]:
            output_file.write(f"{frame[0]} {frame[1]} {frame[2]} {frame[3]} {frame[4]}\n")
        output_file.close()
        cap.release()
        cv2.destroyAllWindows()


def Task3():
    model_path = 'yoloTask3/yolov5/runs/train/exp7/weights/best.pt'
    for index in range(1,16):
        name = str(index) if index > 9 else "0" + str(index)
        test_video_path = f'Task3/{name}.mp4'

        output_file = open(f"submission_files/Dumitrescu_Marius_Cristian_407/Task3/{name}_predicted.txt", 'w')
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)  # Încarcă modelul antrenat

        cap = cv2.VideoCapture(test_video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_count = 0

        x_min_threshold = 150
        x_max_threshold = 1000

        bowling_pins_start = 0
        bowling_pins_final = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count >= 5:
                bowling_pins = 0
                results = model(frame)

                filtered_annotations = []

                for det in results.xyxy[0]:
                    x_min, y_min, x_max, y_max, conf, cls = det.tolist()

                    if int(cls) == 1 and x_min > x_min_threshold and x_max < x_max_threshold:
                        bowling_pins = bowling_pins + 1
                        filtered_annotations.append(det)
                if bowling_pins_start == 0:
                    if bowling_pins > 6:
                        bowling_pins = 10
                    bowling_pins_start = bowling_pins
                else:
                    bowling_pins_final = bowling_pins
            frame_count += 1

        output_file.write(f"{bowling_pins_start}\n")
        output_file.write(f"{bowling_pins_final}\n")
        output_file.close()
        cap.release()
        cv2.destroyAllWindows()

#Task1()
Task2()
#Task3()



