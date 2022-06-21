import os

import numpy as np
import cv2 as cv
from mtcnn import MTCNN
import csv


def calculate_iou(right_face, detected_face):
    x_1, y_1, x_2, y_2 = right_face
    x_3, y_3, x_4, y_4 = detected_face
    area_inter = abs(min(x_2, x_4) - max(x_1, x_3)) * abs(min(y_2, y_4) - max(y_1, y_3))
    area_union = abs(x_2 - x_1) * abs(y_2 - y_1) + abs(x_4 - x_3) * abs(y_4 - y_3) - area_inter
    return area_inter / area_union


def evaluate_faces(frame, tp, fp):
    fn = 1 - tp
    precision = 0
    recall = 0
    if tp + fp != 0:
        precision = tp / (tp + fp)
    if tp + fn != 0:
        recall = tp / (tp + fn)
    evaluation = "TP: {}, FP: {} FN: {}".format(tp, fp, fn)
    cv.rectangle(frame, (0, 0), (0 + frame.shape[1], 30), (0, 0, 0), -1)
    cv.putText(frame, evaluation, org=(0, 10), fontFace=cv.FONT_HERSHEY_DUPLEX,
               fontScale=0.3, color=(255, 255, 255), thickness=1)

    precision_recall = "precision: {}, recall: {}".format(precision, recall)
    cv.putText(frame, precision_recall, org=(0, 20), fontFace=cv.FONT_HERSHEY_DUPLEX,
               fontScale=0.3, color=(255, 255, 255), thickness=1)
    return frame, [tp, fp, fn, precision, recall]


def show_detection(frame, box, detected_faces):
    best_face_rectangle = None
    iou_min = 0.5
    best_face_iou = 0
    tp = 0

    for (column, row, width, height) in detected_faces:
        cv.rectangle(frame, (column, row), (column + width, row + height), (0, 0, 255), 2)
        face_iou = calculate_iou([int(box[0][0]), int(box[0][1]), int(box[3][0]), int(box[3][1])],
                                 [column, row, column + width, row + height])
        if iou_min <= face_iou > best_face_iou:
            best_face_iou = face_iou
            best_face_rectangle = (column, row, width, height)

    if best_face_rectangle is not None:
        tp = 1
        (column, row, width, height) = best_face_rectangle
        cv.rectangle(frame, (column, row), (column + width, row + height), (0, 255, 0), 2)
    frame, evaluation_data = evaluate_faces(frame, tp, len(detected_faces) - tp)
    return frame, evaluation_data


def make_viola_jones(frame, box):
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    detected_faces = face_cascade.detectMultiScale(frame)
    frame, evaluation_data = show_detection(frame, box, detected_faces)
    return frame, detected_faces, evaluation_data


def make_average_evaluation(average_data, evaluation_data):
    for i in range(len(average_data)):
        average_data[i] += evaluation_data[i]
    return average_data


def write_data(writer, file_name, average_data, frames):
    average_data[:] = [round(x / frames, 4) for x in average_data]
    row = {'name': file_name,
           'TP': average_data[0],
           'FP': average_data[1],
           'FN': average_data[2],
           'precision': average_data[3],
           'recall': average_data[4]
           }

    writer.writerow(row)


def load_save_video(file_path, video, bounding_box, landmarks2d, writer):
    height = video.shape[0]
    width = video.shape[1]
    frames = video.shape[3]

    average_data = [0, 0, 0, 0, 0]
    video_output = cv.VideoWriter(file_path + '.mp4', cv.VideoWriter_fourcc(*'mp4v'), 20, (width, height), True)

    for i in range(frames):
        frame = cv.cvtColor(np.ascontiguousarray(video[:, :, :, i], dtype=np.uint8), cv.COLOR_RGB2BGR)
        box = bounding_box[:, :, i]
        frame, detected_faces, evaluation_data = make_viola_jones(frame, box)

        for landmark in landmarks2d[:, :, i]:
            cv.circle(frame, (int(landmark[0]), int(landmark[1])), 1, (255, 0, 0), 1)
        cv.rectangle(frame, (int(box[0][0]), int(box[0][1])), (int(box[3][0]), int(box[3][1])), (255, 0, 0), 1)

        average_data = make_average_evaluation(average_data, evaluation_data)
        video_output.write(frame)
    video_output.release()
    write_data(writer, file_path[12:], average_data, frames)


if __name__ == '__main__':
    detector = MTCNN()

    path = 'data/videa/'
    files = os.listdir(path)

    header = ['name', 'TP', 'FP', 'FN', 'precision', 'recall']
    csv_path = 'data/saveData_viola_all'

    with open(csv_path + '.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for file in files:
            data = np.load(path + file)
            colorImg_original = data['colorImages']
            boundingBox = data['boundingBox']
            landmarks2D = data['landmarks2D']
            load_save_video('data/output/viola_all/' + file[:-4], colorImg_original, boundingBox,
                            landmarks2D,
                            writer)
