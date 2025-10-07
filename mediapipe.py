import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

def provide_face_point(face_frame):
    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(face_frame)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = face_frame.shape
                points = [(int(landmark.x * w), int(landmark.y * h)) for landmark in face_landmarks.landmark]
                return np.array(points, dtype=np.int32)
    return None


def blur(face_frame, point_array, kernel_size, sigma):
    mask = np.zeros(face_frame.shape[:2], dtype=np.uint8)
    hull = cv2.convexHull(point_array)
    cv2.fillConvexPoly(mask, hull, 255)
    blurred = cv2.GaussianBlur(face_frame, (kernel_size, kernel_size), sigma)
    face_blurred = np.where(mask[..., None] == 255, blurred, face_frame)
    return face_blurred


def overlay_image(face_frame, overlay_img, point_array, scale=1.0):

    mask = np.zeros(face_frame.shape[:2], dtype=np.uint8)
    hull = cv2.convexHull(point_array)
    cv2.fillConvexPoly(mask, hull, 255)

    x, y, w, h = cv2.boundingRect(hull)

    new_w = int(w * scale)
    new_h = int(h * scale)
    overlay_resized = cv2.resize(overlay_img, (new_w, new_h))

    x_center = x + w // 2
    y_center = y + h // 2
    x1 = max(0, x_center - new_w // 2)
    y1 = max(0, y_center - new_h // 2)
    x2 = min(face_frame.shape[1], x1 + new_w)
    y2 = min(face_frame.shape[0], y1 + new_h)

    overlay_resized = overlay_resized[:y2 - y1, :x2 - x1]


    if overlay_resized.shape[2] == 4:  # has alpha
        b, g, r, a = cv2.split(overlay_resized)
        overlay_rgb = cv2.merge([b, g, r])
        alpha_mask = a / 255.0
    else:
        overlay_rgb = overlay_resized
        alpha_mask = np.ones(overlay_resized.shape[:2], dtype=float)

    roi = face_frame[y1:y2, x1:x2]

    for c in range(3):
        roi[:, :, c] = (
            overlay_rgb[:, :, c] * alpha_mask +
            roi[:, :, c] * (1 - alpha_mask)
        )

   #new_pixel = overlay_pixel * alpha + background_pixel * (1 - alpha)

    face_frame[y1:y2, x1:x2] = roi

    return face_frame

def apply_mode(mode, face_frame, point_array, kernel_size, sigma, overlay_img=None, scale=None):
    if mode == "blur":
        return blur(face_frame, point_array, kernel_size, sigma)
    elif mode == "overlay" and overlay_img is not None:
        return overlay_image(face_frame, overlay_img, point_array, scale)
    else:
        return face_frame


def main(media_path, output_path, mode="blur", kernel_size=51, sigma=3, overlay_img=None, scale=None):
  #image case
    if media_path.lower().endswith((".jpeg", ".jpg", ".png")):
        face_frame = cv2.imread(media_path)
        point_array = provide_face_point(face_frame)
        if point_array is not None:
            result = apply_mode(mode, face_frame, point_array, kernel_size, sigma, overlay_img, scale)
            cv2.imwrite(output_path, result)
        else:
            print("No face detected.")
        return

    # VIDEO case
    cap = cv2.VideoCapture(media_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        point_array = provide_face_point(frame)
        if point_array is not None:
            processed = apply_mode(mode, frame, point_array, kernel_size, sigma, overlay_img, scale)
        else:
            processed = frame

        out.write(processed)

    cap.release()
    out.release()




overlay = cv2.imread("/content/freepik_br_8eeb83e4-7cfb-4b1a-a4a3-a8ee46f8de53.png", cv2.IMREAD_UNCHANGED)
main("/content/8136218-hd_1080_1920_25fps.mp4", "output.mp4", mode="blur", kernel_size = 3 , sigma = 1)

#main("/content/9698787-uhd_3840_2160_25fps.jpg", "output.jpg", mode="overlay", overlay_img=overlay, scale=1.0)
