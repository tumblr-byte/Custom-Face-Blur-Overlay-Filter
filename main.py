import os
import cv2
import face_recognition


# Globals for known faces
known_faces = []
known_encodings = []


# Load and encode a known face
def know_face(know_image_path):
    if os.path.exists(know_image_path):
        image = face_recognition.load_image_file(know_image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_faces.append(image)
            known_encodings.append(encodings[0])
        else:
            print("No face found in the image:", know_image_path)
    else:
        print("No such path found:", know_image_path)


# Overlay function 
def overlay_image(frame, overlay_img, face_location, scale):
    top, right, bottom, left = face_location
    face_h = bottom - top
    face_w = right - left

    # Resize overlay relative to face size
    overlay_h = int(face_h * scale)
    overlay_w = int(face_w * scale)
    overlay_resized = cv2.resize(overlay_img, (overlay_w, overlay_h))
   

    # Shift the overlay image based on your preference and the chosen scale
    y_offset = top + face_h // 2 - overlay_h // 2 - int(0.2 * face_h) 
    x_offset = left + face_w // 2 - overlay_w // 2

    # Clip to frame boundaries
    y1 = max(y_offset, 0)
    x1 = max(x_offset, 0)
    y2 = min(y1 + overlay_h, frame.shape[0])
    x2 = min(x1 + overlay_w, frame.shape[1])

    overlay_cropped = overlay_resized[0:(y2 - y1), 0:(x2 - x1)]

    # If overlay has transparency (alpha channel), blend it with the frame (optional)
    if overlay_cropped.shape[2] == 4:
        b, g, r, a = cv2.split(overlay_cropped)
        alpha = a / 255.0
        alpha_inv = 1.0 - alpha
        for c in range(3):
            #The alpha blending formula = output_pixel = overlay_pixel × α + frame_pixel × (1−α)
            frame[y1:y2, x1:x2, c] = (
                overlay_cropped[:, :, c] * alpha +
                frame[y1:y2, x1:x2, c] * alpha_inv
            ).astype('uint8')
    else:
        frame[y1:y2, x1:x2] = overlay_cropped

    return frame


def apply_mode(mode, frame, face_location, kernel_size=3, sigma=3, overlay_img=None, scale=1.0):
    top, right, bottom, left = face_location
    face_frame = frame[top:bottom, left:right]

    if mode == "blur":
        blurred = cv2.GaussianBlur(face_frame, (kernel_size, kernel_size), sigma)
        frame[top:bottom, left:right] = blurred
    elif mode == "overlay" and overlay_img is not None:
        frame = overlay_image(frame, overlay_img, face_location, scale)
    return frame


# Detect faces and check known/unknown
def unknown_face(frame):
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame)
    results = []

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.65)
        is_known = any(matches)
        results.append(((top, right, bottom, left), is_known))
    return results


# Draw labels on faces
def label_face(frame, face_location, is_known):
    top, right, bottom, left = face_location
    color = (0, 255, 0) if is_known else (0, 0, 255)
    label = "known" if is_known else "unknown"
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    cv2.putText(frame, label, (left + 5, top + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def main():
    know_face("img.jpeg")  # Load known face
    overlay_img = cv2.imread("2.png", cv2.IMREAD_UNCHANGED)
    if overlay_img is None:
        print("Overlay image not found!")
        return
   
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    target_label = "known" # or (unknown)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = unknown_face(frame)

        for face_location, is_known in faces:
            # Decide whether to overlay
            if (is_known and target_label == "known") or (not is_known and target_label == "unknown"):
                frame = apply_mode("blur", frame, face_location, kernel_size =5 , sigma= 1)
                #frame = apply_mode("overlay", frame, face_location, overlay_img=overlay_img, scale=1.0)
            label_face(frame, face_location, is_known)

        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

main()