# Custom Face Blur & Overlay Filter (Real-Time)

A real-time face filter tool that allows you to **blur faces** or **overlay custom images** using Python.  
This project is designed for experimentation, fun, and learning, but it also demonstrates practical applications like **privacy masking** or **creative effects** for multiple people in real-time.

---

## Features

- Detect multiple faces in real-time using a webcam or video feed.  
- Apply blur or overlay to faces dynamically.  
- Customizable filters for known or unknown people:
  - **Apply effects only to the known person(s).**  
  - **Apply effects to everyone except the known person(s).**  
- Works with `face_recognition` for accurate recognition and filtering.  
- Optional fun overlays like images (with transparency) or masks.

---

## How Known and Unknown Faces Work

- **Uploading a known face is required:** the system needs at least one known face image to compare against.  
- **Filter options:**
  - **Known person filter:** Only the uploaded known face(s) will have the blur or overlay applied. All other faces remain unchanged.  
  - **Unknown person filter:** All faces **except the uploaded known face(s)** will have the effect applied.  

This ensures **precise control** over who gets filtered, even when multiple people appear in the frame.

💡 **Note:** Without uploading a known face, the program cannot distinguish between known and unknown faces, so the filtering logic will not work correctly.
