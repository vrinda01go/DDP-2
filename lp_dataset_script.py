# import cv2
# import pandas as pd
# import os

# video_path = "C:\\Users\\workstation\\lp_dataset_neurips\\lp_videos\\vid1.mp4"
# output_excel = "C:\\Users\\workstation\\lp_dataset_neurips\\lp_videos\\license_plates_annotations.xlsx"
# output_folder = "C:\\Users\\workstation\\lp_dataset_neurips\\lp_videos\\frames"
# os.makedirs(output_folder, exist_ok=True)

# data = []
# start_frame = None
# end_frame = None
# scale_factor = 1.0  # Modify if scaling is needed

# cap = cv2.VideoCapture(video_path)
# frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# print(f"Total frames: {frame_count}")

# frame_idx = 0
# box = []

# def draw_bbox(event, x, y, flags, param):
#     global box
#     if event == cv2.EVENT_LBUTTONDOWN:
#         box = [(x, y)]
#     elif event == cv2.EVENT_LBUTTONUP:
#         box.append((x, y))
#         print("Bounding Box:", box)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     display = frame.copy()
#     resized_frame = cv2.resize(display, (0, 0), fx=scale_factor, fy=scale_factor)
#     window_name = f"Frame {frame_idx}"
#     cv2.namedWindow(window_name)
#     cv2.setMouseCallback(window_name, draw_bbox)
#     cv2.imshow(window_name, resized_frame)

#     key = cv2.waitKey(0) & 0xFF
#     if key == ord('s'):  # save frame
#         if start_frame is None:
#             start_frame = frame_idx
#         if len(box) == 2:
#             x1, y1 = box[0]
#             x2, y2 = box[1]
#             bbox = [min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)]

#             filename = f"{output_folder}/frame_{frame_idx}.jpg"
#             cv2.imwrite(filename, frame)

#             data.append({
#                 "video": video_path,
#                 "frame_number": frame_idx,
#                 "start_frame": start_frame,
#                 "end_frame": None,  # to be filled when sequence ends
#                 "x": bbox[0], "y": bbox[1],
#                 "width": bbox[2], "height": bbox[3],
#                 "scale_factor": scale_factor
#             })

#         box = []
#     elif key == ord('e'):  # mark end of current sequence
#         if start_frame is not None:
#             end_frame = frame_idx
#             for d in data:
#                 if d["start_frame"] == start_frame and d["end_frame"] is None:
#                     d["end_frame"] = end_frame
#             start_frame = None
#     elif key == ord('q'):
#         break

#     frame_idx += 1
#     cv2.destroyWindow(window_name)

# cap.release()
# cv2.destroyAllWindows()

# df = pd.DataFrame(data)
# df.to_excel(output_excel, index=False)
# print(f"Annotations saved to {output_excel}")

##################### manual annotations & save video clip (m,n,s,v,q)##############################
# import cv2
# import os
# import pandas as pd

# # === Configurations ===
# video_path = "lp_videos/vid1.mp4"  # Replace with your actual video file
# video_base = os.path.splitext(os.path.basename(video_path))[0]
# output_clip_dir = os.path.join("lp_videos", f"{video_base}_clips")
# output_crop_dir = os.path.join("lp_videos", f"{video_base}_cropped_plates")
# output_excel = os.path.join("lp_videos", "clip_annotations.xlsx")
# scale_factor = 1.0  # Keep original while resizing for view
# target_resolution = (1280, 720)  # 720p

# os.makedirs(output_clip_dir, exist_ok=True)
# os.makedirs(output_crop_dir, exist_ok=True)

# # === Video capture ===
# cap = cv2.VideoCapture(video_path)
# fps = cap.get(cv2.CAP_PROP_FPS)
# frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# print(f"Total frames: {frame_count}, FPS: {fps}")

# frame_idx = 0
# current_frame = None
# drawing = False
# box = []
# start_frame = None
# end_frame = None
# clip_count = 0
# clip_data = []

# # === Track annotations per clip ===
# annotations_in_clip = []

# def draw_bbox(event, x, y, flags, param):
#     global drawing, box, current_frame
#     if event == cv2.EVENT_LBUTTONDOWN:
#         drawing = True
#         box = [(x, y)]
#     elif event == cv2.EVENT_MOUSEMOVE and drawing:
#         img_copy = current_frame.copy()
#         cv2.rectangle(img_copy, box[0], (x, y), (0, 255, 0), 2)
#         cv2.imshow(param, img_copy)
#     elif event == cv2.EVENT_LBUTTONUP:
#         drawing = False
#         box.append((x, y))
#         print("Bounding Box:", box)

# def save_clip_and_crops(start, end, clip_path, clip_annotations, crop_folder):
#     cap_clip = cv2.VideoCapture(video_path)
#     cap_clip.set(cv2.CAP_PROP_POS_FRAMES, start)

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(clip_path, fourcc, fps, target_resolution)

#     frame_local_idx = start
#     while frame_local_idx <= end:
#         ret, frame = cap_clip.read()
#         if not ret:
#             break

#         resized_frame = cv2.resize(frame, target_resolution)
#         out.write(resized_frame)

#         # Save crops only for annotated frames
#         frame_annots = [a for a in clip_annotations if a["frame_number"] == frame_local_idx]
#         for i, annot in enumerate(frame_annots):
#             x, y, w, h = annot["bbox"]
#             crop = frame[y:y+h, x:x+w]
#             crop_filename = f"crop_{os.path.basename(clip_path).replace('.mp4','')}_f{frame_local_idx}_{i}.jpg"
#             cv2.imwrite(os.path.join(crop_folder, crop_filename), crop)

#         frame_local_idx += 1

#     cap_clip.release()
#     out.release()

# # === Frame-by-frame loop ===
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     display = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
#     current_frame = display.copy()
#     window_name = f"Frame {frame_idx}"
#     cv2.namedWindow(window_name)
#     cv2.setMouseCallback(window_name, draw_bbox, param=window_name)
#     cv2.imshow(window_name, display)

#     key = cv2.waitKey(0) & 0xFF

#     if key == ord('m'):  # mark start
#         start_frame = frame_idx
#         annotations_in_clip = []
#         print(f"▶️ Start frame set at {start_frame}")

#     elif key == ord('n'):  # mark end
#         end_frame = frame_idx
#         print(f"⏹️ End frame set at {end_frame}")

#     elif key == ord('s'):  # save current frame annotation
#         if len(box) == 2:
#             x1, y1 = box[0]
#             x2, y2 = box[1]
#             x, y = min(x1, x2), min(y1, y2)
#             w, h = abs(x2 - x1), abs(y2 - y1)

#             if w > 5 and h > 5:
#                 annotations_in_clip.append({
#                     "frame_number": frame_idx,
#                     "bbox": (int(x), int(y), int(w), int(h))
#                 })
#                 print(f"✅ Annotation added for frame {frame_idx}")
#         box = []

#     elif key == ord('v'):  # save the clip and crops
#         if start_frame is not None and end_frame is not None:
#             duration = round((end_frame - start_frame + 1) / fps, 2)
#             clip_filename = f"{video_base}_{clip_count}_{start_frame}_{end_frame}_{int(duration)}.mp4"

#             # clip_filename = f"clip_{clip_count:03d}.mp4"  
#             clip_path = os.path.join(output_clip_dir, clip_filename)

#             save_clip_and_crops(start_frame, end_frame, clip_path, annotations_in_clip, output_crop_dir)

#             annotated_frames = [a["frame_number"] for a in annotations_in_clip]
#             bboxes = [a["bbox"] for a in annotations_in_clip]

#             clip_data.append({
#                 "clip_path": clip_path,
#                 "original_video": os.path.basename(video_path),
#                 "start_frame": start_frame,
#                 "end_frame": end_frame,
#                 "fps": fps,
#                 "duration_sec": duration,
#                 "annotated_frames": annotated_frames,
#                 "bbox_coords": bboxes
#             })

#             clip_count += 1
#             print(f"🎞️ Saved {clip_filename} with {len(annotations_in_clip)} annotations.")

#             start_frame = None
#             end_frame = None
#             annotations_in_clip = []
#         else:
#             print("⚠️ Start/end frame missing. Clip not saved.")

#     elif key == ord('q'):
#         break

#     frame_idx += 1
#     try:
#         if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
#             cv2.destroyWindow(window_name)
#     except:
#         pass

# cap.release()
# cv2.destroyAllWindows()

# # === Save final Excel file ===
# if clip_data:
#     new_df = pd.DataFrame(clip_data)

#     if os.path.exists(output_excel):
#         # Load old data and append
#         old_df = pd.read_excel(output_excel)
#         final_df = pd.concat([old_df, new_df], ignore_index=True)
#     else:
#         # No old data exists
#         final_df = new_df

#     final_df.to_excel(output_excel, index=False)
#     print(f"✅ Updated annotation file saved to {output_excel}")
# else:
#     print("⚠️ No clips saved.")







######## YOLO putting bounding box around all lp and no lp also (in some cases)
# from ultralytics import YOLO
# import cv2
# import os
# import pandas as pd

# # === Configuration ===
# video_path = "lp_videos/vid1.mp4"  # Update with your actual video path
# video_base = os.path.splitext(os.path.basename(video_path))[0]
# start_frame = 0   # Set your desired start frame
# end_frame = 120   # Set your desired end frame (e.g., 5s @ 30fps = 150 frames)
# output_clip_dir = f"lp_videos/{video_base}_clips"
# output_crop_dir = f"lp_videos/{video_base}_cropped_plates"
# output_excel = f"lp_videos/clip_annotations_yolo.xlsx"
# target_resolution = (1280, 720)

# os.makedirs(output_clip_dir, exist_ok=True)
# os.makedirs(output_crop_dir, exist_ok=True)

# # === Load YOLOv8 model (you can replace with a license plate-specific model if available) ===
# model = YOLO("yolov8n.pt")  # Use "yolov8n.pt" or your custom-trained weights

# cap = cv2.VideoCapture(video_path)
# fps = cap.get(cv2.CAP_PROP_FPS)
# frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# print(f"Total frames: {frame_count}, FPS: {fps}")
# cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# clip_name = f"{video_base}_clip_{start_frame}_{end_frame}.mp4"
# clip_path = os.path.join(output_clip_dir, clip_name)
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(clip_path, fourcc, fps, target_resolution)

# frame_idx = start_frame
# annotations = []

# while frame_idx <= end_frame:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     resized_frame = cv2.resize(frame, target_resolution)
#     out.write(resized_frame)

#     # === Run YOLO detection ===
#     results = model(resized_frame, verbose=False)[0]

#     for i, box in enumerate(results.boxes):
#         cls_id = int(box.cls[0])
#         conf = float(box.conf[0])
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         w, h = x2 - x1, y2 - y1

#         # Save annotation
#         annotations.append({
#             "clip_path": clip_path,
#             "original_video": os.path.basename(video_path),
#             "frame_number": frame_idx,
#             "x": x1, "y": y1, "width": w, "height": h,
#             "class_id": cls_id,
#             "confidence": round(conf, 4)
#         })

#         # Save crop
#         crop = frame[y1:y2, x1:x2]
#         crop_filename = f"crop_{video_base}_f{frame_idx}_{i}.jpg"
#         cv2.imwrite(os.path.join(output_crop_dir, crop_filename), crop)

#     frame_idx += 1

# cap.release()
# out.release()

# # === Save annotations to Excel ===
# if annotations:
#     df = pd.DataFrame(annotations)
#     df.to_excel(output_excel, index=False)
#     print(f"✅ Annotations and crops saved. Excel: {output_excel}")
# else:
#     print("⚠️ No objects detected in selected frame range.")

import cv2
import os
import pandas as pd

# === Configurations ===
# video_path = "lp_videos/New York.mp4"
# video_base = os.path.splitext(os.path.basename(video_path))[0]
# output_clip_dir = os.path.join("lp_videos", f"{video_base}_clips")
# output_excel = os.path.join("lp_videos", "clip_annotations.xlsx")
# target_resolution = (1280, 720)

# # Manually specify start and end frames
# clip_ranges = [
#     (1200, 1500),     # Clip 1: start_frame, end_frame
#     (300, 540),   # Clip 2
#     (800, 950),   # Clip 3
# ]  # <--- EDIT here: list of (start_frame, end_frame)

# os.makedirs(output_clip_dir, exist_ok=True)

# # === Video capture ===
# cap = cv2.VideoCapture(video_path)
# fps = cap.get(cv2.CAP_PROP_FPS)
# frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# print(f"Total frames: {frame_count}, FPS: {fps}")

# clip_count = 0
# clip_data = []

# def save_clip(start, end, clip_path):
#     cap_clip = cv2.VideoCapture(video_path)
#     cap_clip.set(cv2.CAP_PROP_POS_FRAMES, start)

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(clip_path, fourcc, fps, target_resolution)

#     for frame_idx in range(start, end + 1):
#         ret, frame = cap_clip.read()
#         if not ret:
#             break

#         resized_frame = cv2.resize(frame, target_resolution)
#         out.write(resized_frame)

#     cap_clip.release()
#     out.release()

# # === Save specified clips ===
# for start_frame, end_frame in clip_ranges:
#     duration = round((end_frame - start_frame + 1) / fps, 2)
#     clip_filename = f"{video_base}_{clip_count}_{start_frame}_{end_frame}_{int(duration)}.mp4"
#     clip_path = os.path.join(output_clip_dir, clip_filename)

#     save_clip(start_frame, end_frame, clip_path)

#     clip_data.append({
#         "clip_path": clip_path,
#         "original_video": os.path.basename(video_path),
#         "start_frame": start_frame,
#         "end_frame": end_frame,
#         "fps": fps,
#         "duration_sec": duration
#     })

#     clip_count += 1
#     print(f"🎞️ Saved {clip_filename}")

# cap.release()

# # === Save final Excel file ===
# if clip_data:
#     new_df = pd.DataFrame(clip_data)

#     if os.path.exists(output_excel):
#         old_df = pd.read_excel(output_excel)
#         final_df = pd.concat([old_df, new_df], ignore_index=True)
#     else:
#         final_df = new_df

#     final_df.to_excel(output_excel, index=False)
#     print(f"✅ Updated annotation file saved to {output_excel}")
# else:
#     print("⚠️ No clips saved.")

import cv2
import os
import pandas as pd

# === Configurations ===
video_path = "lp_videos/Mumbai.mp4"
video_base = os.path.splitext(os.path.basename(video_path))[0]
output_clip_dir = os.path.join("lp_videos", f"{video_base}_clips")
output_crop_dir = os.path.join("lp_videos", f"{video_base}_cropped_plates")
output_excel = os.path.join("lp_videos", f"{video_base}_annotations.xlsx")  # Now per video
scale_factor = 1.0
resize_to = (1280, 720)
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"🎥 Video: {video_base}")
print(f"🧮 FPS: {fps}")
print(f"🧮 Total Frames: {frame_count}")
clip_frames = [
    (15*fps, 19*fps),
    (232*fps,237*fps),
    (264*fps,272*fps),
    (342*fps,346*fps),
    (398*fps,403*fps),
    (468*fps,473*fps),
]

os.makedirs(output_clip_dir, exist_ok=True)
os.makedirs(output_crop_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames: {frame_count}, FPS: {fps}")

all_annotations = []

# === BBox Drawing Callback ===
# === BBox Drawing Callback ===
def draw_bbox(event, x, y, flags, param):
    global drawing, start_point, boxes, current_frame

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        img_copy = current_frame.copy()
        x0, y0 = start_point
        width = abs(x - x0)
        height = abs(y - y0)
        cv2.rectangle(img_copy, start_point, (x, y), (0, 255, 0), 2)
        cv2.putText(img_copy, f"W:{width} H:{height}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)  # Display width-height live
        cv2.imshow(param, img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x0, y0 = start_point
        x_min, y_min = min(x0, x), min(y0, y)
        w, h = abs(x - x0), abs(y - y0)
        if w > 5 and h > 5:
            boxes.append((x_min, y_min, w, h))
            print(f"✅ Saved box: x={x_min}, y={y_min}, w={w}, h={h}")


# === Clip Loop ===
clip_idx = 0
for start_sec, end_sec in clip_frames:
    start_frame = int(start_sec)
    end_frame = int(end_sec)

    print(f"\n🎯 Clip {clip_idx}:Frames {start_frame}-{end_frame}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []

    while cap.get(cv2.CAP_PROP_POS_FRAMES) <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, resize_to)
        frames.append(frame)

    clip_filename = f"{video_base}_clip{clip_idx}.mp4"
    clip_path = os.path.join(output_clip_dir, clip_filename)

    # Save clip
    out = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, resize_to)
    for f in frames:
        out.write(f)
    out.release()
    print(f"🎞️ Saved clip {clip_filename}")

    # Now annotate
    frames_to_skip = int(fps)  # frames to skip when pressing 'n'
    frame_num_in_clip = 0
    while frame_num_in_clip < len(frames):
        frame = frames[frame_num_in_clip]
        window_name = f"Clip {clip_idx} Frame {frame_num_in_clip}"
        current_frame = frame.copy()
        boxes = []

        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, draw_bbox, param=window_name)

        while True:
            cv2.imshow(window_name, current_frame)
            key = cv2.waitKey(0) & 0xFF

            if key == ord('s'):
                # Save drawn boxes
                for idx, (x, y, w, h) in enumerate(boxes):
                    crop = frame[y:y+h, x:x+w]
                    crop_filename = f"crop_{video_base}_clip{clip_idx}_frame{frame_num_in_clip}_box{idx}.jpg"
                    crop_path = os.path.join(output_crop_dir, crop_filename)
                    cv2.imwrite(crop_path, crop)

                    all_annotations.append({
                        "clip_filename": clip_filename,
                        "frame_number_in_clip": frame_num_in_clip,
                        "bbox": (x, y, w, h),
                        "crop_path": crop_path
                    })
                frame_num_in_clip += 1  # normal next frame after save
                break

            elif key == ord('n'):
                print(f"➡️ Skipped frame {frame_num_in_clip}")
                frame_num_in_clip += frames_to_skip  # ⬅️ jump ahead fps frames
                break

            elif key == ord('r'):
                boxes = []
                current_frame = frame.copy()
                print("↩️ Reset boxes.")

            elif key == ord('q'):
                print("⛔ Exiting annotation.")
                exit()

        cv2.destroyWindow(window_name)
        frame_num_in_clip += 1

    clip_idx += 1

cap.release()
cv2.destroyAllWindows()

# === Save Excel ===
if all_annotations:
    df = pd.DataFrame(all_annotations)
    df.to_excel(output_excel, index=False)
    print(f"✅ Annotation file saved at {output_excel}")
else:
    print("⚠️ No annotations made!")
