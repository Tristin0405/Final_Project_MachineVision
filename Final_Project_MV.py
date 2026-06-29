import cv2 as cv
import numpy as np

# PHẦN 1: TỰ ĐỘNG TÌM VẠCH SÂN VÀ TÍNH MA TRẬN M 
cap = cv.VideoCapture(r"D:\MonHoc\MachineVision\Final\Badmintun.mp4")

width_court, height_court = 610, 670
M = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        exit()

    frame = cv.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
    h_frame, w_frame = frame.shape[:2]

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)

    #Mask toàn bộ mặt sân
    lower_floor = np.array([35, 40, 40])
    upper_floor = np.array([130, 255, 255])
    floor_mask_raw = cv.inRange(hsv, lower_floor, upper_floor)
    kernel_floor = cv.getStructuringElement(cv.MORPH_RECT, (25, 25))
    floor_mask_closed = cv.morphologyEx(floor_mask_raw, cv.MORPH_CLOSE, kernel_floor)
    contours, _ = cv.findContours(floor_mask_closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    roi_mask = np.zeros_like(floor_mask_raw)
    
    if contours:
        largest_contour = max(contours, key=cv.contourArea)
        cv.drawContours(roi_mask, [largest_contour], -1, 255, -1)

    #Mask vạch trắng kết hợp CLAHE
    clahe = cv.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4))
    v_clahe = clahe.apply(v)
    hsv_adjusted = cv.merge([h, s, v_clahe])
    
    dynamic_lower_v = int(np.percentile(v_clahe, 93)) 
    dynamic_upper_s = int(np.percentile(s, 15))
    dynamic_upper_s = max(25, min(dynamic_upper_s, 40))
    lower_white = np.array([0, 0, dynamic_lower_v])
    upper_white = np.array([180, dynamic_upper_s, 255])

    white_mask = cv.inRange(hsv_adjusted, lower_white, upper_white)
    mask = cv.bitwise_and(white_mask, white_mask, mask=roi_mask)
    
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)
    edges = cv.Canny(mask, 50, 150)
    lines = cv.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=40, minLineLength=150, maxLineGap=50)

    #Gom nhóm đường thẳng
    left_lines, right_lines = [], []
    center_x = w_frame // 2

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if 35 < abs(angle) < 85:
                length = np.hypot(x2-x1, y2-y1)
                x_center = (x1+x2)/2
                if x_center < center_x: left_lines.append((x_center, length, x1, y1, x2, y2))
                else: right_lines.append((x_center, length, x1, y1, x2, y2))

    #Tính toán Ma trận M
    if len(left_lines) > 0 and len(right_lines) > 0:
        left_line = min(left_lines, key=lambda x: x[0])
        _, _, xl1, yl1, xl2, yl2 = left_line
        max_line = max(right_lines, key=lambda x: x[0])
        _, _, xr1, yr1, xr2, yr2 = max_line

        if yl1 > yl2: xl1, xl2, yl1, yl2 = xl2, xl1, yl2, yl1
        if yr1 > yr2: xr1, xr2, yr1, yr2 = xr2, xr1, yr2, yr1

        y_top = min(yl1, yr1)
        y_bottom = max(yl2, yr2)

        def get_x_from_y(x1, y1, x2, y2, target_y):
            if y2 == y1: return x1
            return int(x1 + (x2 - x1) * (target_y - y1) / (y2 - y1))

        new_xl_bottom = get_x_from_y(xl1, yl1, xl2, yl2, y_bottom)
        new_xr_bottom = get_x_from_y(xr1, yr1, xr2, yr2, y_bottom)

        perspective_ratio = 0.38 
        y_net = int(y_top + (y_bottom - y_top) * perspective_ratio)
        mid_left = (get_x_from_y(xl1, yl1, xl2, yl2, y_net), y_net)
        mid_right = (get_x_from_y(xr1, yr1, xr2, yr2, y_net), y_net)

        src = np.float32([mid_left, mid_right, [new_xr_bottom, y_bottom], [new_xl_bottom, y_bottom]])
        dst = np.float32([[0,0], [width_court,0], [width_court,height_court], [0,height_court]])
        
        M = cv.getPerspectiveTransform(src, dst)
        break 


#THIẾT LẬP TRACKING & THÔNG SỐ HEATMAP
cap.set(cv.CAP_PROP_POS_FRAMES, 0) # Tua lại video từ đầu

#CÁC THÔNG SỐ HEATMAP
HEAT_PER_FRAME = 2      # Nhiệt lượng mỗi frame (Số nhỏ để đỏ từ từ)
MAX_HEAT = 100          # Ngưỡng nhiệt tối đa (Chạm mốc này là đỏ rực)
RADIUS = 20             # Bán kính vết chân (Càng to thì hòn đảo càng lớn)
BLUR_SIZE = 45          # Mức độ làm nhòe (Số lẻ, càng lớn thì màu càng quyện vào nhau)
MASK_THRESHOLD = 15     # Ngưỡng cắt rìa (Tạo viền răng cưa/hòn đảo cho vùng màu xanh dương)

heatmap_acc = np.zeros((height_court, width_court), dtype=np.float32)

kalman = cv.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.2
kalman.errorCovPost = np.eye(4, dtype=np.float32)

last_w, last_h = 60, 120 
initialized = False
last_y = 0              
last_cx, last_cy = 0, 0 
current_speed = 10.0    
ema_foot_x, ema_foot_y = None, None
alpha_ema = 0.7 

fgbg = cv.createBackgroundSubtractorKNN(history=1000, dist2Threshold=600, detectShadows=True)

# PHẦN 3: VÒNG LẶP CHÍNH (TRACKING & VẼ HEATMAP CHUẨN)
while cap.isOpened(): 
    ret, frame = cap.read()
    if not ret: break
        
    frame = cv.resize(frame, (int(0.5 * frame.shape[1]), int(0.5 * frame.shape[0])))
    h_frame, w_frame = frame.shape[:2]
    roi_y = h_frame // 2
    roi = frame[roi_y:, :].copy() 

    #DYNAMIC LEARNING RATE & ANTI-SHRINKAGE
    current_lr = 0.0001 if (initialized and current_speed < 2.0) else 0.01
    fgmask = fgbg.apply(roi, learningRate=current_lr)
    _, fgmask = cv.threshold(fgmask, 250, 255, cv.THRESH_BINARY)
    
    kernel_open = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)) 
    masked = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel_open, iterations=1)
    masked = cv.morphologyEx(masked, cv.MORPH_CLOSE, kernel_close, iterations=2)
    
    contours, _ = cv.findContours(masked, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    detected_box = None
    
    if contours:
        largest = max(contours, key=cv.contourArea)
        if cv.contourArea(largest) > 500: 
            hull = cv.convexHull(largest)
            x, y, w_box, h_box = cv.boundingRect(hull)
            if w_box < h_box * 2: 
                if initialized:
                    if h_box < last_h * 0.8 and abs(y - last_y) < 20: h_box = last_h 
                detected_box = (x, y, w_box, h_box)
                last_y = y 

    #KALMAN FILTER
    prediction = kalman.predict()
    pred_cx, pred_cy = int(prediction[0][0]), int(prediction[1][0])

    if detected_box is not None:
        x, y, w_box, h_box = detected_box
        cx, cy = x + w_box // 2, y + h_box // 2
        last_w, last_h = w_box, h_box
        measurement = np.array([[np.float32(cx)],[np.float32(cy)]], dtype=np.float32)

        if not initialized:
            kalman.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
            initialized = True
            final_cx, final_cy = cx, cy
        else:
            estimated = kalman.correct(measurement)
            final_cx, final_cy = int(estimated[0][0]), int(estimated[1][0])
    else:
        final_cx, final_cy = pred_cx, pred_cy

    if initialized:
        current_speed = np.sqrt((final_cx - last_cx)**2 + (final_cy - last_cy)**2)
        last_cx, last_cy = final_cx, final_cy

    #TÌM TỌA ĐỘ CHÂN TỐI ƯU & BỘ LỌC EMA
    if initialized:
        fx = max(0, min(final_cx - last_w // 2, w_frame - last_w))
        true_y = max(roi_y, min(final_cy - last_h // 2 + roi_y, h_frame - last_h))

        foot_h_ratio = int(last_h * 0.3)
        foot_start_y = true_y + last_h - foot_h_ratio
        
        mask_roi_y1 = max(0, foot_start_y - roi_y) 
        mask_roi_y2 = min(masked.shape[0], true_y + last_h - roi_y)
        mask_roi_x1 = max(0, fx)
        mask_roi_x2 = min(masked.shape[1], fx + last_w)
        foot_mask = masked[mask_roi_y1:mask_roi_y2, mask_roi_x1:mask_roi_x2]
        
        raw_foot_x = fx + last_w // 2
        raw_foot_y = true_y + last_h
        
        if foot_mask.size > 0:
            white_points = cv.findNonZero(foot_mask)
            if white_points is not None:
                y_coords = white_points[:, 0, 1]
                max_y_local = np.max(y_coords) 
                lowest_points = white_points[y_coords >= max_y_local - 5]
                avg_x_local = int(np.mean(lowest_points[:, 0, 0]))
                raw_foot_x = mask_roi_x1 + avg_x_local
                raw_foot_y = mask_roi_y1 + max_y_local + roi_y

        if ema_foot_x is None or ema_foot_y is None:
            ema_foot_x, ema_foot_y = raw_foot_x, raw_foot_y
        else:
            ema_foot_x = alpha_ema * raw_foot_x + (1 - alpha_ema) * ema_foot_x
            ema_foot_y = alpha_ema * raw_foot_y + (1 - alpha_ema) * ema_foot_y
            
        final_foot_pos = (int(ema_foot_x), int(ema_foot_y))

        #HIỂN THỊ CAMERA GỐC
        cv.rectangle(frame, (fx, true_y), (fx + last_w, true_y + last_h), (0, 255, 0), 2)
        cv.circle(frame, final_foot_pos, 6, (0, 0, 255), -1)

        # 4. TÍCH LŨY VÀ TẠO BẢN ĐỒ NHIỆT
        pts_camera = np.array([[[final_foot_pos[0], final_foot_pos[1]]]], dtype=np.float32)
        pts_warped = cv.perspectiveTransform(pts_camera, M)
        warp_x = int(pts_warped[0][0][0])
        warp_y = int(pts_warped[0][0][1])

        # Bơm nhiệt: Tạo mask tạm thời và CỘNG DỒN (cv.add) vào heatmap tổng
        if 0 <= warp_x < width_court and 0 <= warp_y < height_court:
            temp_heat = np.zeros_like(heatmap_acc)
            cv.circle(temp_heat, (warp_x, warp_y), RADIUS, HEAT_PER_FRAME, -1)
            cv.add(heatmap_acc, temp_heat, dst=heatmap_acc)

    # Blur và tạo màu (Colormap)
    heatmap_blur = cv.GaussianBlur(heatmap_acc, (BLUR_SIZE, BLUR_SIZE), 0)
    heatmap_clipped = np.clip(heatmap_blur, 0, MAX_HEAT) 
    heatmap_norm = np.uint8(heatmap_clipped * (255.0 / float(MAX_HEAT)))
    
    # Dùng COLORMAP_JET để có màu Xanh dương -> Vàng -> Đỏ
    color_map = cv.applyColorMap(heatmap_norm, cv.COLORMAP_JET)

    #VẼ LÊN NỀN SA BÀN 2D
    tactical_board = np.zeros((height_court, width_court, 3), dtype=np.uint8)
    tactical_board[:] = (75, 140, 75) 

    # Trộn Heatmap lên nền xanh
    mask_heat = heatmap_norm > MASK_THRESHOLD 
    if np.any(mask_heat):
        tactical_board[mask_heat] = cv.addWeighted(tactical_board, 0.35, color_map, 0.65, 0)[mask_heat]

    #VẼ VẠCH KẺ TRẮNG LÊN TRÊN CÙNG
    cv.rectangle(tactical_board, (0, 0), (width_court, height_court), (255, 255, 255), 2)
    cv.line(tactical_board, (width_court//2, 0), (width_court//2, height_court), (255, 255, 255), 2)
    short_service_y = int(height_court * 0.3)
    cv.line(tactical_board, (0, short_service_y), (width_court, short_service_y), (255, 255, 255), 2)
    cv.line(tactical_board, (46, 0), (46, height_court), (255, 255, 255), 2)
    cv.line(tactical_board, (width_court - 46, 0), (width_court - 46, height_court), (255, 255, 255), 2)

    # Hiển thị
    cv.imshow("Live Camera", frame)
    cv.imshow("Heatmap", tactical_board)

    if cv.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
