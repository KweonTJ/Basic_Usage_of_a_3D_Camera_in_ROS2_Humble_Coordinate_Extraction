#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
import cv2
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped, Vector3
from cv_bridge import CvBridge

# [설정 상수]
TRACKING_SCALE = 0.5     
DETECTION_INTERVAL = 20  
TRACKER_SKIP = 3
MIN_DEPTH_TRUST = 0.5   # 45cm 이하는 Depth 센서 불신 -> 비전 계산 사용

class LowPassFilter:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.val = None

    def update(self, current_val):
        if self.val is None:
            self.val = current_val
        else:
            self.val = self.alpha * current_val + (1 - self.alpha) * self.val
        return self.val

class TrackedFruit:
    def __init__(self, obj_id, frame, bbox, z, fx):
        self.id = obj_id
        x, y, w, h = bbox
        
        # 필터 설정
        self.filter_x = LowPassFilter(0.4)
        self.filter_y = LowPassFilter(0.4)
        self.filter_z = LowPassFilter(0.6) # Z값 반응성 높임
        self.filter_w = LowPassFilter(0.3) 
        self.filter_h = LowPassFilter(0.3)
        
        self.track_bbox = bbox 
        self.x, self.y, self.w, self.h = int(x), int(y), int(w), int(h)
        
        # 초기 필터값 설정
        cx, cy = x + w / 2, y + h / 2
        self.filter_x.update(cx)
        self.filter_y.update(cy)
        self.z = self.filter_z.update(z)
        self.last_valid_z = z
        self.filter_w.update(w)
        self.filter_h.update(h)

        self.missing_count = 0
        
        # [학습 데이터] 실제 크기 (미터 단위)
        # 초기값: 깊이와 픽셀로 추정
        self.learned_real_w = (w * z) / fx if fx > 0 else 0.08
        self.learned_real_h = (h * z) / fx if fx > 0 else 0.08
        self.learned_count = 1
        
        # 디버깅용 데이터 저장소
        self.debug_sensor_z = z
        self.debug_visual_z = z
        self.mode = "INIT"

        self.tracker = cv2.TrackerCSRT_create()
        small_frame = cv2.resize(frame, (0, 0), fx=TRACKING_SCALE, fy=TRACKING_SCALE)
        small_bbox = tuple([int(v * TRACKING_SCALE) for v in bbox])
        self.tracker.init(small_frame, small_bbox)

    def update_visual(self, frame, do_calc=True):
        if do_calc:
            small_frame = cv2.resize(frame, (0, 0), fx=TRACKING_SCALE, fy=TRACKING_SCALE)
            success, small_bbox = self.tracker.update(small_frame)
            if success:
                self.track_bbox = tuple([int(v / TRACKING_SCALE) for v in small_bbox])
        else:
            success = True 

        tx, ty, tw, th = self.track_bbox
        cx, cy = tx + tw / 2, ty + th / 2
        
        smooth_x = self.filter_x.update(cx)
        smooth_y = self.filter_y.update(cy)
        self.smooth_w = self.filter_w.update(tw)
        self.smooth_h = self.filter_h.update(th)
        
        self.x = int(smooth_x - self.smooth_w/2)
        self.y = int(smooth_y - self.smooth_h/2)
        self.w, self.h = int(self.smooth_w), int(self.smooth_h)
        
        return success

    def get_center_smooth(self):
        return self.filter_x.val, self.filter_y.val

    def update_real_size(self, cur_w_m, cur_h_m):
        # 3cm ~ 30cm 사이의 정상적인 값만 학습
        if 0.03 < cur_w_m < 0.30:
            self.learned_real_w = (self.learned_real_w * self.learned_count + cur_w_m) / (self.learned_count + 1)
            self.learned_real_h = (self.learned_real_h * self.learned_count + cur_h_m) / (self.learned_count + 1)
            self.learned_count = min(self.learned_count + 1, 100)

class FruitPickerData(Node):
    def __init__(self):
        super().__init__('fruit_picker_data')
        self.bridge = CvBridge()
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)

        self.sub_depth = self.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, qos)
        self.sub_color = self.create_subscription(Image, '/camera/color/image_raw', self.color_callback, qos)
        self.sub_info = self.create_subscription(CameraInfo, '/camera/depth/camera_info', self.info_callback, qos)
        
        self.result_pub = self.create_publisher(Image, '/object/picking_view', 5)
        self.close_view_pub = self.create_publisher(Image, '/object/close_range_view', 5)

        # [토픽 발행기 생성] 최대 4개 물체에 대해 각각 생성
        self.pub_position = []   # 최종 좌표
        self.pub_size = []       # 크기 정보
        self.pub_depth_comp = [] # 깊이 비교 (Sensor vs Visual)
        
        for i in range(1, 5):
            # 1. 최종 좌표 (Robot이 구독할 토픽)
            self.pub_position.append(self.create_publisher(PointStamped, f'/object/fruit_{i}/position', 5))
            
            # 2. 크기 정보: x=실제너비(m), y=실제높이(m), z=픽셀너비(px)
            self.pub_size.append(self.create_publisher(PointStamped, f'/object/fruit_{i}/size', 5))
            
            # 3. 깊이 비교: x=최종Z, y=센서Z(Raw), z=비전계산Z
            self.pub_depth_comp.append(self.create_publisher(PointStamped, f'/object/fruit_{i}/depth_comp', 5))

        self.latest_color = None
        self.fruits = []
        self.next_id = 1
        self.frame_count = 0 
        self.fx = 550.0 
        self.fy = 550.0
        self.cx = 320.0
        self.cy = 240.0

    def info_callback(self, msg: CameraInfo):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def color_callback(self, msg: Image):
        try:
            self.latest_color = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except: pass

    def depth_callback(self, msg: Image):
        if self.latest_color is None: return

        try:
            self.frame_count += 1
            depth_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            depth = depth_raw.astype(np.float32) * 0.001 
            
            display_img = self.latest_color.copy()
            close_view_img = self.latest_color.copy()
            h_img, w_img = display_img.shape[:2]

            updated_fruits = []
            
            for fruit in self.fruits:
                success = fruit.update_visual(self.latest_color, do_calc=(self.frame_count % TRACKER_SKIP == 0))
                
                if success:
                    cx, cy = fruit.get_center_smooth()
                    u, v = int(cx), int(cy)
                    
                    # 1. Depth 센서 값 추출 (중심부 20%)
                    sample_w = max(3, int(fruit.w * 0.2))
                    sample_h = max(3, int(fruit.h * 0.2))
                    v_min, v_max = max(0, v - sample_h // 2), min(h_img, v + sample_h // 2)
                    u_min, u_max = max(0, u - sample_w // 2), min(w_img, u + sample_w // 2)
                    
                    roi_z = depth[v_min:v_max, u_min:u_max]
                    sensor_z = 0.0
                    if roi_z.size > 0:
                        valid_z = roi_z[(roi_z > 0.1) & (roi_z < 3.0)]
                        if valid_z.size > 0:
                            sensor_z = np.median(valid_z)

                    # 2. 비전(Visual) 거리 계산
                    # 공식: Z = (Focal * Real_Width) / Pixel_Width
                    visual_z = 0.0
                    if fruit.w > 0:
                        visual_z = (self.fx * fruit.learned_real_w) / fruit.w

                    # 3. 하이브리드 결정 로직
                    final_z = 0.0
                    
                    if sensor_z > MIN_DEPTH_TRUST:
                        # [원거리] 센서 신뢰
                        final_z = sensor_z
                        fruit.mode = "SENS"
                        
                        # 실제 크기 학습
                        cur_real_w = (fruit.w * final_z) / self.fx
                        cur_real_h = (fruit.h * final_z) / self.fy
                        fruit.update_real_size(cur_real_w, cur_real_h)
                    else:
                        # [근거리 or 센서실패] 비전 신뢰
                        if visual_z > 0:
                            final_z = visual_z
                            fruit.mode = "VISUAL"
                        else:
                            final_z = fruit.last_valid_z # 비상시 유지
                            fruit.mode = "HOLD"

                    # 필터 업데이트
                    fruit.z = fruit.filter_z.update(final_z)
                    if fruit.z > 0: fruit.last_valid_z = fruit.z
                    
                    # 디버깅 데이터 저장
                    fruit.debug_sensor_z = sensor_z
                    fruit.debug_visual_z = visual_z

                    fruit.missing_count = 0
                    updated_fruits.append(fruit)

                    # --- 시각화 (Close View) ---
                    color = (0, 255, 0) if fruit.mode == "SENS" else (0, 0, 255)
                    cv2.rectangle(close_view_img, (fruit.x, fruit.y), (fruit.x+fruit.w, fruit.y+fruit.h), color, 3)
                    lines = [
                        f"ID:{fruit.id} [{fruit.mode}]",
                        f"Final Z: {fruit.z:.3f}m",
                        f"Sens Z : {sensor_z:.3f}m",
                        f"Vis  Z : {visual_z:.3f}m",
                        f"Real W : {fruit.learned_real_w*100:.1f}cm"
                    ]
                    for idx, line in enumerate(lines):
                        cv2.putText(close_view_img, line, (fruit.x+fruit.w+10, fruit.y+20+(idx*20)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 4)
                        cv2.putText(close_view_img, line, (fruit.x+fruit.w+10, fruit.y+20+(idx*20)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

                    # --- 토픽 발행 (ID 1~4) ---
                    if fruit.id <= 4:
                        idx = fruit.id - 1
                        header = msg.header
                        header.frame_id = f"object_{fruit.id}"

                        # (A) Position (X, Y, Z) - 실제 로봇 좌표
                        # Z가 바뀌었으니 X, Y도 비율에 맞춰 재계산
                        real_x = (cx - self.cx) * fruit.z / self.fx
                        real_y = (cy - self.cy) * fruit.z / self.fy
                        
                        pos_msg = PointStamped()
                        pos_msg.header = header
                        pos_msg.point.x, pos_msg.point.y, pos_msg.point.z = float(real_x), float(real_y), float(fruit.z)
                        self.pub_position[idx].publish(pos_msg)

                        # (B) Size (Real W, Real H, Pixel W)
                        size_msg = PointStamped()
                        size_msg.header = header
                        size_msg.point.x = float(fruit.learned_real_w) # m 단위
                        size_msg.point.y = float(fruit.learned_real_h) # m 단위
                        size_msg.point.z = float(fruit.w)              # pixel 단위
                        self.pub_size[idx].publish(size_msg)

                        # (C) Depth Compare (Final, Sensor, Visual) - 그래프용
                        depth_msg = PointStamped()
                        depth_msg.header = header
                        depth_msg.point.x = float(fruit.z)           # 최종 필터링된 값
                        depth_msg.point.y = float(sensor_z)          # 센서 원본 (0일 수 있음)
                        depth_msg.point.z = float(visual_z)          # 비전 계산값
                        self.pub_depth_comp[idx].publish(depth_msg)

                else:
                    fruit.missing_count += 1
                    if fruit.missing_count < 60: updated_fruits.append(fruit)

            # 신규 탐지 (간소화된 로직)
            if len(updated_fruits) < 4 and (self.frame_count % DETECTION_INTERVAL == 0):
                # Depth 이미지 전처리
                range_mask = np.logical_and(depth >= 0.2, depth <= 1.2).astype(np.uint8) * 255
                kernel = np.ones((5,5), np.uint8)
                range_mask = cv2.erode(range_mask, kernel, iterations=1)
                contours, _ = cv2.findContours(range_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                detected = []
                for cnt in contours:
                    if cv2.contourArea(cnt) < 500: continue
                    x, y, w, h = cv2.boundingRect(cnt)
                    roi_d = depth[y:y+h, x:x+w]
                    z_val = np.median(roi_d[roi_d > 0.1]) if np.any(roi_d > 0.1) else 0
                    if z_val > 0: detected.append((x, y, w, h, z_val))
                
                # 중복 제거 및 등록
                for (x, y, w, h, z) in detected:
                    if len(updated_fruits) >= 4: break
                    # 기존 물체와 거리 비교
                    cx, cy = x + w/2, y + h/2
                    is_dup = False
                    for f in updated_fruits:
                        fcx, fcy = f.get_center_smooth()
                        if np.hypot(cx-fcx, cy-fcy) < 50: is_dup = True
                    
                    if not is_dup:
                        updated_fruits.append(TrackedFruit(self.next_id, self.latest_color, (x,y,w,h), z, self.fx))
                        self.next_id += 1

            self.fruits = updated_fruits
            self.result_pub.publish(self.bridge.cv2_to_imgmsg(display_img, encoding='bgr8'))
            self.close_view_pub.publish(self.bridge.cv2_to_imgmsg(close_view_img, encoding='bgr8'))

        except Exception as e:
            self.get_logger().error(f'Error: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = FruitPickerData()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()