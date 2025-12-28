#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge

class TrackedFruit:
    """과일의 ID 관리 및 배경 노이즈를 최소화한 정밀 추적기"""
    def __init__(self, obj_id, frame, bbox, z):
        self.id = obj_id
        self.z = z
        # [개선 1] ROI Shrinking: 배경 간섭을 줄이기 위해 실제 박스의 중앙 70%만 추적 범위로 사용
        self.bbox = self._shrink_bbox(bbox, 0.7)
        self.missing_count = 0
        self.is_picking_mode = False 
        
        # CSRT 추적기 초기화
        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(frame, self.bbox)

    def _shrink_bbox(self, bbox, scale):
        """바운딩 박스를 중앙 기준으로 축소시켜 경계면의 노이즈 유입 방지"""
        x, y, w, h = bbox
        nw, nh = w * scale, h * scale
        nx = x + (w - nw) / 2
        ny = y + (h - nh) / 2
        return (int(nx), int(ny), int(nw), int(nh))

    def update_visual(self, frame):
        """영상에서 물체의 위치 업데이트"""
        success, new_bbox = self.tracker.update(frame)
        if success:
            self.bbox = new_bbox
        return success, new_bbox

    def get_center(self):
        """추적 중인 박스의 중앙 픽셀 좌표 반환"""
        x, y, w, h = self.bbox
        return int(x + w/2), int(y + h/2)

class FruitPickerPro(Node):
    def __init__(self):
        super().__init__('fruit_picker_pro')
        self.bridge = CvBridge()
        
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5)

        # 구독 설정
        self.sub_depth = self.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, qos)
        self.sub_color = self.create_subscription(Image, '/camera/color/image_raw', self.color_callback, qos)
        
        # 퍼블리셔 설정
        self.result_pub = self.create_publisher(Image, '/object/picking_view', 10)
        
        # [요구사항] 총 4개의 토픽을 개별 발행하기 위한 퍼블리셔 리스트
        self.target_pubs = []
        for i in range(1, 5):
            self.target_pubs.append(self.create_publisher(PointStamped, f'/object/target_{i}', 10))

        self.latest_color = None
        self.fruits = []
        self.next_id = 0
        self.max_missing = 30 # 약 1초간 유지

    def color_callback(self, msg: Image):
        self.latest_color = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def depth_callback(self, msg: Image):
        if self.latest_color is None: return

        try:
            depth_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            depth = depth_raw.astype(np.float32) / 1000.0
            display_img = self.latest_color.copy()

            # 1. Depth 마스크 생성 및 노이즈 제거 강화
            mask = np.logical_and(depth >= 0.35, depth <= 1.6).astype(np.uint8) * 255
            
            # [개선 2] Morphological Open으로 물체 간 연결 부위 절단, Close로 구멍 메우기
            kernel = np.ones((7, 7), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) 
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
            
            detected_depth_objs = []
            for i in range(1, num):
                if stats[i, cv2.CC_STAT_AREA] < 1500: continue # 최소 면적 상향 (노이즈 제거)
                z_vals = depth[labels == i][depth[labels == i] > 0]
                z = np.mean(z_vals) if len(z_vals) > 0 else 0.0
                detected_depth_objs.append({'bbox': stats[i][:4], 'z': z, 'center': centroids[i]})

            # 2. 기존 객체 추적 및 매칭
            updated_fruits = []
            for fruit in self.fruits:
                success, _ = fruit.update_visual(self.latest_color)
                
                if success:
                    f_center = np.array(fruit.get_center())
                    
                    # [개선 3] Centroid 매칭 엄격화 (거리 임계값 축소로 오인식 방지)
                    best_match_idx = -1
                    min_dist = 50.0 
                    for idx, d_obj in enumerate(detected_depth_objs):
                        dist = np.linalg.norm(f_center - d_obj['center'])
                        if dist < min_dist:
                            min_dist, best_match_idx = dist, idx
                    
                    if best_match_idx != -1:
                        # Depth 데이터로 정보 갱신 및 박스 재축소(Shrink) 적용
                        fruit.z = detected_depth_objs[best_match_idx]['z']
                        fruit.bbox = fruit._shrink_bbox(detected_depth_objs[best_match_idx]['bbox'], 0.7)
                        fruit.is_picking_mode = False
                        detected_depth_objs.pop(best_match_idx)
                    else:
                        # 근거리에 Depth 객체가 없으면 Picking 모드 진입 (혹은 거리 유실)
                        fruit.is_picking_mode = True
                    
                    fruit.missing_count = 0
                    
                    # 시각화 데이터 준비
                    x, y, w, h = [int(v) for v in fruit.bbox]
                    x, y = max(0, x), max(0, y)
                    color = (0, 165, 255) if fruit.is_picking_mode else (0, 255, 0)
                    cv2.rectangle(display_img, (x, y), (x + w, y + h), color, 2)
                    u, v = fruit.get_center()
                    cv2.circle(display_img, (u, v), 5, (255, 0, 0), -1)
                    text = f"ID:{fruit.id} ({fruit.z:.2f}m)"
                    cv2.putText(display_img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    updated_fruits.append(fruit)
                else:
                    # 추적 실패 시 유예 기간 부여
                    fruit.missing_count += 1
                    if fruit.missing_count < self.max_missing:
                        updated_fruits.append(fruit)

            # 3. 신규 객체 등록 (최대 4개)
            if len(updated_fruits) < 4:
                detected_depth_objs.sort(key=lambda x: x['z']) # 가까운 순으로 등록
                for d_obj in detected_depth_objs:
                    if len(updated_fruits) >= 4: break
                    new_fruit = TrackedFruit(self.next_id, self.latest_color, d_obj['bbox'], d_obj['z'])
                    updated_fruits.append(new_fruit)
                    self.next_id += 1

            # 4. [요구사항] ID 순으로 정렬하여 4개의 개별 토픽 발행
            updated_fruits.sort(key=lambda f: f.id)
            
            for i in range(4): # 총 4개의 퍼블리셔 순회
                if i < len(updated_fruits):
                    f = updated_fruits[i]
                    u, v = f.get_center()
                    
                    pt = PointStamped()
                    pt.header = msg.header
                    pt.point.x, pt.point.y, pt.point.z = float(u), float(v), float(f.z)
                    self.target_pubs[i].publish(pt)
                else:
                    # 물체가 4개 미만일 경우, 빈 데이터나 특정 플래그를 보낼 필요가 없다면 통과
                    pass

            self.fruits = updated_fruits
            self.result_pub.publish(self.bridge.cv2_to_imgmsg(display_img, encoding='bgr8'))

        except Exception as e:
            self.get_logger().error(f'System Error: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = FruitPickerPro()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()