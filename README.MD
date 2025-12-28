#겨울방학 #캡스톤_디자인 
[Camera](https://rucrazia.tistory.com/111)
# 개요
- 본 프로젝트는 Orbbec Astra 3D 카메라를 ROS2 Humble 환경에서 사용하기 위한 기초 단계로,  깊이 카메라로부터 획득한 **Depth 정보와 Point Cloud 데이터를 ROS2 토픽으로 처리하고,  특정 픽셀 또는 ROI에 대응하는 3차원 좌표를 추출하는 방법을 구현**하는 것을 목표로 한다.
- Jetson Nano 기반 TurtleBot 환경에서 Astra 카메라 드라이버를 설정하고,  RGB 이미지–Depth–Point Cloud 간의 관계를 명확히 이해하며,  카메라 좌표계 (Camera Frame) 기준의 3D 좌표를 안정적으로 획득하는 파이프라인을 구성한다.
- 이후 매니퓰레이터 제어 및 고급 인식·제어 알고리즘의 **기반 데이터 생성 단계**에 해당한다.

# 환경 설정
### 하드웨어
- Astra 3D Depth Camera
	- 최소 인식 거리 : 0.5m
### 소프트웨어
- Ubuntu 22.04
- ROS2 Humble
- Python 3.10
#### 주의 사항
- numpy의 경우 1.xx.x 버전의 필요로 가상환경 필수

# 패키지 생성
- **depth-object-dectector** : 기존 Depth 시 나오는 노이즈를 최대한 억제한다
- **기대 효과** : depth시 0.5m이내에 있는 물체의 노이즈 억제. 측정한 물체의 좌표 추출
### Setp 1
- 3D Camera와 물체 사이의 거리 측정 및 바운딩 박스 체크
- 카메라와 제일 가까운 물체 1개 측정
### Step 2
- n개의 물체 측정 및 물체 갯수 파악
- 개별 바운딩 박스 체크 및 좌표 topic 발행
	- 과일 채집을 위해 겹쳐있거나 몰려 있는 물체의 좌표를 정확하게 추출하기 위함
### Step 3 
- Depth + Color 퓨전 추정
- Depth에서 특정 거리이상시 물체가 측정되지 않더라도, Color 영상으로 이어서 추정
- 바운딩 박스에 한번 잡힌 물체는 카메라가 이동하더라도 물체를 그대로 잡고 진행

# 실행
### teminal 1
```
(ros2_venv) ros2 run depth_object_detector depth_object_detector
```
### terminal 2 
```
rqt
```
### terminal 3
```
ros2 launch astra_camera astra_mini.launch.py
```
### terminal 4
```
for i in {1..4}; do 
  echo "------------------------------";
  echo "[Target $i Coordinate]";
  ros2 topic echo /object/target_$i --once; 
done
```
