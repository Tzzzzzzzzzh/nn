import sys
import glob
import random
import numpy as np
import queue
import cv2
import os
import xml.etree.ElementTree as ET
import time
import pygame

# 如果你的环境已经能 import carla，下面这段不需要改。
# 如果不能 import carla，可以把 CARLA egg 路径加入 sys.path。
# 示例：
# sys.path.append(r"D:\CARLA_0.9.11\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.11-py3.7-win-amd64.egg")
import carla


# ==============================
# 基础配置
# ==============================
HOST = "localhost"
PORT = 2000
TM_PORT = 8000

MAP_NAME = "Town05"
NUM_BACKGROUND_VEHICLES = 10

IMAGE_W = 1024
IMAGE_H = 1024
CAMERA_FOV = 70

SPEED_LIMIT = 50.0                 # km/h，超速阈值
RED_LIGHT_SPEED_THRESHOLD = 3.0    # km/h，红灯状态下速度超过该值才算闯红灯
DISTANCE_THRESHOLD = 50.0          # 交通标志检测距离
WEATHER_TRANSITION_INTERVAL = 10   # 天气切换间隔，单位：秒
CAPTURE_COOLDOWN = 5               # 同一交通标志重复采集冷却时间，单位：秒


# ==============================
# Pygame 初始化
# ==============================
pygame.init()
screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption("CARLA Control Window: M switch mode, X exit")


# ==============================
# 全局状态变量
# ==============================
violation_info = {
    "speeding": False,
    "red_light": False,
    "ignore_sign": False,
    "current_speed": 0.0,
    "traffic_light_state": "None",
    "at_traffic_light": False,
    "control_mode": "AUTO"
}

violation_count = {
    "speeding_cnt": 0,
    "red_light_cnt": 0
}

last_violation = {
    "speeding": False,
    "red_light": False
}

control_state = {
    "manual_mode": False,
    "reverse": False
}

captured_sign_locations = set()
last_capture_time = 0


# ==============================
# CARLA 连接
# ==============================
client = carla.Client(HOST, PORT)
client.set_timeout(60.0)


def load_map(map_name):
    return client.load_world(map_name)


# ==============================
# 车辆与速度
# ==============================
def spawn_background_vehicles(num_vehicles, world, spawn_points, traffic_manager):
    vehicle_bp_lib = world.get_blueprint_library().filter("vehicle.*")
    spawned_vehicles = []

    random.shuffle(spawn_points)

    for spawn_point in spawn_points:
        if len(spawned_vehicles) >= num_vehicles:
            break

        vehicle_bp = random.choice(vehicle_bp_lib)
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

        if vehicle:
            vehicle.set_autopilot(True, traffic_manager.get_port())
            spawned_vehicles.append(vehicle)

    print(f"背景车辆生成数量: {len(spawned_vehicles)} / {num_vehicles}")
    return spawned_vehicles


def spawn_ego_vehicle(world, spawn_points, vehicle_filter="vehicle.audi.a2"):
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.find(vehicle_filter)

    random.shuffle(spawn_points)

    for spawn_point in spawn_points:
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle is not None:
            print("主车生成成功")
            return vehicle

    raise RuntimeError("主车生成失败：所有出生点都被占用或不可用，请重启 CARLA 或减少背景车辆数量。")


def get_vehicle_speed(vehicle):
    vel = vehicle.get_velocity()
    speed = 3.6 * np.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
    return round(speed, 2)


# ==============================
# 改进点 1：红灯判断逻辑
# 不再使用全图 HSV 红色像素阈值。
# 改为读取 CARLA 交通灯真值状态，避免车尾灯、红色车辆、红色建筑误判。
# ==============================
def traffic_light_state_to_text(state):
    if state == carla.TrafficLightState.Red:
        return "Red"
    if state == carla.TrafficLightState.Yellow:
        return "Yellow"
    if state == carla.TrafficLightState.Green:
        return "Green"
    if state == carla.TrafficLightState.Off:
        return "Off"
    if state == carla.TrafficLightState.Unknown:
        return "Unknown"
    return str(state)


def detect_red_light_violation_by_carla(vehicle):
    """
    使用 CARLA API 判断红灯违章。

    判断条件：
    1. 车辆当前处在交通灯影响区域：vehicle.is_at_traffic_light()
    2. 当前交通灯状态为 Red
    3. 车辆速度大于 RED_LIGHT_SPEED_THRESHOLD

    返回：
    red_light_violation: 是否判定为闯红灯
    state_text: 当前交通灯状态文本
    at_traffic_light: 是否处在交通灯影响区域
    """
    at_traffic_light = vehicle.is_at_traffic_light()
    state = vehicle.get_traffic_light_state()
    state_text = traffic_light_state_to_text(state)

    speed = get_vehicle_speed(vehicle)

    red_light_violation = (
        at_traffic_light
        and state == carla.TrafficLightState.Red
        and speed > RED_LIGHT_SPEED_THRESHOLD
    )

    return red_light_violation, state_text, at_traffic_light


def detect_violations(vehicle):
    speed = get_vehicle_speed(vehicle)

    red_light_violation, state_text, at_traffic_light = detect_red_light_violation_by_carla(vehicle)

    violation_info["current_speed"] = speed
    violation_info["speeding"] = speed > SPEED_LIMIT
    violation_info["red_light"] = red_light_violation
    violation_info["ignore_sign"] = False
    violation_info["traffic_light_state"] = state_text
    violation_info["at_traffic_light"] = at_traffic_light
    violation_info["control_mode"] = "MANUAL" if control_state["manual_mode"] else "AUTO"

    # 只在从“未违章”变为“违章”的那一帧计数一次，避免同一事件每帧重复计数
    if violation_info["speeding"] and not last_violation["speeding"]:
        violation_count["speeding_cnt"] += 1

    if violation_info["red_light"] and not last_violation["red_light"]:
        violation_count["red_light_cnt"] += 1

    last_violation["speeding"] = violation_info["speeding"]
    last_violation["red_light"] = violation_info["red_light"]


# ==============================
# 改进点 2：自动驾驶 / 手动控制模式切换
# M 键切换：
# AUTO 模式：只让 Traffic Manager 控制车辆，不执行 apply_control
# MANUAL 模式：关闭 autopilot，只执行键盘控制
# ==============================
def set_autopilot_mode(vehicle, traffic_manager, enable_auto):
    control_state["manual_mode"] = not enable_auto

    if enable_auto:
        vehicle.set_autopilot(True, traffic_manager.get_port())
        violation_info["control_mode"] = "AUTO"
        print("已切换到 AUTO 自动驾驶模式")
    else:
        vehicle.set_autopilot(False)
        violation_info["control_mode"] = "MANUAL"
        print("已切换到 MANUAL 手动控制模式")


def handle_input(vehicle, traffic_manager):
    """
    返回 False 表示退出程序。
    返回 True 表示继续运行。
    """
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_x:
                print("X key pressed, exit.")
                return False

            if event.key == pygame.K_m:
                # M 键切换自动 / 手动
                if control_state["manual_mode"]:
                    set_autopilot_mode(vehicle, traffic_manager, True)
                else:
                    set_autopilot_mode(vehicle, traffic_manager, False)

            if event.key == pygame.K_r and control_state["manual_mode"]:
                control_state["reverse"] = not control_state["reverse"]
                print(f"倒车模式: {'ON' if control_state['reverse'] else 'OFF'}")

    # AUTO 模式下，不调用 apply_control，避免和 Traffic Manager 冲突
    if not control_state["manual_mode"]:
        return True

    keys = pygame.key.get_pressed()

    control = carla.VehicleControl()
    control.reverse = control_state["reverse"]

    if keys[pygame.K_w]:
        control.throttle = 0.65
    else:
        control.throttle = 0.0

    if keys[pygame.K_s]:
        control.brake = 0.75
    else:
        control.brake = 0.0

    if keys[pygame.K_a]:
        control.steer = -0.45
    elif keys[pygame.K_d]:
        control.steer = 0.45
    else:
        control.steer = 0.0

    vehicle.apply_control(control)
    return True


# ==============================
# 可视化绘制
# ==============================
def draw_violation_info(img):
    speed = violation_info["current_speed"]
    mode = violation_info["control_mode"]
    tl_state = violation_info["traffic_light_state"]
    at_tl = violation_info["at_traffic_light"]

    cv2.putText(img, f"Mode: {mode}   Press M to switch", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    cv2.putText(img, f"Speed: {speed} km/h", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.putText(img, f"TrafficLight: {tl_state}   AtLight: {at_tl}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    cv2.putText(img, f"Speeding Count: {violation_count['speeding_cnt']}", (20, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    cv2.putText(img, f"RedLight Count: {violation_count['red_light_cnt']}", (20, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    if violation_info["speeding"]:
        cv2.putText(img, "VIOLATION: SPEEDING!", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    if violation_info["red_light"]:
        cv2.putText(img, "VIOLATION: RED LIGHT!", (20, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)


# ==============================
# 相机投影与交通标志框
# ==============================
def build_projection_matrix(w, h, fov, is_behind_camera=False):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)

    if is_behind_camera:
        K[0, 0] = K[1, 1] = -focal
    else:
        K[0, 0] = K[1, 1] = focal

    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


def get_image_point(loc, K, w2c):
    point = np.array([loc.x, loc.y, loc.z, 1])
    point_camera = np.dot(w2c, point)

    # UE4 坐标系转换为标准相机坐标系
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

    if point_camera[2] == 0:
        point_camera[2] = 1e-6

    point_img = np.dot(K, point_camera)
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img[0:2]


def dot_product(v1, v2):
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z


def get_signs_bounding_boxes(world, vehicle_transform, camera_transform, K, world_2_camera, image_w, image_h):
    global captured_sign_locations, last_capture_time

    bounding_boxes = []
    camera_location = camera_transform.location
    vehicle_location = vehicle_transform.location
    vehicle_right_vector = vehicle_transform.get_right_vector()

    for obj in world.get_level_bbs(carla.CityObjectLabel.TrafficSigns):
        distance = obj.location.distance(vehicle_location)
        vector_to_object = obj.location - vehicle_location

        if distance < DISTANCE_THRESHOLD:
            right_side_dot_product = dot_product(vehicle_right_vector, vector_to_object)

            # 只采集车辆右侧交通标志
            if right_side_dot_product > 0:
                vector_to_camera = obj.location - camera_location
                camera_dot_product = dot_product(camera_transform.get_forward_vector(), vector_to_camera)

                sign_location_tuple = (
                    round(obj.location.x, 2),
                    round(obj.location.y, 2),
                    round(obj.location.z, 2)
                )

                if camera_dot_product > 0 and sign_location_tuple not in captured_sign_locations:
                    verts = [v for v in obj.get_world_vertices(carla.Transform())]

                    x_coords = [get_image_point(v, K, world_2_camera)[0] for v in verts]
                    y_coords = [get_image_point(v, K, world_2_camera)[1] for v in verts]

                    xmin, xmax = int(min(x_coords)), int(max(x_coords))
                    ymin, ymax = int(min(y_coords)), int(max(y_coords))

                    box_w = xmax - xmin
                    box_h = ymax - ymin
                    area = box_w * box_h

                    min_area_threshold = 10

                    if xmin >= 0 and ymin >= 0 and xmax < image_w and ymax < image_h:
                        aspect_ratio = box_w / float(box_h) if box_h != 0 else 0

                        if area > min_area_threshold and 0.5 < aspect_ratio < 2.0:
                            bounding_boxes.append({
                                "label": "TrafficSign",
                                "xmin": xmin,
                                "ymin": ymin,
                                "xmax": xmax,
                                "ymax": ymax
                            })

                            current_time = time.time()
                            if current_time - last_capture_time > CAPTURE_COOLDOWN:
                                captured_sign_locations.add(sign_location_tuple)
                                last_capture_time = current_time

    return bounding_boxes


# ==============================
# NMS
# ==============================
def compute_iou(box1, box2):
    x1 = max(box1["xmin"], box2["xmin"])
    y1 = max(box1["ymin"], box2["ymin"])
    x2 = min(box1["xmax"], box2["xmax"])
    y2 = min(box1["ymax"], box2["ymax"])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1["xmax"] - box1["xmin"]) * (box1["ymax"] - box1["ymin"])
    box2_area = (box2["xmax"] - box2["xmin"]) * (box2["ymax"] - box2["ymin"])

    if box1_area == 0 or box2_area == 0:
        return 0.0

    return inter_area / float(box1_area + box2_area - inter_area)


def non_maximum_suppression(bboxes, iou_threshold=0.2):
    if len(bboxes) == 0:
        return []

    bboxes = sorted(
        bboxes,
        key=lambda x: (x["xmax"] - x["xmin"]) * (x["ymax"] - x["ymin"]),
        reverse=True
    )

    final_bboxes = []

    while bboxes:
        current_box = bboxes.pop(0)
        final_bboxes.append(current_box)
        bboxes = [box for box in bboxes if compute_iou(current_box, box) < iou_threshold]

    return final_bboxes


# ==============================
# 天气
# ==============================
weather_conditions = ["rainy", "sunny", "night", "foggy"]


def update_weather(world, condition):
    if condition == "rainy":
        weather = carla.WeatherParameters(
            cloudiness=80.0,
            precipitation=80.0,
            precipitation_deposits=80.0,
            wind_intensity=10.0,
            sun_azimuth_angle=270.0,
            sun_altitude_angle=10.0,
            fog_density=10.0,
            wetness=70.0
        )
    elif condition == "sunny":
        weather = carla.WeatherParameters(
            cloudiness=20.0,
            precipitation=0.0,
            precipitation_deposits=0.0,
            wind_intensity=5.0,
            sun_azimuth_angle=180.0,
            sun_altitude_angle=60.0,
            fog_density=0.0,
            wetness=0.0
        )
    elif condition == "night":
        weather = carla.WeatherParameters(
            cloudiness=0.0,
            precipitation=0.0,
            precipitation_deposits=0.0,
            wind_intensity=3.0,
            sun_azimuth_angle=0.0,
            sun_altitude_angle=-5.0,
            fog_density=0.0,
            wetness=0.0
        )
    elif condition == "foggy":
        weather = carla.WeatherParameters(
            cloudiness=0.0,
            precipitation=0.0,
            precipitation_deposits=0.0,
            wind_intensity=3.0,
            sun_azimuth_angle=0.0,
            sun_altitude_angle=0.0,
            fog_density=60.0,
            wetness=0.0
        )
    else:
        raise ValueError("Unknown weather condition")

    world.set_weather(weather)


def get_weather_params(world):
    weather = world.get_weather()
    return {
        "cloudiness": weather.cloudiness,
        "precipitation": weather.precipitation,
        "precipitation_deposits": weather.precipitation_deposits,
        "wind_intensity": weather.wind_intensity,
        "sun_azimuth_angle": weather.sun_azimuth_angle,
        "sun_altitude_angle": weather.sun_altitude_angle,
        "fog_density": weather.fog_density,
        "wetness": weather.wetness
    }


def get_weather_category(w):
    if w["cloudiness"] > 70 or w["precipitation"] > 50:
        return 0  # rainy
    elif w["sun_altitude_angle"] > 30:
        return 1  # sunny
    elif w["fog_density"] > 50:
        return 2  # foggy
    else:
        return 3  # night or other


# ==============================
# XML 保存
# ==============================
def create_xml_file(output_dir, image_name, bboxes, width, height, weather_params):
    annotation = ET.Element("annotation")

    filename = ET.SubElement(annotation, "filename")
    filename.text = image_name

    size = ET.SubElement(annotation, "size")

    width_elem = ET.SubElement(size, "width")
    width_elem.text = str(width)

    height_elem = ET.SubElement(size, "height")
    height_elem.text = str(height)

    depth_elem = ET.SubElement(size, "depth")
    depth_elem.text = "3"

    weather_category = get_weather_category(weather_params)
    weather = ET.SubElement(annotation, "weather")

    condition = ET.SubElement(weather, "condition")
    condition.text = str(weather_category)

    violation_node = ET.SubElement(annotation, "violation")

    ET.SubElement(violation_node, "speeding").text = str(violation_info["speeding"])
    ET.SubElement(violation_node, "red_light").text = str(violation_info["red_light"])
    ET.SubElement(violation_node, "ignore_sign").text = str(violation_info["ignore_sign"])
    ET.SubElement(violation_node, "current_speed").text = str(violation_info["current_speed"])
    ET.SubElement(violation_node, "traffic_light_state").text = str(violation_info["traffic_light_state"])
    ET.SubElement(violation_node, "at_traffic_light").text = str(violation_info["at_traffic_light"])
    ET.SubElement(violation_node, "control_mode").text = str(violation_info["control_mode"])

    for bbox in bboxes:
        obj = ET.SubElement(annotation, "object")

        name = ET.SubElement(obj, "name")
        name.text = bbox["label"]

        bndbox = ET.SubElement(obj, "bndbox")

        ET.SubElement(bndbox, "xmin").text = str(bbox["xmin"])
        ET.SubElement(bndbox, "ymin").text = str(bbox["ymin"])
        ET.SubElement(bndbox, "xmax").text = str(bbox["xmax"])
        ET.SubElement(bndbox, "ymax").text = str(bbox["ymax"])

    tree = ET.ElementTree(annotation)
    xml_file = os.path.join(output_dir, image_name.replace(".png", ".xml"))
    tree.write(xml_file, encoding="utf-8", xml_declaration=True)


# ==============================
# 主程序
# ==============================
def main():
    world = None
    original_settings = None
    traffic_manager = None
    vehicle = None
    camera = None
    background_vehicles = []

    image_queue = queue.Queue(maxsize=50)

    try:
        # 加载地图
        world = load_map(MAP_NAME)
        print(f"地图已加载: {MAP_NAME}")

        # 保存原始设置，退出时恢复，避免 CARLA 下次运行卡在同步模式
        original_settings = world.get_settings()

        # 同步模式
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        # Traffic Manager
        traffic_manager = client.get_trafficmanager(TM_PORT)
        traffic_manager.set_synchronous_mode(True)

        spawn_points = world.get_map().get_spawn_points()

        # 先生成主车，再生成背景车，降低主车生成失败概率
        vehicle = spawn_ego_vehicle(world, spawn_points, "vehicle.audi.a2")

        # 默认自动驾驶
        set_autopilot_mode(vehicle, traffic_manager, True)

        # 遵守红绿灯
        traffic_manager.ignore_lights_percentage(vehicle, 0.0)

        # 负值表示比限速更快，原代码是 -50
        traffic_manager.vehicle_percentage_speed_difference(vehicle, -50)

        # 背景车辆
        background_vehicles = spawn_background_vehicles(
            NUM_BACKGROUND_VEHICLES,
            world,
            spawn_points,
            traffic_manager
        )

        # 相机
        bp_lib = world.get_blueprint_library()
        camera_bp = bp_lib.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(IMAGE_W))
        camera_bp.set_attribute("image_size_y", str(IMAGE_H))
        camera_bp.set_attribute("fov", str(CAMERA_FOV))

        camera_init_trans = carla.Transform(
            carla.Location(x=1.0, z=2.0),
            carla.Rotation(pitch=-3.0)
        )

        camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)

        def image_callback(image):
            if not image_queue.full():
                image_queue.put(image)

        camera.listen(image_callback)

        image_w = camera_bp.get_attribute("image_size_x").as_int()
        image_h = camera_bp.get_attribute("image_size_y").as_int()
        fov = camera_bp.get_attribute("fov").as_float()
        K = build_projection_matrix(image_w, image_h, fov)

        # 保存路径：保持与你原代码一致，向上三级到项目根目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
        output_dir = os.path.join(project_root, "OutPut", "data01")
        os.makedirs(output_dir, exist_ok=True)
        print(f"数据保存路径: {output_dir}")

        last_weather_change_time = time.time()
        current_condition_index = 0

        print("程序启动成功")
        print("按 M 切换 AUTO / MANUAL")
        print("手动模式：W 前进，S 刹车，A 左转，D 右转，R 切换倒车，X 退出")

        while True:
            world.tick()

            running = handle_input(vehicle, traffic_manager)
            if not running:
                break

            try:
                image = image_queue.get(timeout=2.0)
            except queue.Empty:
                print("警告：未收到相机图像，跳过当前帧。")
                continue

            # CARLA raw_data 通常为 BGRA，OpenCV 使用 BGR，取前三通道即可
            img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
            img_bgr = img[:, :, :3].astype(np.uint8)

            # 违章检测：改进后的红灯逻辑在这里执行
            detect_violations(vehicle)

            # 天气切换
            current_time = time.time()
            if current_time - last_weather_change_time > WEATHER_TRANSITION_INTERVAL:
                current_condition = weather_conditions[current_condition_index]
                update_weather(world, current_condition)
                print(f"天气切换为: {current_condition}")

                current_condition_index = (current_condition_index + 1) % len(weather_conditions)
                last_weather_change_time = current_time

            # 交通标志投影框
            world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

            bboxes = get_signs_bounding_boxes(
                world,
                vehicle.get_transform(),
                camera.get_transform(),
                K,
                world_2_camera,
                image_w,
                image_h
            )

            bboxes = non_maximum_suppression(bboxes)

            img_show = img_bgr.copy()

            if bboxes:
                for box in bboxes:
                    cv2.rectangle(
                        img_show,
                        (box["xmin"], box["ymin"]),
                        (box["xmax"], box["ymax"]),
                        (0, 0, 255),
                        2
                    )

                draw_violation_info(img_show)

                image_name = f"image_{int(time.time() * 1000)}.png"

                # 保存原图，不保存叠加文字和框后的图
                cv2.imwrite(os.path.join(output_dir, image_name), img_bgr)

                create_xml_file(
                    output_dir,
                    image_name,
                    bboxes,
                    image_w,
                    image_h,
                    get_weather_params(world)
                )
            else:
                draw_violation_info(img_show)

            cv2.imshow("CARLA - Traffic Sign and Violation Detection", img_show)

            if cv2.waitKey(10) & 0xFF == ord("x"):
                print("X key pressed in OpenCV window.")
                break

    finally:
        print("正在清理资源...")

        try:
            if camera is not None:
                camera.stop()
                camera.destroy()
        except Exception as e:
            print(f"camera 清理异常: {e}")

        try:
            if vehicle is not None:
                vehicle.destroy()
        except Exception as e:
            print(f"vehicle 清理异常: {e}")

        for v in background_vehicles:
            try:
                v.destroy()
            except Exception:
                pass

        try:
            if traffic_manager is not None:
                traffic_manager.set_synchronous_mode(False)
        except Exception as e:
            print(f"traffic_manager 恢复异常: {e}")

        try:
            if world is not None and original_settings is not None:
                world.apply_settings(original_settings)
        except Exception as e:
            print(f"world settings 恢复异常: {e}")

        cv2.destroyAllWindows()
        pygame.quit()

        print("程序已退出，资源已清理。")


if __name__ == "__main__":
    main()
