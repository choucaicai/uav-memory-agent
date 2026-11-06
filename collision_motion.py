"""Collision-aware discrete AirSim movement; no model or evaluator dependencies."""
from __future__ import annotations

import math
import time

import airsim


def finite_float(value):
    value = float(value)
    return value if math.isfinite(value) else None


def vector_json(value):
    return [finite_float(getattr(value, axis)) for axis in ('x_val', 'y_val', 'z_val')]


def collision_json(info):
    return {
        'has_collided': bool(info.has_collided), 'time_stamp': int(info.time_stamp),
        'object_name': str(info.object_name), 'object_id': int(info.object_id),
        'position': vector_json(info.position), 'impact_point': vector_json(info.impact_point),
        'normal': vector_json(info.normal), 'penetration_depth': finite_float(info.penetration_depth),
    }


class CollisionTracker:
    """AirSim collision info is sticky; count changes, not repeated reads of has_collided."""
    def __init__(self):
        self.last_event = None

    @staticmethod
    def event_key(info):
        if not info.has_collided:
            return None
        return (int(info.time_stamp), str(info.object_name), int(info.object_id),
                tuple(vector_json(info.impact_point)), tuple(vector_json(info.position)))

    def reset(self, info):
        self.last_event = self.event_key(info)

    def consume(self, info):
        key = self.event_key(info)
        if key is None or key == self.last_event:
            return None
        self.last_event = key
        return collision_json(info)


def read_pose(client):
    pose = client.simGetVehiclePose()
    xyz = vector_json(pose.position)
    yaw = float(airsim.to_eularian_angles(pose.orientation)[2])
    if any(value is None for value in xyz) or not math.isfinite(yaw):
        raise ValueError('AirSim returned a non-finite vehicle pose')
    return [*xyz, yaw]


def airsim_pose(pose):
    x, y, z, yaw = map(float, pose)
    return airsim.Pose(airsim.Vector3r(x, y, z), airsim.to_quaternion(0., 0., yaw))


def yaw_difference(target, source):
    return (target - source + math.pi) % (2 * math.pi) - math.pi


def reached(actual, target, position_tolerance, yaw_tolerance):
    distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(actual[:3], target[:3])))
    return distance <= position_tolerance and abs(yaw_difference(actual[3], target[3])) <= yaw_tolerance


def waypoints(start, target, max_distance, max_yaw_degrees):
    distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(start[:3], target[:3])))
    delta_yaw = yaw_difference(target[3], start[3])
    count = max(1, math.ceil(distance / max_distance), math.ceil(abs(delta_yaw) / math.radians(max_yaw_degrees)))
    for index in range(1, count + 1):
        alpha = index / count
        yield [*(a + (b - a) * alpha for a, b in zip(start[:3], target[:3])),
               (start[3] + delta_yaw * alpha + math.pi) % (2 * math.pi) - math.pi]


class CollisionMotion:
    def __init__(self, client, *, substep_m=0.25, yaw_step_degrees=5.0,
                 poll_seconds=0.02, timeout_seconds=3.0, settle_seconds=0.1,
                 position_tolerance=0.03, yaw_tolerance_degrees=1.0,
                 placement_attempts=5, placement_retry_seconds=1.0):
        if min(substep_m, yaw_step_degrees, poll_seconds, timeout_seconds, position_tolerance,
               yaw_tolerance_degrees) <= 0 or settle_seconds < 0:
            raise ValueError('Invalid collision movement parameters')
        if placement_attempts < 1 or placement_retry_seconds < 0:
            raise ValueError('Invalid placement retry parameters')
        self.placement_attempts = int(placement_attempts)
        self.placement_retry_seconds = placement_retry_seconds
        self.client = client
        self.substep_m = substep_m
        self.yaw_step_degrees = yaw_step_degrees
        self.poll_seconds = poll_seconds
        self.timeout_seconds = timeout_seconds
        self.settle_seconds = settle_seconds
        self.position_tolerance = position_tolerance
        self.yaw_tolerance = math.radians(yaw_tolerance_degrees)
        self.tracker = CollisionTracker()

    def begin_episode(self, start):
        # AirSim can push the vehicle away right after reset/teleport. Retry
        # from a fresh reset; success criteria are unchanged.
        failures = []
        for attempt in range(1, self.placement_attempts + 1):
            placed, actual = self._place(start)
            if placed:
                if failures:
                    print(f'PLACEMENT_RECOVERED attempt={attempt}/{self.placement_attempts} '
                          f'requested={list(start)} failed_poses={failures}', flush=True)
                self.tracker.reset(self.client.simGetCollisionInfo())
                return read_pose(self.client)
            failures.append(actual)
            print(f'PLACEMENT_RETRY attempt={attempt}/{self.placement_attempts} '
                  f'requested={list(start)} actual={actual}', flush=True)
            time.sleep(self.placement_retry_seconds * attempt)
        raise TimeoutError(f'Unable to place vehicle at episode start after {self.placement_attempts} '
                           f'attempts: requested={start}, actual={failures}')

    def _place(self, start):
        # Reset old collision state and external-physics kinematics before placing
        # the drone. Only placement is allowed to teleport through geometry.
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.simSetVehiclePose(airsim_pose(start), ignore_collision=True)
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            time.sleep(self.poll_seconds)
            actual = read_pose(self.client)
            if reached(actual, start, self.position_tolerance, self.yaw_tolerance):
                break
            if time.monotonic() >= deadline:
                return False, actual
        time.sleep(self.settle_seconds)
        return True, actual

    def poll_collision(self):
        return self.tracker.consume(self.client.simGetCollisionInfo())

    def move_to(self, target):
        actual = read_pose(self.client)
        trace = []
        event = self.poll_collision()
        if event:
            return {'pose': actual, 'trace': trace, 'collision': event, 'motion_timeout': False}
        # Unreal's collision sweep plus short substeps prevents stepping through
        # geometry on 3/6/9 metre discrete actions. Never write the desired pose
        # into the evaluation trajectory; always read the achieved pose back.
        for waypoint in waypoints(actual, target, self.substep_m, self.yaw_step_degrees):
            self.client.simSetVehiclePose(airsim_pose(waypoint), ignore_collision=False)
            deadline = time.monotonic() + self.timeout_seconds
            while True:
                time.sleep(self.poll_seconds)
                actual = read_pose(self.client)
                event = self.poll_collision()
                if event:
                    trace.append(actual)
                    return {'pose': actual, 'trace': trace, 'collision': event, 'motion_timeout': False}
                if reached(actual, waypoint, self.position_tolerance, self.yaw_tolerance):
                    trace.append(actual)
                    break
                if time.monotonic() >= deadline:
                    trace.append(actual)
                    return {'pose': actual, 'trace': trace, 'collision': None, 'motion_timeout': True}
        # Give a delayed rendering/collision callback time to arrive, including
        # for STOP/rotation actions and contact at the final waypoint.
        deadline = time.monotonic() + self.settle_seconds
        while True:
            actual = read_pose(self.client)
            event = self.poll_collision()
            if event or time.monotonic() >= deadline:
                trace.append(actual)
                return {'pose': actual, 'trace': trace, 'collision': event, 'motion_timeout': False}
            time.sleep(self.poll_seconds)
