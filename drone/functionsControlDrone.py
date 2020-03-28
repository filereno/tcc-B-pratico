##########DEPENDENCIES#############
from dronekit import connect, VehicleMode,LocationGlobalRelative,APIException
import time
import socket
import math
import argparse
from pymavlink import mavutil
#########FUNCTIONS#################

def connectMyCopter():

    parser = argparse.ArgumentParser(description='commands')
    parser.add_argument('--connect')
    args = parser.parse_args()

    connection_string = args.connect

    if not connection_string:
        import dronekit_sitl
        sitl = dronekit_sitl.start_default()
        connection_string = sitl.connection_string()

    vehicle = connect(connection_string,wait_ready=True)

    return vehicle

def arm_and_takeoff(vehicle, targetHeight):
    while vehicle.is_armable!=True:
        print("Esperando o veiculo se armar")
        time.sleep(1)
    print("Veiculo armado")

    vehicle.mode = VehicleMode("GUIDED")

    while vehicle.mode!='GUIDED':
        print("Aguardando entrar em modo GUIDED")
        time.sleep(1)
    print("Veiculo em modo GUIDED")

    vehicle.armed = True
    while vehicle.armed==False:
        print("Esperando o veiculo se armar")
        time.sleep(1)
    print("Cuidado as helices virtuais estao em funcionamento")

    vehicle.simple_takeoff(targetHeight) ##meters


    while True:
        print("Current Altitude: %d"%vehicle.location.global_relative_frame.alt, targetHeight)

        if vehicle.location.global_relative_frame.alt>=.92*targetHeight:
            break
        time.sleep(1)
    print("Target altitude reached!!")
    return None

def send_local_ned_velocity(vehicle, vx, vy, vz):
    """
    Move vehicle in direction based on specified velocity vectors.
    """
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,       # time_boot_ms (not used)
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED, # frame
        0b0000111111000111, # type_mask (only speeds enabled)
        0, 0, 0, # x, y, z positions (not used)
        vx, vy, vz, # x, y, z velocity in m/s
        0, 0, 0, # x, y, z acceleration (not supported yet, ignored in GCS_Mavlink)
        0, 0)    # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink)

    # send command to vehicle on 1 Hz cycle
    print(" Airspeed: %s" % vehicle.airspeed)
    vehicle.send_mavlink(msg)
    vehicle.flush()


def send_global_ned_velocity(vehicle, vx, vy, vz):
    """
    Move vehicle in direction based on specified velocity vectors.
    """
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,       # time_boot_ms (not used)
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_FRAME_LOCAL_NED, # frame
        0b0000111111000111, # type_mask (only speeds enabled)
        0, 0, 0, # x, y, z positions (not used)
        vx, vy, vz, # x, y, z velocity in m/s
        0, 0, 0, # x, y, z acceleration (not supported yet, ignored in GCS_Mavlink)
        0, 0)    # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink)

    # send command to vehicle on 1 Hz cycle
    vehicle.send_mavlink(msg)
    vehicle.flush()


