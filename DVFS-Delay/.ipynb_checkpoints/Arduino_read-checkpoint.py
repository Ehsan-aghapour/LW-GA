
# https://forum.arduino.cc/t/demo-of-pc-arduino-comms-using-python/219184/11
# 19 July 2014
# 08 Dec 2016 - updated for Python3



# =====================================

#  Function Definitions

# =====================================

def sendToArduino(sendStr):
	ser.write(sendStr.encode('utf-8')) # change for Python3


# ======================================

def ser_open():
	global ser
	ser = serial.Serial(serPort, baudRate)
	print ("Serial port " + serPort + " opened  Baudrate " + str(baudRate))

def ser_close():
	ser.close()



# ======================================

# THE DEMO PROGRAM STARTS HERE

# ======================================

import serial
import time

import signal
import sys
import threading

# print ()
# print ()

# NOTE the user must ensure that the serial port and baudrate are correct
# serPort = "/dev/ttyS80"
serPort = "/dev/ttyACM0"
baudRate = 115200





def handler(signum, frame):
	#res = input("Ctrl-c was pressed. Do you really want to exit? y/n ")
	f.close()
	ser.flush()
	ser.close()
	#print()
	print('closing file and serial port\n')
	exit(1)

signal.signal(signal.SIGINT, handler)

def run(file_name):
	ser_open()
	global f
	f=open(file_name,'w')
	t = threading.currentThread()
	#while True:	
	print("Start Monitoring Power...")
	while getattr(t, "do_run", True):
		try:
			r=ser.readline()
            print(r)
		except:
			print(f'Arduino read error probably device disconnected (restarted maybe)')
			f.close()
			#ser.flush()
			ser.close()
			#print()
			print('closing file and serial port')
			return -1
		try:
			if r.decode().strip()[0] == '1':
				f.write(r.decode())
				break
			else:
				continue
		except:
			continue
	while getattr(t, "do_run", True):		
		try:
			r=ser.readline()
		except:
			print(f'Arduino read error probably device disconnected (restarted maybe)')
			f.close()
			#ser.flush()
			ser.close()
			#print()
			print('closing file and serial port')
			return -1;
		try:
			#print(r.decode())
			f.write(r.decode())
		except UnicodeDecodeError:
			continue
	f.close()
	ser.flush()
	ser.close()
	#print()
	print('closing file and serial port')







if __name__ == "__main__":
	print("start monitoring...\n")
	if (len(sys.argv) > 1):
		file_name='data.csv'
		file_name=f'data_{sys.argv[1]}.csv'
	run(file_name)

