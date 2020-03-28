# USAGE
# python recognize_faces_video.py --encodings encodings.pickle
# python recognize_faces_video.py --encodings encodings.pickle --output output/jurassic_park_trailer_output.avi --display 0
# python recognize_faces_video.py --encodings encodings.pickle \
#	--output output/webcam_face_recognition_output.avi --display 1

# import the necessary packages
from collections import deque
from imutils.video import VideoStream
from imutils.face_utils import rect_to_bb
import face_recognition
import argparse
import imutils
import pickle
import time
import numpy as np
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-o", "--output", type=str,
	help="path to output video")
ap.add_argument("-y", "--display", type=int, default=1,
	help="whether or not to display output frame to screen")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

pts = deque(maxlen=args["buffer"])
counter = 0
(dX, dY) = (0, 0)
direction = ""

# allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
writer = None
time.sleep(2.0)

# loop sobre quadros do fluxo de arquivos de vídeo
while True:
	# pegue o quadro do fluxo de vídeo encadeado
	frame = vs.read()
	
	# converta o quadro de entrada de BGR para RGB e redimensione-o para ter
	# uma largura de 750px (para acelerar o processamento)
	
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	rgb = imutils.resize(frame, width=750)
	r = frame.shape[1] / float(rgb.shape[1])
	# calcula varias coordenas de posicionamento x,y em pixels na tela do opencv
	(height, width) = frame.shape[:2]
	startX = int(height/height)
	startY = int(width/width)
	centerScreenX = int(width/2)#Coordenada X do centro da tela
	centerScreenY = int(height/2)#Coordenada Y do centro da tela
	top_left_x = int (width / 3)# Y=360 CIANO                   		|480
	top_left_y = int ((height / 2) + (height / 4))# X=213 CIANO 		|640
	bottom_right_x = int ((width / 3) * 2)# Y=120 AMARELO 				|480
	bottom_right_y = int ((height / 2) - (height / 4))# X=426 AMARELO	|640
	# converte em string
	max_w_str = str(width)
	max_h_str = str(height) 
	top_left_y_string = str(top_left_y)
	top_left_x_string = str(top_left_x)
	bottom_right_x_string = str(bottom_right_x)
	bottom_right_y_string = str(bottom_right_y)
	# desenha na tela
	cv2.circle(frame,(width,height),10, (0,21,255),-2)
	cv2.putText(frame, "dx: {}, dy: {}".format(max_w_str, max_h_str),(frame.shape[0]+40, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,0.40, (0,21,255), 1)
	cv2.rectangle(frame, (top_left_x,top_left_y), (bottom_right_x,bottom_right_y), 255, 1)
	cv2.circle(frame,(centerScreenX,centerScreenY),4, (255),-2) #Printa na tela o centro do quadro
	cv2.circle(frame,(top_left_x,top_left_y),4, (128,0,128),-2)#CIANO
	cv2.circle(frame,(bottom_right_x,bottom_right_y),4, (0,255,255),-2)#AMARELO

	cv2.putText(frame, top_left_x_string, (top_left_x, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (128,0,128), 1)
	cv2.putText(frame, top_left_y_string, (0, top_left_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (128,0,128), 1)
	cv2.putText(frame, bottom_right_x_string, (bottom_right_x, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0,255,255), 1)
	cv2.putText(frame, bottom_right_y_string, (0, bottom_right_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0,255,255), 1)

	# detecta as coordenadas (x, y) das caixas delimitadoras
	# correspondente a cada face no quadro de entrada e depois calcule
	# os tratamentos faciais para cada rosto
	boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
	encodings = face_recognition.face_encodings(rgb, boxes)
	center = None
	names = []	

	# loop sobre os revestimentos faciais  
	for encoding in encodings:	
	
		# tente combinar cada rosto na imagem de entrada com o nosso conhecido
		# codificações
		matches = face_recognition.compare_faces(data["encodings"],
			encoding)
		name = "Unknown"

		# verifique se encontramos uma correspondência
		if True in matches:

			# encontre os índices de todas as faces correspondentes e inicialize um
			# dictionary para contar o número total de vezes que cada face
			# foi correspondido
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}

			# faz um loop sobre os índices correspondentes e mantém uma contagem para
			# cada rosto reconhecido
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1

			# determinar a face reconhecida com o maior número
			# Nº de votos (nota: no caso de um empate improvável em Python
			# seleciona a primeira entrada no dicionário)
			name = max(counts, key=counts.get)
		
		# atualiza a lista de nomes
		names.append(name)

	# passe pelas faces reconhecidas
	for ((top, right, bottom, left), name) in zip(boxes, names):

		# redimensionar as coordenadas do rosto
		top = int(top * r)
		right = int(right * r)
		bottom = int(bottom * r)
		left = int(left * r)
		#Pontos X e Y do centro do quadro da face
		cx = int((left + right)/2.0)
		cy = int((top + bottom)/2.0)
		center = (cx, cy)
		cx_str = str(cx)
		cy_str = str(cy)

		# desenha o nome do rosto previsto na imagem
		cv2.rectangle(frame, (left, top), (right, bottom),
			(0, 255, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			0.75, (0, 255, 0), 2)
		cv2.circle(frame,(cx,cy),4, (0, 255, 0),-2) #Printa na tela o centro do quadro
		cv2.putText(frame, "dx: {}, dy: {}".format(cx_str, cy_str),(400, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,0.40, (0, 255, 0), 1)
		cv2.arrowedLine(frame,(centerScreenX,centerScreenY),(cx,cy), (255, 255, 255),1, 8, 0, 0.1)
		pts.appendleft(center)

		# calcula as coordenadas x e y em relação aos eixos centrais da imagem
		if cx <= centerScreenX and cy <= centerScreenY:
			auxX = int(centerScreenX-cx)
			auxY = int(centerScreenY-cy)
		elif cx >= centerScreenX and cy <= centerScreenY:
			auxX = int(-1*(cx-centerScreenX))
			auxY = int(centerScreenY-cy)
		elif cx >= centerScreenX and cy >= centerScreenY:
			auxX = int(-1*(cx-centerScreenX))
			auxY = int(-1*(cy-centerScreenY))
		elif cx <= centerScreenX and cy >= centerScreenY:
			auxX = int(centerScreenX-cx)
			auxY = int(-1*(cy-centerScreenY))
		else:
			auxX = 0
			auxY = 0
		position = (auxX, auxY)
		print(position)

		# verifica em qual regiao de interesse o drone esta
		if cx in range(startX,top_left_x) and cy in range(startY,bottom_right_y):
			print("NOROESTE")
		elif cx in range(top_left_x,bottom_right_x) and cy in range(startY,bottom_right_y):
			print("NORTE")
		elif cx in range(bottom_right_x,width) and cy in range(startY,bottom_right_y):
			print("NORDESTE")
		elif cx in range(startX,top_left_x) and cy in range(bottom_right_y,top_left_y):
			print("OESTE")
		elif cx in range(bottom_right_x,width) and cy in range(bottom_right_y,top_left_y):
			print("LESTE")
		elif cx in range(startX,top_left_x) and cy in range(top_left_y,height):
			print("SUDOESTE")
		elif cx in range(top_left_x,bottom_right_x) and cy in range(top_left_y,height):
			print("SUL")
		elif cx in range(bottom_right_x,width) and cy in range(top_left_y,height):
			print("SUDESTE")
		else:
			pass
	for i in np.arange(1, len(pts)):
		# se um dos pontos rastreados for Nenhum, ignore
		# eles
		if pts[i - 1] is None or pts[i] is None:
			continue
		# verifique se foram acumulados pontos suficientes no
		# o buffer
		if counter >= 10 and i == 1 and pts[-10] is not None:
			# calcular a diferença entre x e y
			# coordena e reinicializa a direção
			# variáveis de texto
			dX = pts[-10][0] - pts[i][0]
			dY = pts[-10][1] - pts[i][1]
			(dirX, dirY) = ("", "")
			# garantir que haja movimento significativo no
			# direção x
			if np.abs(dX) > 10:
				dirX = "East" if np.sign(dX) == 1 else "West"
			# garantir que haja movimento significativo no
			# direção y
			if np.abs(dY) > 10:
				dirY = "North" if np.sign(dY) == 1 else "South"
			# manipula quando as duas direções não estão vazias
			# if dirX != "" and dirY != "":
			# 	direction = "{}-{}".format(dirY, dirX)
			# # caso contrário, apenas uma direção não estará vazia
			# else:
			direction = dirX if dirX != "" else dirY
		# thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
		# cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

	cv2.putText(frame, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 3)
	cv2.putText(frame, "dx: {}, dy: {}".format(dX, dY),
		(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
		0.35, (255, 255, 255), 1)
	counter += 1
	# se o gravador de vídeo for None * AND *, devemos escrever
	# o vídeo de saída em disco inicializa o gravador
	if writer is None and args["output"] is not None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 20,
			(frame.shape[1], frame.shape[0]), True)
	# se o gravador não for None, escreva o quadro com os
	# faces no disco
	if writer is not None:
		writer.write(frame)
	# verifique se devemos exibir o quadro de saída para
	# a tela
	if args["display"] > 0:
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
		# se a tecla `q` foi pressionada, interrompa o loop
		if key == ord("q"):
			break
# faça um pouco de limpeza
cv2.destroyAllWindows()
vs.stop()
# verifique se o ponto do gravador de vídeo precisa ser liberado
if writer is not None:
	writer.release()