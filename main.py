import os
import cv2
import face_recognition

known_face_encodings = []  # Lista com os encodings dos rostos conhecidos
known_face_names = []  # Lista com os nomes dos rostos conhecidos

# Função para carregar as imagens de pessoas conhecidas
def load_images():
    for filename in os.listdir('pessoas'):
        # Carregar a imagem
        image = face_recognition.load_image_file('pessoas/' + filename)
        # Extrair o encoding do rosto
        face_encoding = face_recognition.face_encodings(image)[0]
        # Adicionar o encoding e o nome da pessoa
        known_face_encodings.append(face_encoding)
        known_face_names.append(filename.split('.')[0])


# Função principal
def init():
    load_images()  # Carregar as imagens de pessoas conhecidas
    video_capture = cv2.VideoCapture(2) # Inicializar a câmera

    while True:
        ret, frame = video_capture.read() # Capturar cada frame da câmera
        frame = cv2.flip(frame, 1)  # Espelhar o frame horizontalmente

        # Encontrar rostos no frame
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        # Loop sobre cada rosto encontrado no frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Verificar se o rosto é um rosto conhecido
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Desconhecido"  # Se não for um rosto conhecido, então o nome será "Desconhecido"

            # Se um rosto conhecido for encontrado, use o primeiro
            # (Pode ajustar isso para mostrar mais informações se houver mais de um rosto conhecido)
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            # Desenhar um retângulo em torno do rosto e exibir o nome
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        cv2.imshow('Video', frame)  # Exibir o frame com o retângulo e o nome

        # Se pressionar 'q', sair do loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# Iniciar o programa
init()
