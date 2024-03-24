# importing librarys
import cv2
import numpy as npy
import face_recognition as face_rec
# function
def resize(img, size) :
    width = int(img.shape[1]*size)
    height = int(img.shape[0] * size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation= cv2.INTER_AREA)


# img declaration
abhijeet = face_rec.load_image_file('sampe_images\abhijeet.jpg')
abhijeet = cv2.cvtColor(abhijeet, cv2.COLOR_BGR2RGB)
abhijeet = resize(abhijeet, 0.50)
abhijeet_test = face_rec.load_image_file('sampe_images\elonmusk.jpg')
abhijeet_test = resize(abhijeet_test, 0.50)
abhijeet_test = cv2.cvtColor(abhijeet_test, cv2.COLOR_BGR2RGB)

# finding face location

faceLocation_abhijeet = face_rec.face_locations(abhijeet)[0]
encode_abhijeet = face_rec.face_encodings(abhijeet)[0]
cv2.rectangle(abhijeet, (faceLocation_abhijeet[3], faceLocation_abhijeet[0]), (faceLocation_abhijeet[1], faceLocation_abhijeet[2]), (255, 0, 255), 3)


faceLocation_abhijeettest = face_rec.face_locations(abhijeet_test)[0]
encode_abhijeettest = face_rec.face_encodings(abhijeet_test)[0]
cv2.rectangle(abhijeet_test, (faceLocation_abhijeet[3], faceLocation_abhijeet[0]), (faceLocation_abhijeet[1], faceLocation_abhijeet[2]), (255, 0, 255), 3)

results = face_rec.compare_faces([encode_abhijeet], encode_abhijeettest)
print(results)
cv2.putText(abhijeet_test, f'{results}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255), 2 )

cv2.imshow('main_img', abhijeet)
cv2.imshow('test_img', abhijeet_test)
cv2.waitKey(0)
cv2.destroyAllWindows()
