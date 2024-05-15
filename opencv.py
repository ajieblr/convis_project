import cv2

clasifier = cv2.CascadeClassifier('haarcasecade\haarcascade_lowerbody.xml')

camera = cv2.VideoCapture('test2.mp4')
while True:
    ret, img = camera.read()
    blur = cv2.blur(img,(3,3))
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    full_body = clasifier.detectMultiScale(gray)
    # full_body = clasifier.detectMultiScale(gray)

    for (x,y,w,h) in full_body:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0),2)
        cv2.putText(img, 'person', (x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0), 2)
    cv2.imshow('Live', img)
    key =  cv2.waitKey(1)

    if key == 27:
        break
cv2.destroyAllWindows()
camera.release()