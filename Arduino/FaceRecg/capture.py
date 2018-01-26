import freenect
import cv2

def get_video():
    array, _ = freenect.sync_get_video()
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    #array = cv2.cvtColor(array,  cv2.COLOR_BGR2GRAY)
    return array

def set_path(namePic):
    setPath = '/home/uawsscu/PycharmProjects/Project3/image/' + namePic + '.jpg'
    return setPath

def cap_ture(pathPic):
    i=0


    while 1:

        setPath = '/home/uawsscu/PycharmProjects/project3/Arduino/FaceRecg/dataSet/' + pathPic + str(i)+'.jpg'
        print "ok"
        frame = get_video()


        params = list()
        crop_img = frame[250:350, 290:390]  # Crop from x, y, w, h -> 100, 200, 300, 400
        # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
        cv2.imwrite(setPath, crop_img, params)
        i+=1
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        elif i > 20:
            break

        cv2.imshow('RGB image', frame)

        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break

    cv2.destroyAllWindows()

cap_ture('mess2')