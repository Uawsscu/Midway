import freenect
import cv2
import numpy as np
import time
from espeak import espeak
espeak.set_parameter(espeak.Parameter.Pitch, 60)
espeak.set_parameter(espeak.Parameter.Rate, 110)
espeak.set_parameter(espeak.Parameter.Range, 600)

def get_video():
    array, _ = freenect.sync_get_video()
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    return array

def Detect(objectName,type):
    start = time.time()
    time.clock()
    elapsed = 0
    seconds = 5  # 20 S.
    MIN_MATCH_COUNT = 30
    try :
        setpath = '/home/uawsscu/PycharmProjects/project3/image/' + objectName + '.jpg'
        detector = cv2.SIFT()

        FLANN_INDEX_KDITREE = 0
        flannParam = dict(algorithm=FLANN_INDEX_KDITREE, tree=5)
        flann = cv2.FlannBasedMatcher(flannParam, {})

        trainImg = cv2.imread(setpath, 0)
        trainKP, trainDesc = detector.detectAndCompute(trainImg, None)
    except :
        print ".."


    QUESTION_COUNT = 0
    goodCount = 0
    while elapsed < seconds:
        try :
            QueryImgBGR = get_video()

            QueryImg = cv2.cvtColor(QueryImgBGR, cv2.COLOR_BGR2GRAY)
            queryKP, queryDesc = detector.detectAndCompute(QueryImg, None)
            matches = flann.knnMatch(queryDesc, trainDesc, k=2)

            goodMatch = []
            for m, n in matches:
                if m.distance < 0.65 * n.distance:
                    goodMatch.append(m)
            if len(goodMatch) > MIN_MATCH_COUNT:
                if type == "command":
                    goodCount = 1

                tp = []
                qp = []
                for m in goodMatch:
                    tp.append(trainKP[m.trainIdx].pt)
                    qp.append(queryKP[m.queryIdx].pt)
                tp, qp = np.float32((tp, qp))
                H, status = cv2.findHomography(tp, qp, cv2.RANSAC, 3.0)
                h, w = trainImg.shape
                trainBorder = np.float32([[[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]])
                queryBorder = cv2.perspectiveTransform(trainBorder, H)
                cv2.polylines(QueryImgBGR, [np.int32(queryBorder)], True, (0, 255, 0), 5)

            else:
                print "Not Enough -->> %d : %d" % (len(goodMatch), MIN_MATCH_COUNT)

            QueryImgBGR = cv2.drawKeypoints(QueryImgBGR, queryKP)
            cv2.imshow('result', QueryImgBGR)

            if cv2.waitKey(10) == ord('q'):
                break
        except :
            goodCount = -1
            espeak.synth("No , I don't know")
            time.sleep(2)
            print "..."
            break
        if goodCount != -1 and type == "question":
            espeak.synth("Yes, I know")
            time.sleep(1.5)
            break

        elapsed = time.time() - start
        time.sleep(1)

    if goodCount == 0 and type == "command":
        print goodCount

        espeak.synth("I can't see.")
        time.sleep(1.5)

    cv2.destroyAllWindows()

#Detect("ballcc", 'question')
#Detect("ball", 'command')

