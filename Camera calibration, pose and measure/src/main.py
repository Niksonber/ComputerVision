## Adaptado por Nikson Bernardes
## fonte original https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
##                https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_pose/py_pose.html#pose-estimation

import numpy as np
import sys
import glob

ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path )
import cv2


REPEATS = 1


class ImageAndOperations:
    def __init__(self, *name):
        self.numOfClicks = -1
        self.point0 = None
        self.point1 = None
        self.image = None
        self.gray = None
        self.objpoints = [] # pontos no mundo real
        self.imgpoints = [] # pontos no plano da imagem
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.name = str(name)
        self.window = cv2.namedWindow(self.name)
        self._printInMm = False
        self.extrinsic = None
        self.intrisic = None
        cv2.setMouseCallback(self.name, self.callback)
    @property
    def printInMm(self):
        return self._printInMm
    def printInMm_set(self, value, extrinsic, intrisic):
        self._printInMm = value
        self.extrinsic = extrinsic
        self.intrisic = intrisic
    def updateImage(self, image):
        self.image = image
        cv2.imshow(self.name, self.image)
        #cv2.waitKey(2000)
    def updateObjPoints(self, objp):
        self.gray = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(self.gray, (8,6),None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            self.objpoints.append(objp)
            corners2 = cv2.cornerSubPix(self.gray,corners,(11,11),(-1,-1),self.criteria)
            self.imgpoints.append(corners2)
            # Draw and display the corners
            self.image = cv2.drawChessboardCorners(self.image, (7,6), corners2, ret)
            #cv2.imshow(self.name, image)
            #cv2.waitKey(0)
    def clearObjPoints(self):
        self.objpoints = [] # 3d point in real world space
        self.imgpoints = [] # 2d points in image plane.
    def callback(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.numOfClicks = (self.numOfClicks+1)%2
            if self.numOfClicks ==1:
                self.point1 = np.array([x,y])
                print("Distancia em pixels para ", self.name, ": ")
                print(self.calcDistPixels())
                if self.printInMm:    
                    print("Distancia em cm para ", self.name, ": ")
                    print(self.calcDistMm(self.intrisic, self.extrinsic))
                self.drawLine()
                print('-------------------------------------------------------')
            else:
                self.point0 = np.array([x,y])

    def drawLine(self):
        cv2.line(self.image, tuple(self.point0),tuple(self.point1), (255, 0, 0), 3)
        cv2.imshow(self.name, self.image)
    def calcDistPixels(self):
        #point0 e point1 são arrays numpy
        return np.linalg.norm(self.point1-self.point0)
    def calcDistMm(self,intrisics, extrinsics):
        subPoints = self.point1-self.point0
        homogeniusCoord = np.asmatrix(np.append(subPoints, 1.0)).transpose()
        coord = np.linalg.pinv(extrinsics) * (np.linalg.inv(intrisics) * homogeniusCoord)
        #print(coord[:-1, :]/coord[-1, :])
        return np.linalg.norm(coord[:-1, :]/coord[-1, :]) #transforma para coordenadas cartesianas e calcula a norma

class Camera:
    def __init__(self):
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.objp = np.zeros((6*8,3), np.float32)
        self.objp[:,:2] = 3.0 *np.mgrid[0:8,0:6].T.reshape(-1,2)
        self.images = glob.glob ('../data/calibrate/*.jpg')
        self.imagesTr = glob.glob('../data/trans/*.jpg')
        self.imagesTest = glob.glob('../data/test/*.jpg')
        self.raw = ImageAndOperations("raw")
        self.undistorted = ImageAndOperations("undistorted")
        self.trans = [None, None, None]
        self.rot = [None, None, None]
    def calibrate(self):
        intrisics = np.array([None])
        dist = np.array([None])
        for i in range(REPEATS):
            for imageName in self.images:
                img = cv2.imread(imageName)
                self.raw.updateImage(img)
                self.raw.updateObjPoints(self.objp)
            ret, intrisics[i], dist[i], rvecs, tvecs = cv2.calibrateCamera(self.raw.objpoints, self.raw.imgpoints, self.raw.gray.shape[::-1],None,None)
            self.intrisics = intrisics[i]
            self.dist = dist[i]
            self.raw.clearObjPoints()
        #cv2.destroyAllWindows()
        #calcula media e desvio padrao
        self.intrisics = intrisics.sum(0)/REPEATS
        #self.dPIntrisics = np.sqrt(np.power(intrisics-self.intrisics, 2).sum(0)/(REPEATS-1))
        self.dist = dist.sum(0)/REPEATS
        #self.dPdist = np.sqrt(np.power(dist-self.dist, 2).sum(0)/(REPEATS-1))
        print("Parametros intrinsecos e desvio parao\n")
        print(self.intrisics)
        #print(self.dPIntrisics)
        print("distorcoes e desvio parao\n")
        print(self.dist)
        #print(self.dPdist)
        self.saveParam()
        self.showUndistortion()

    def undistortion(self):
        h,  w = self.raw.image.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.intrisics,self.dist,(w,h),1,(w,h))
        # undistort
        self.undistorted.updateImage(cv2.undistort(self.raw.image, self.intrisics, self.dist, None, newcameramtx))
    def saveParam(self):
        fintrisc = cv2.FileStorage("intrisics.xml", cv2.FILE_STORAGE_WRITE)
        fintrisc.write(name='intrisics', val=self.intrisics)
        fdist = cv2.FileStorage("distortion.xml", cv2.FILE_STORAGE_WRITE)
        fdist.write(name='distortion', val=self.dist)
    def showUndistortion(self):
        print('clique qualquer botao para sair')
        for imageName in self.images:
            img = cv2.imread(imageName)
            self.raw.updateImage(img)
            self.undistortion()
            cv2.waitKey(0)
            
    def draw(self, img, corners, imgpts):
        corner = tuple(corners[0].ravel())
        cv2.line(img, corner, tuple(map(int,imgpts[0].ravel())), (255,0,0), 5)
        cv2.line(img, corner, tuple(map(int,imgpts[1].ravel())), (0,255,0), 5)
        cv2.line(img, corner, tuple(map(int,imgpts[2].ravel())), (0,0,255), 5)
        
        return img

    def FindExtrinsicsForImg(self, img, imageName):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (8,6),None)
        axis = np.array([[3.0,0.,0.], [0.,3.,.0], [.0,.0,-3.]])
        axis.reshape(-1,3)
        if ret == True:
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),self.criteria)
            self.raw.updateImage(cv2.drawChessboardCorners(self.raw.image, (7,6), corners2, ret))
            self.undistortion()

            # Find the rotation and translation vectors.
            _, rvecs, tvecs, inliers = cv2.solvePnPRansac(self.objp, corners2, self.intrisics, self.dist)
            
            # project 3D points to image plane
            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, self.intrisics, self.dist)

            img = self.draw(img,corners2,imgpts)    
            k = cv2.waitKey(0) & 0xff
            if k == 's':
                cv2.imwrite(imageName[:6]+'.jpg', img)
            return rvecs, tvecs
        return None, None
    def FindExtrinsics(self):
        for i in range(3):
            images = glob.glob('../data/trans/' + str(i) + '/*.jpg')
            cv2.waitKey(0)
            count = 0
            trans = []
            for imageName in images:
                img = cv2.imread(imageName)
                self.raw.updateImage(img)
                self.undistortion()
                rot1, trans1 = self.FindExtrinsicsForImg(self.raw.image, imageName)
                trans.append(np.linalg.norm(trans1))
                if imageName==images[0]:
                    self.rot[i]=rot1
                    self.trans[i]=trans1
                else:
                    self.rot[i]+=rot1
                    self.trans[i]+=trans1
                count+=1
            trans = np.array(trans)
            self.trans[i]/=count
            self.rot[i]/=count
            print("Distancia para o d", i, " em cm: ")
            print(np.linalg.norm(self.trans[i]), " cm")
            print("Desvio padrao")
            aux = trans - np.linalg.norm(self.trans[i])
            print(np.sqrt(np.power(aux,2).sum()/(aux.size-1)))
            print("--------------------------")
            #print(np.sqrt(np.power(trans-np.linalg.norm(self.trans[i]),2).sum()/trans.size))
            cv2.waitKey(0)
    def measure(self):
        op = int(input('Selecione a distancia (dmin, dint, dmax): 0-1-2:'))
        images = glob.glob('../data/test/' + str(op) + '/*.jpg')
        rotation, jac = cv2.Rodrigues(self.rot[op])
        extrinsics = np.append(rotation, self.trans[op], 1)
        #extrinsics = np.append(extrinsics, np.matrix([0., 0., 0., 1.]), 0)
        for imageName in images:
            self.raw.printInMm_set(True, extrinsics,  self.intrisics)
            self.undistorted.printInMm_set(True, extrinsics,  self.intrisics)
            img = cv2.imread(imageName)
            self.raw.updateImage(img)
            self.undistortion()
            cv2.waitKey(0)

if __name__ == "__main__":
    camera= Camera()
    print('Calibrando a partir de imagens em ../data/calibration/')
    print('-------------------------------------------------------')
    camera.calibrate()
    print('Encontrando extrinsicos a partir de imagens em ../data/tras/')
    print('-------------------------------------------------------')
    camera.FindExtrinsics()
    camera.measure()
    cv2.destroyAllWindows()