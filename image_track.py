import face_alignment
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from skimage import io

import os
from skimage import img_as_ubyte
import cv2

def demo_test():
    # Run the 3D face alignment on a test image, without CUDA.
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, enable_cuda=False, flip_input=False)

    input = io.imread(R'D:\WorkingItems\neckcap\0206\recon\exp\E0223\input\000000.png')
    preds = fa.get_landmarks(input)[-1]

    #TODO: Make this nice
    fig = plt.figure(figsize=plt.figaspect(.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(input)
    ax.plot(preds[0:17,0],preds[0:17,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[17:22,0],preds[17:22,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[22:27,0],preds[22:27,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[27:31,0],preds[27:31,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[31:36,0],preds[31:36,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[36:42,0],preds[36:42,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[42:48,0],preds[42:48,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[48:60,0],preds[48:60,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[60:68,0],preds[60:68,1],marker='o',markersize=6,linestyle='-',color='w',lw=2) 
    ax.axis('off')

    # ax = fig.add_subplot(1, 2, 2, projection='3d')
    # surf = ax.scatter(preds[:,0]*1.2,preds[:,1],preds[:,2],c="cyan", alpha=1.0, edgecolor='b')
    # ax.plot3D(preds[:17,0]*1.2,preds[:17,1], preds[:17,2], color='blue' )
    # ax.plot3D(preds[17:22,0]*1.2,preds[17:22,1],preds[17:22,2], color='blue')
    # ax.plot3D(preds[22:27,0]*1.2,preds[22:27,1],preds[22:27,2], color='blue')
    # ax.plot3D(preds[27:31,0]*1.2,preds[27:31,1],preds[27:31,2], color='blue')
    # ax.plot3D(preds[31:36,0]*1.2,preds[31:36,1],preds[31:36,2], color='blue')
    # ax.plot3D(preds[36:42,0]*1.2,preds[36:42,1],preds[36:42,2], color='blue')
    # ax.plot3D(preds[42:48,0]*1.2,preds[42:48,1],preds[42:48,2], color='blue')
    # ax.plot3D(preds[48:,0]*1.2,preds[48:,1],preds[48:,2], color='blue' )

    # ax.view_init(elev=90., azim=90.)
    # ax.set_xlim(ax.get_xlim()[::-1])
    plt.show()

def export_2d_landmarks(lmk, outname):
    with open(outname, 'w') as f:
        f.write('{}\n'.format(len(lmk)))
        for pt in lmk:
            f.write('{} {}\n'.format(pt[0], pt[1]))

def visualize_landmarks(input, lmk, outname, draw_lable=False):
    img_bak = img_as_ubyte(input)
    img = cv2.cvtColor(img_bak, cv2.COLOR_RGB2BGR)
    for pt in lmk:
        cv2.circle(img, tuple(pt), 2, (0,255,255))
    
    cv2.polylines(img, [np.array(lmk[0:17], np.int32).reshape((-1,1,2))], False, (0,128,255))
    cv2.polylines(img, [np.array(lmk[17:22], np.int32).reshape((-1,1,2))], False, (0,128,255))
    cv2.polylines(img, [np.array(lmk[22:27], np.int32).reshape((-1,1,2))], False, (0,128,255))
    cv2.polylines(img, [np.array(lmk[27:31], np.int32).reshape((-1,1,2))], False, (0,128,255))
    cv2.polylines(img, [np.array(lmk[31:36], np.int32).reshape((-1,1,2))], False, (0,128,255))
    cv2.polylines(img, [np.array(lmk[36:42], np.int32).reshape((-1,1,2))], True, (0,128,255))
    cv2.polylines(img, [np.array(lmk[42:48], np.int32).reshape((-1,1,2))], True, (0,128,255))
    cv2.polylines(img, [np.array(lmk[48:60], np.int32).reshape((-1,1,2))], True, (0,128,255))
    cv2.polylines(img, [np.array(lmk[60:68], np.int32).reshape((-1,1,2))], True, (0,128,255))

    if (draw_lable):
        for i in range(len(lmk)):
            text_ptx = int(round(lmk[i,0]+5))
            text_pty = int(round(lmk[i,1]+5))
            cv2.putText(img, '{:02d}'.format(i), (text_ptx, text_pty), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255))
    
    cv2.imwrite(outname, img)

def track_images(infolder, start_id, end_id, outfolder):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, enable_cuda=False, flip_input=False)
    if (not os.path.isdir(outfolder)):
        os.makedirs(outfolder)
    for i in range(start_id, end_id):
        file_basename = '/{:06d}'.format(i)
        input = io.imread(infolder+file_basename+'.png')
        preds = fa.get_landmarks(input)
        if (preds is None):
            print('WARNING: No face detected in frame {}!'.format(i))
            io.imsave(outfolder+file_basename+'.png', input)
        else:
            lmk = preds[-1]
            export_2d_landmarks(lmk, outfolder+file_basename+'.txt')
            visualize_landmarks(input, lmk, outfolder+file_basename+'.png')
        if (i % 100 == 0):
            print('finished frame {}'.format(i))
    print("All done!")

if __name__ == '__main__':
    #demo_test()
    track_images(R'D:\WorkingItems\neckcap\0206\recon\exp\E0223\input', 0, 765, R'D:\WorkingItems\neckcap\0206\recon\exp\E0223\landmark')