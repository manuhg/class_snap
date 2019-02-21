#!/usr/bin/python2
from __future__ import print_function
from extractor import extractor
import os
import sys
import argparse

class annotation_fuser:
    def __init__(self):
        self.colour_palatte = 
        [(39, 129, 113), (164, 80, 133), (83, 122, 114), (99, 81, 172), (95, 56, 104), (37, 84, 86), (14, 89, 122),
        (80, 7, 65), (10, 102, 25),(90, 185, 109), (106, 110, 132), (169, 158, 85), (188, 185, 26), (103, 1, 17), (82, 144, 81), 
        (92, 7, 184), (49, 81, 155), (179, 177, 69), (93, 187, 158), (13, 39, 73), (12, 50, 60), (16, 179, 33), (112, 69, 165), 
        (15, 139, 63), (33, 191, 159), (182, 173, 32), (34, 113, 133), (90, 135, 34), (53, 34, 86), (141, 35, 190), (6, 171, 8), 
        (118, 76, 112), (89, 60, 55), (15, 54, 88), (112, 75, 181), (42, 147, 38), (138, 52, 63), (128, 65, 149), (106, 103, 24), 
        (168, 33, 45), (28, 136, 135), (86, 91, 108), (52, 11, 76), (142, 6, 189), (57, 81, 168), (55, 19, 148), (182, 101, 89), 
        (44, 65, 179), (1, 33, 26), (122, 164, 26), (70, 63, 134), (137, 106, 82), (120, 118, 52), (129, 74, 42), (182, 147, 112),
        (22, 157, 50), (56, 50, 20), (2, 22, 177), (156, 100, 106), (21, 35, 42), (13, 8, 121), (142, 92, 28), (45, 118, 33), 
        (105, 118, 30), (7, 185, 124), (46, 34, 146), (105, 184, 169), (22, 18, 5), (147, 71, 73), (181, 64, 91), (31, 39, 184),
        (164, 179, 33), (96, 50, 18), (95, 15, 106), (113, 68, 54), (136, 116, 112), (119, 139, 130), (31, 139, 34), (66, 6, 127),
        (62, 39, 2), (49, 99, 180), (49, 119, 155), (153, 50, 183), (125, 38, 3), (129, 87, 143), (49, 87, 40), (128, 62, 120),
        (73, 85, 148), (28, 144, 118), (29, 9, 24), (175, 45, 108), (81, 175, 64), (178, 19, 157), (74, 188, 190), (18, 114, 2),
        (62, 128, 96), (21, 3, 150), (0, 6, 95), (2, 20, 184), (122, 37, 185)]
    
    def box_and_label(self, ann_data, cvimg, opfname=None, colours = None): #once per bbox
        label, coordinates = ann_data
        pt1, pt2 = coordinates
        colours = colours if colours else self.colours

        label = result[0] + ' - '+str(np.around(float(result[1]), decimals=2))
        color = random.choice(colors)
        cv2.rectangle(cvimg, pt1, pt2, color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        pt2 = pt1[0] + t_size[0] + 3, pt1[1] + t_size[1] + 4
        cv2.rectangle(cvimg, pt1, pt2, color, -1)
        cv2.putText(cvimg, label, (pt1[0], pt1[1] + t_size[1] + 4),cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
        if opfname:
            cv2.imwrite(opfname,cvimg)
        else:
            return cvimg
    
    #annotation_dicts_lst
    def annoataions2coordinates(self,annotation_dict): #once per file
        #annotations format = {'annotation': {'data_filename': fname, 'data_type': 'image', 'data_annotation': {'bounding_polygon': ['bbox_dicts_lst'], 'bounding_box': ''}}}
        annt = annotation_dict
        dest_dir = dest_dir+'/' if dest_dir[-1]!='/' else dest_dir
        bbox_dicts_lst = annotations['annotation']['data_annotation']['bounding_box']
        'classification_label':tmp[lbk][i],'point_2D'
        ann_data = []
        for d in bbox_dicts_lst:
            label = d['classification_label']
            coordinates = d['point_2D']
            coordinates = tuple(map(lambda x:  tuple(map(lambda v:int(v), x.split(','))),coordinates))
            ann_data.append([label,coordinates])
        self.ann_data = ann_data


def build_args_parser():
    parser = argparse.ArgumentParser(description='Tool to fuse image and its annotations and yield image(s) with bouding boxes')
    parser.add_argument('-i','--input_file', dest='input_file',help='input video file(s)', default='detections.zip', type=str,required=True)
    parser.add_argument('-o','--output_dir', dest='output_dir',help='Destination Directory for output', default='fused_output', type=str)
    return parser



def main():
    parser = build_args_parser()
    if len(sys.argv) >= 1:
        args = parser.parse_args()
        input_file = args.input_video
        output_dir = args.output_dir
        
        if (not input_file) or (not output_dir):
            print('\nError: Arguments empty/invalid')
            print('input_file:',input_file)
            print('output_dir:',output_dir)
            print('Please check the above')
            exit()
        
    else:
        print('\nExample: python2 fuse_annotations.py -i detections.zip -o fused_images \n')
        parser.print_help()

if __name__ == "__main__":
    main()