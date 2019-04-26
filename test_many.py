import os

def exec_cmd(str):
    print(os.popen(str).read())

videos = 
['https://www.youtube.com/watch?v=C42WckJB_pc','https://www.youtube.com/watch?v=f1cG_e8kPGY',
'https://www.youtube.com/watch?v=G2VaJvNNp4k','https://www.youtube.com/watch?v=h95efX6Wx4s',
'https://www.youtube.com/watch?v=HtagT3BRDO8','https://www.youtube.com/watch?v=iBt2aTjCNmI',
'https://www.youtube.com/watch?v=j0Z_fODM5gE','https://www.youtube.com/watch?v=nsPQvZm_rgM',
'https://www.youtube.com/watch?v=QZsthdsh6yk','https://www.youtube.com/watch?v=R8zVnJy76QI',
'https://www.youtube.com/watch?v=WvqUCSw01Cw','https://www.youtube.com/watch?v=X32dce7_D48',
]

def test_videos():
    for video in videos:
        exec_cmd('python main.py -c cl.txt -i '+video+ '-m yolo -t 1')
        exec_cmd('python main.py -c cl.txt -i '+video+ '-m detectron -t 1')

test_videos()