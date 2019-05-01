from __future__ import print_function
import threading
from Queue import Queue
import cv2
class Videoqueue:
    def __init__(self,cap,batch_size=16,queue_size=10,interval=None):
        self.batch_size = batch_size #num of frames every item in the queue will contain at max
        self.cap = cap
        self.queue = Queue() #it is a queue of sets of frames, each set has batch_size number of frames
        #self.lock =  threading.Lock()
        self.finished = False
        self.started = False
        self.queue_size = queue_size
        if interval:
            self.target_interval = interval
            self.interval = interval
        
    def start(self):
        if self.started:
            print('Video capture thread is already started')
            return
        
        self.started = True
        self.thread = threading.Thread(target=self.run, args=())
        self.thread.start()
        return self
    
    def stop(self):
        try:
            self.thread.join()
        except Exception as e:
            print(e)
    
    def __del__(self):
        self.stop()

    def get(self):
        frames = []
        #with self.lock:
        if not self.finished:
            frames = self.queue.get(block=True,timeout=1)
        else:
            frames = self.queue.get(block=False)
        self.queue.task_done()
        frames = [] if frames is None else frames
        return frames
    
    def run(self):
        while not self.finished:
            self.fill_queue()
    
    # def finish(self):
    #     self.finished = True

    def fill_queue(self):
        batch = []
        
        #with self.lock:
        qs = self.queue.qsize()
        if  self.finished:
            return
        
        if qs<self.queue_size:
            for _ in range(self.queue_size-qs):
                for _ in range(self.batch_size):
                    if self.finished:
                        break
                    
                    if self.interval:
                        self.cap.set(cv2.CAP_PROP_POS_MSEC, self.target_interval)
                        self.target_interval += self.interval
                    ret,frame = self.cap.read()
                    if not ret:
                        #with self.lock:
                        self.finished = True
                    else:
                        batch.append(frame)
                self.queue.put(batch)
                batch = []
                if self.finished:
                    return

