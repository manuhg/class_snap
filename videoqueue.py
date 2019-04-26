import threading

class Videoqueue:
    def __init__(self,cap,batch_size=8,queue_size=4,interval=None):
        self.batch_size = batch_size #num of frames every item in the queue will contain at max
        self.cap = cap
        self.queue = Queue() #it is a queue of sets of frames, each set has batch_size number of frames
        self.lock =  threading.Lock()
        self.finished = False
        self.started = False
        self.queue_size = queue_size
        if interval:
            self.target_interval = 0
            self.interval = interval
    
    def start(self):
        if self.started:
            print('Video capture thread is already started')
            return
        
        self.started = True
        self.thread = threading.Thread(target=self.run, args=())
        self.thread.start()
        return self
        return
    
    def get(self):
        return self.queue.get()

    def run():
        while not self.finished:
            self.fill_queue()
    
    def finish(self):
        self.finished = True
    
    def stop(self):
        self.thread.join()

    def fill_queue(self,interval):
        batch = []
        
        with self.lock:
            qs = self.queue.size()
        
        if qs<self.queue_size and not self.finished:
            for i in range(self.queue_size-qs):
                for j in range(batch_size):

                    if self.interval:
                        cap.set(cv2.CAP_PROP_POS_MSEC, target_interval)
                        target_interval += interval
                    ret,frame = self.cap.read()
                    if not ret:
                        self.finish()
                    else:
                        batch.append(frame)
                self.queue.put(batch)

