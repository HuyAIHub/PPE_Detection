import threading,time
from cVariables import GlobVar,GlobConstants
# from Run_model import RunModel
from track import RunModel

class RunModelThreading(threading.Thread):
    def __init__(self):
        super().__init__()
        self.name = "RunModelThread"
        self.task_lst_run =[]

    def run(self):
        '''Run Thread '''

        while True:
            try:
                lst_name = [x.getName() for x in self.task_lst_run]
                for i in range(GlobVar.dict_cam.__len__()):

                    if (GlobVar.dict_cam[i].command == GlobConstants.CMD_ADD) and "thread-RunModel--"+str(GlobVar.dict_cam[i].cameraID) not in lst_name:
                        camerathread = RunModel(GlobVar.dict_cam[i])
                        self.task_lst_run.append(camerathread)
                        
                    elif (GlobVar.dict_cam[i].command == GlobConstants.CMD_DELETE) and "thread-RunModel--"+str(GlobVar.dict_cam[i].cameraID) in lst_name:
                        for thread in threading.enumerate():
                            if (str(thread.getName()) == "thread-RunModel--" +str(GlobVar.dict_cam[i].cameraID)):
                                thread.doStop = True
                                self.task_lst_run.remove([x for x in self.task_lst_run if x.cameraID == GlobVar.dict_cam[i].cameraID][0])

                    elif (GlobVar.dict_cam[i].command == GlobConstants.CMD_UPDATE) and "thread-RunModel--"+str(GlobVar.dict_cam[i].cameraID) in lst_name:
                        for thread in threading.enumerate():
                            if (str(thread.getName()) == "thread-RunModel--" +str(GlobVar.dict_cam[i].cameraID)):
                                thread.doStop = True
                                self.task_lst_run.remove([x for x in self.task_lst_run if x.cameraID == GlobVar.dict_cam[i].cameraID][0])
                                time.sleep(2)

                                GlobVar.dict_cam[i].command = "Add"
                                camerathread = RunModel(GlobVar.dict_cam[i])
                                self.task_lst_run.append(camerathread)

            except BaseException as e:
                print(str(e))
                continue
            time.sleep(1)