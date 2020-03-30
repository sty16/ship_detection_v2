target = test
objs =  demo.o  cuHostMat.o
CC = g++
NVCC = nvcc
CPPFLAGS = -I. -I/usr/local/include/opencv4/opencv -I/usr/local/include/opencv4 \
-I/usr/local/MATLAB/R2019b/extern/include/ -std=c++11
CUINCLUDES = -I. -I/usr/local/cuda/samples/common/inc \
-I/usr/local/include/opencv4/opencv -I/usr/local/include/opencv4 \
-I/usr/local/MATLAB/R2019b/extern/include/
CUGENCODE_FLAGS =  -arch=sm_50 -rdc=true  -std=c++11
LDFLAGS = -L/usr/lib/x86_64-linux-gnu/  -L/usr/local/lib \
-L/usr/local/cuda/lib64/  -lcudart   -lcudadevrt   -lcublas                                          \
-lopencv_dnn -lopencv_gapi -lopencv_highgui -lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_stitching \
-lopencv_video -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_videoio -lopencv_imgcodecs \
-lopencv_imgproc -lopencv_core -ltiff -L/usr/local/MATLAB/R2019b/bin/glnxa64/ \
 -lmat -lmx -leng -lmex  -Wl,-rpath /usr/local/MATLAB/R2019b/bin/glnxa64  


$(target):$(objs)
	nvcc -arch=sm_50 -dlink  cuHostMat.o demo.o -o demo_link.o
	$(CC) demo_link.o  $(objs) $(LDFLAGS) -o $(target) 
	rm -f $(objs) demo_link.o

%.o:%.cu
	$(NVCC) -c $(CUINCLUDES) $(CUGENCODE_FLAGS)   $^
%.o:%.cpp
	$(CC) -c $(CPPFLAGS) $^


.PHONY:clean  
clean:
	rm -f $(target) $(objs)