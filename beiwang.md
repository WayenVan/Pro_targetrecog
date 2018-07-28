arm-linux-g++ -Wno-psabi -I/home/wayen/arm/opencv_arm/opencv_install/include -L/home/wayen/arm/opencv_arm/opencv_install/lib -lpthread -lopencv_imgproc -lopencv_highgui -lopencv_core -lpthread -lrt -o test1 test1.cpp

* 记得添加库的顺序