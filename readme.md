arm-linux-g++ -Wno-psabi -I/home/wayen/arm/opencv_arm/opencv_install/include -L/home/wayen/arm/opencv_arm/opencv_install/lib -lpthread -lopencv_imgproc -lopencv_highgui -lopencv_core -lpthread -lrt -o test1 test1.cpp

* 记得添加库的顺序

* 编译文件
 `g++ -ggdb main.cpp $(pkg-config --cflags --libs opencv2.4) -L/home/wayen/Qt5.11.1/5.11.1/gcc_64/lib -lQt5Widgets -lQt5Gui -lQt5Core -otest1`