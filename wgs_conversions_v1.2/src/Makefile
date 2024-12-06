#指定交叉编译工具链
CROSS_COMPILE = arm-hisiv100nptl-linux-
AS      = $(CROSS_COMPILE)as  
LD      = $(CROSS_COMPILE)ld  
CC      = $(CROSS_COMPILE)gcc  
CPP     = $(CC) -E  
AR      = $(CROSS_COMPILE)ar  
NM      = $(CROSS_COMPILE)nm
 
#获取工程的根目录的全路径
SOURCE_ROOT = $(shell pwd)
 
CFLAGS		:= -Wall -O2 -fno-builtin
CPPFLAGS	:= 
 
#-I是指定头文件的意思，所以这里是将所有的头文件的变量都包含在 INCLUDE_DIR 变量中
INCLUDE_DIR := -I $(SOURCE_ROOT)/ \
	-I$(SOURCE_ROOT)/wgs_convertor\
	
 
#生成的目标的库文件名称是mylib.so
APP_NAME=mylib.so
 

