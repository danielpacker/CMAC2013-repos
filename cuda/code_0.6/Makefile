

#-----------------------------------------------------#
#     This is the Makefile for cuda2DMinModel.        #
#                                                     #
#     Authors: E. Bartocci (eziobart@ams.sunysb.edu)  #
#              F. Fenton   (fhf3@cornell.edu)         #
#                                                     #
#     Description:                                    #
#                                                     #
#      The following code is part of "cuda2DMinModel" #
#      a CUDA implementation of the 2D simulator of   #
#      the Bueno-Orovio-Cherry-Fenton Minimal Model.  #
#      This code is provided as a supplement of the   #
#      paper "Teaching cardiac electrophysiology      #
#      modeling to undergraduate students:            #
#      "Lab exercises and GPU programming for the     # 
#       study of arrhythmias and spiral wave dynamics"#
#       submitted to Advances in Physiology Education #
#                                                     # 
#                                                     #
#     Date:  01/03/11                                 #
#      Copyright 2011                                 #
#      Ver. 0.5                                       #
#                                                     #
#                                                     #
# ==   Free distribution with authors permission   == #
#                                                     #
# ==   SUNY Stony Brook, Stony Brook, NY              #
# ==   Cornell University, Ithaca, NY                 # 
# ----------------------------------------------------# 

#-----------------------------------------------------#
#             Check the Operative System              #
#-----------------------------------------------------#

OS = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
# 'linux' is output for Linux system, 'darwin' for OS X
DARWIN = $(strip $(findstring DARWIN, $(OS)))
ifneq ($(DARWIN),)
   SNOWLEOPARD = $(strip $(findstring 10.6, $(shell egrep "<string>10\.6" /System/Library/CoreServices/SystemVersion.plist)))
endif



#echo $(OS)

# Compiler flags appropriate for the NVCC compiler
# -------------------------------------------------
CC             = nvcc 
ifneq ($(DARWIN),)
	SDK_CUDA_HOME  = "/Developer/GPU\ Computing"
else
	SDK_CUDA_HOME  = ${HOME}/NVIDIA_GPU_Computing_SDK
endif
LIBS          := -lglut -lXi -lXmu -lGL -lGLU -lX11 -lcudart





CU_SRCS += \
main.cu \
graphics.cu 

CU_DEPS += \
cu_main.d \
cu_graphics.d 

OBJS += \
cu_main.o \
cu_graphics.o 


cu_%.o: %.cu
	@echo 'OPERATIVE SYSTEM FOUND'
	@echo $(OS)
	@echo 'Building file: $<'
	@echo 'Invoking: CUDA NVCC Compiler'
	$(CC) -c -I/usr/X11/include -I/usr/local/cuda/include -I"$(SDK_CUDA_HOME)/C/common/inc" -o "$@" "$<" && \
	echo -n '$(@:%.o=%.d)' $(dir $@) > '$(@:%.o=%.d)' && \
	$(CC) -M -I/usr/X11/include -I/usr/local/cuda/include -I"$(SDK_CUDA_HOME)/C/common/inc"   "$<" >> '$(@:%.o=%.d)'
	@echo 'Finished building: $<'
	@echo ' '
		

# All Target
all: cuda2DMinModel

# Tool invocations
cuda2DMinModel: $(OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: C++ Linker'
	nvcc -L/usr/local/cuda/lib -L/usr/X11R6/lib/ -L/usr/X11R6/include/X11 -L"$(SDK_CUDA_HOME)/shared/lib" -L"$(SDK_CUDA_HOME)/C/lib/" -o "cuda2DMinModel" $(OBJS) $(USER_OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '



clean:	tidy
	rm -f cuda2DMinModel *.o *~
	rm -f *.d

tidy:
	rm -f cuda2DMinModel *.o 

tar: 
	tar -cvf cuda2DMinModel.tar README main.cu graphics.cu  \
	graphics.h cusim.h cell_model.h sim_parameters.dat Makefile
