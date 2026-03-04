# Makefile for building hot_path.so for YOLOv8 C Hot Path demo
#
# Uses the same hot_path.c from control-loop/hot_path/ — it's generic
# (works for any model size, the config struct carries all specifics).
#
# Build:
#   cd examples/yolo-camera && nix develop
#   make
#
# Clean:
#   make clean

HOT_PATH_SRC = ../control-loop/hot_path

CC ?= clang
CFLAGS = -O3 -march=armv8.2-a+fp16 -mtune=cortex-a78ae \
         -Wall -Wextra -Wno-unused-parameter -fPIC
LDFLAGS_SHARED = -shared

.PHONY: all clean

all: hot_path.so

hot_path.so: $(HOT_PATH_SRC)/hot_path.c $(HOT_PATH_SRC)/hot_path.h
	$(CC) $(CFLAGS) $(LDFLAGS_SHARED) -I$(HOT_PATH_SRC) -o $@ $(HOT_PATH_SRC)/hot_path.c

clean:
	rm -f hot_path.so
