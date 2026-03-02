# Makefile for building hot_path.so and neon_mlp.so
#
# Build on Jetson Orin AGX (aarch64):
#   make
#
# Build from nix shell (examples/tinygrad):
#   cd examples/control-loop/hot_path && make
#
# Clean:
#   make clean

CC ?= gcc
CFLAGS = -O3 -march=armv8.2-a+fp16 -mtune=cortex-a78ae \
         -Wall -Wextra -Wno-unused-parameter \
         -fPIC

# Shared library flags
LDFLAGS_SHARED = -shared

.PHONY: all clean

all: hot_path.so neon_mlp.so neon_mlp_f32.so

hot_path.so: hot_path.c hot_path.h
	$(CC) $(CFLAGS) $(LDFLAGS_SHARED) -o $@ hot_path.c

neon_mlp.so: neon_mlp.c neon_mlp.h
	$(CC) $(CFLAGS) $(LDFLAGS_SHARED) -o $@ neon_mlp.c

neon_mlp_f32.so: neon_mlp_f32.c neon_mlp_f32.h
	$(CC) $(CFLAGS) $(LDFLAGS_SHARED) -o $@ neon_mlp_f32.c

clean:
	rm -f hot_path.so neon_mlp.so neon_mlp_f32.so
