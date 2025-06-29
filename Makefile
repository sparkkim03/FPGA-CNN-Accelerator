# Makefile for compiling the convolution test program

# Compiler
CC = gcc

# Compiler flags
# -g: Add debug information
# -Wall: Turn on all warnings
# -o: Specify output file name
CFLAGS = -g -Wall

# The final executable name
TARGET = convtest

# Source files
SRCS = test/test.c src/conv.c src/pooling.c src/dense.c src/lenet_weights.c

# Object files (derived from source files)
OBJS = $(SRCS:.c=.o)

# Default rule: build the target
all: $(TARGET)

# Rule to link the object files into the final executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) -lm

# Rule to compile a .c file into a .o file
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Rule to clean up build files
clean:
	rm -f $(OBJS) $(TARGET)

# Phony targets
.PHONY: all clean
