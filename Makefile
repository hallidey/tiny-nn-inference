# Simple Makefile for Tiny Neural Inference Engine (TNIE)

CC       ?= cc
CPPFLAGS ?= -Iinclude -Iexamples/xor
CFLAGS   ?= -Wall -Wextra -Wpedantic -std=c11 -O2
LDLIBS   ?= -lm

SRC_DIR = src
CORE_SRCS = $(SRC_DIR)/tnie_nn.c $(SRC_DIR)/tnie_activations.c
DEMO_SRCS = examples/xor/main.c examples/xor/model_xor.c
SRCS      = $(CORE_SRCS) $(DEMO_SRCS)

OBJS    = $(SRCS:.c=.o)
TARGET  = tnie_xor_demo
TEST_TARGET = tnie_tests

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDLIBS)

$(TEST_TARGET): tests/test_tnie.c $(CORE_SRCS) examples/xor/model_xor.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $^ $(LDLIBS)

test: $(TEST_TARGET)
	./$(TEST_TARGET)

clean:
	$(RM) $(OBJS) $(TARGET) $(TEST_TARGET)
