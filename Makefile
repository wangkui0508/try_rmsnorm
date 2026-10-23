CC      ?= clang
CFLAGS  ?= -O3 -mcpu=apple-m1 -fPIC -Wall -Wextra
LDFLAGS ?= -shared

BUILD := build
LIB   := $(BUILD)/librmsnorm.dylib
BENCH := $(BUILD)/bench_rmsnorm

.PHONY: all clean bench
all: $(LIB) $(BENCH)

$(BUILD):
	mkdir -p $(BUILD)

$(LIB): csrc/rmsnorm.c | $(BUILD)
	$(CC) $(CFLAGS) $(LDFLAGS) $< -o $@

$(BENCH): csrc/rmsnorm.c csrc/bench.c | $(BUILD)
	$(CC) $(CFLAGS) $^ -o $@

bench: $(BENCH)
	./$(BENCH) 1 300000 1 0
	./$(BENCH) 4096 1000 4 0

clean:
	rm -f $(LIB) $(BENCH)
