CC ?= cc
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Wpedantic
# -pthread: portable pthreads (no-op-ish on musl libc, which carries its own
# threads). No OpenMP, no glibc-only, plain POSIX + libm. No SIMD yet.
CFLAGS += -pthread
LDLIBS = -lm -pthread

# Sanitizer builds go through a serial clean+rebuild (never parallel with
# clean) via the asan/ubsan/tsan targets below.
ifeq ($(SANITIZER),asan)
CFLAGS += -fsanitize=address,undefined -g -O1 -fno-omit-frame-pointer
endif
ifeq ($(SANITIZER),ubsan)
CFLAGS += -fsanitize=undefined -g -O1 -fno-omit-frame-pointer
endif
ifeq ($(SANITIZER),tsan)
CFLAGS += -fsanitize=thread -g -O1 -fno-omit-frame-pointer
endif

SRC = sjev.c model.c token.c math.c data.c train.c
OBJ = $(SRC:.c=.o)
BIN = sjev

PREFIX ?= /usr/local
BINDIR ?= $(PREFIX)/bin

all: $(BIN)

$(BIN): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Sanitiser builds for parity / dev / race detection. Serial by
# construction: clean finishes before the rebuild starts.
asan:
	$(MAKE) clean
	$(MAKE) $(BIN) SANITIZER=asan

ubsan:
	$(MAKE) clean
	$(MAKE) $(BIN) SANITIZER=ubsan

tsan:
	$(MAKE) clean
	$(MAKE) $(BIN) SANITIZER=tsan

clean:
	rm -f $(OBJ) $(BIN)

install: $(BIN)
	install -Dm755 $(BIN) $(DESTDIR)$(BINDIR)/$(BIN)

check: $(BIN)
	sh ../tests/run_tests.sh

.PHONY: all clean install asan ubsan tsan check
