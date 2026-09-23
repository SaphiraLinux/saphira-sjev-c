CC ?= cc
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Wpedantic
# musl-compatible, no glibc-only, plain POSIX + libm. No SIMD yet.
LDLIBS = -lm

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

# sanitiser build for parity / dev
asan: CFLAGS += -fsanitize=address,undefined -g -O1 -fno-omit-frame-pointer
asan: clean $(BIN)

clean:
	rm -f $(OBJ) $(BIN)

install: $(BIN)
	install -Dm755 $(BIN) $(DESTDIR)$(BINDIR)/$(BIN)

# simple test: requires Python to export a model first
check: $(BIN)
	@echo "run src/tests/test_parity.py after exporting a model"

.PHONY: all clean install asan check
