# SJEV top-level build driver. The real build lives in src/.
all:
	$(MAKE) -C src all

clean:
	$(MAKE) -C src clean

install:
	$(MAKE) -C src install

asan:
	$(MAKE) -C src asan

ubsan:
	$(MAKE) -C src ubsan

tsan:
	$(MAKE) -C src tsan

# Full functional suite (builds first, then runs tests/run_tests.sh).
check:
	$(MAKE) -C src check

.PHONY: all clean install asan ubsan tsan check
