#
# Makefile for non-Microsoft compilers
#

all: MakeAll


MakeAll:
	$(MAKE) -C snap-core
	$(MAKE) -C drivers

clean:
	$(MAKE) clean -C snap-core
	$(MAKE) clean -C drivers
	rm -rf Debug Release ipch
