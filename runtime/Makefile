RTDIR := $(HUMROOT)/runtime
INCLUDES := -I$(RTDIR) -I/usr/include -I$(HUMROOT)/driver/hum
INCLUDES += -I/usr/local/cuda/include

CXX_FLAGS := -O3 -fPIC $(INCLUDES) -DCOMPILE_RUNTIME -DUSE_MEM_PREFETCH
#CXX_FLAGS += -DDISABLE_MEMCPY_SCHEDULING

LINK := -shared -fPIC
SOURCES := $(wildcard *.cpp Device/*.cpp Wrapper/*.cpp)
LIBRARY := -pthread -lrt -ldl -L/usr/local/cuda/lib64 -L/usr/lib64 -lcuda

TARGET := $(HUMROOT)/lib/libhum.so
PERL := perl

OBJ_DIR=objs
SUBDIRS=Device Wrapper

OBJS=$(patsubst %.cpp,$(OBJ_DIR)/%.o,$(SOURCES))
DEPS=$(patsubst %.cpp,$(OBJ_DIR)/.deps/%.d,$(SOURCES))
OBJ_SUBDIRS=$(patsubst %,$(OBJ_DIR)/%,$(SUBDIRS))
DEPS_DIR=$(patsubst %,$(OBJ_DIR)/.deps/%,$(SUBDIRS))

$(TARGET): $(OBJS)
	mkdir -p $(HUMROOT)/lib
	$(CXX) $(CXX_FLAGS) $(OBJS) $(LINK) $(LIBRARY) -o $@

$(OBJS): $(OBJ_DIR)/%.o:%.cpp | $(OBJ_DIR)
	$(CXX) $(CXX_FLAGS) $< -c -o $@

$(DEPS): $(OBJ_DIR)/.deps/%.d:%.cpp | $(DEPS_DIR)
	/bin/bash -ec '$(CC) -MM $(CXX_FLAGS) $< | $(PERL) -e '\''while(<>){s/^(.*)\.o:/$(subst /,\/,$(OBJ_DIR))\/$$1.o $(subst /,\/,$(OBJ_DIR))\/.deps\/$$1.d:/g;print;}'\''> $@'

$(DEPS_DIR): | $(OBJ_SUBDIRS)
	mkdir -p $(DEPS_DIR)

$(OBJ_SUBDIRS): | $(OBJ_DIR)
	mkdir -p $(OBJ_SUBDIRS)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)
	mkdir -p $(OBJ_SUBDIRS)

clean:
	rm -rf $(OBJS) $(OBJ_DIR)
	rm -rf $(DEPS)

-include $(DEPS)
