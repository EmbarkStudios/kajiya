GLSLC=glslangValidator
GLSLCFLAGS=-V

SHD=src/imgui.vert.spv \
	src/imgui.frag.spv

all: shaders
.PHONY: all clean shaders

clean:
	$(RM) $(SHD)

shaders: $(SHD)

%.spv: % Makefile
	$(GLSLC) $(GLSLCFLAGS) -o $@ $<

