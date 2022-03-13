GLSLC=glslangValidator
GLSLCFLAGS=-V

SHD=src/egui.vert.spv \
	src/egui.frag.spv

all: shaders
.PHONY: all clean shaders

clean:
	$(RM) $(SHD)

shaders: $(SHD)

%.spv: % Makefile
	$(GLSLC) $(GLSLCFLAGS) -o $@ $<

