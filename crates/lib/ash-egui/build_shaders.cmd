set GLSLC=glslangValidator -V

%GLSLC% src/egui.vert -o src/egui.vert.spv
%GLSLC% src/egui.frag -o src/egui.frag.spv

