# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.31

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/Users/ibleminen/Library/CloudStorage/OneDrive-AalborgUniversitet/VISUAL STUDIO/img_server/c++"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/Users/ibleminen/Library/CloudStorage/OneDrive-AalborgUniversitet/VISUAL STUDIO/img_server/c++"

# Include any dependencies generated for this target.
include CMakeFiles/img_server.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/img_server.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/img_server.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/img_server.dir/flags.make

CMakeFiles/img_server.dir/codegen:
.PHONY : CMakeFiles/img_server.dir/codegen

CMakeFiles/img_server.dir/main.cpp.o: CMakeFiles/img_server.dir/flags.make
CMakeFiles/img_server.dir/main.cpp.o: main.cpp
CMakeFiles/img_server.dir/main.cpp.o: CMakeFiles/img_server.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir="/Users/ibleminen/Library/CloudStorage/OneDrive-AalborgUniversitet/VISUAL STUDIO/img_server/c++/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/img_server.dir/main.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/img_server.dir/main.cpp.o -MF CMakeFiles/img_server.dir/main.cpp.o.d -o CMakeFiles/img_server.dir/main.cpp.o -c "/Users/ibleminen/Library/CloudStorage/OneDrive-AalborgUniversitet/VISUAL STUDIO/img_server/c++/main.cpp"

CMakeFiles/img_server.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/img_server.dir/main.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/ibleminen/Library/CloudStorage/OneDrive-AalborgUniversitet/VISUAL STUDIO/img_server/c++/main.cpp" > CMakeFiles/img_server.dir/main.cpp.i

CMakeFiles/img_server.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/img_server.dir/main.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/ibleminen/Library/CloudStorage/OneDrive-AalborgUniversitet/VISUAL STUDIO/img_server/c++/main.cpp" -o CMakeFiles/img_server.dir/main.cpp.s

# Object files for target img_server
img_server_OBJECTS = \
"CMakeFiles/img_server.dir/main.cpp.o"

# External object files for target img_server
img_server_EXTERNAL_OBJECTS =

bin/img_server: CMakeFiles/img_server.dir/main.cpp.o
bin/img_server: CMakeFiles/img_server.dir/build.make
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_gapi.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_stitching.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_alphamat.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_aruco.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_bgsegm.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_bioinspired.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_ccalib.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_dnn_objdetect.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_dnn_superres.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_dpm.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_face.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_freetype.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_fuzzy.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_hdf.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_hfs.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_img_hash.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_intensity_transform.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_line_descriptor.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_mcc.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_quality.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_rapid.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_reg.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_rgbd.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_saliency.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_signal.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_stereo.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_structured_light.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_superres.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_surface_matching.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_tracking.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_videostab.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_wechat_qrcode.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_xfeatures2d.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_xobjdetect.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_xphoto.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_shape.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_highgui.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_datasets.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_plot.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_text.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_ml.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_phase_unwrapping.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_optflow.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_ximgproc.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_video.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_videoio.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_imgcodecs.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_objdetect.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_calib3d.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_dnn.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_features2d.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_flann.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_photo.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_imgproc.4.10.0.dylib
bin/img_server: /Users/ibleminen/miniconda3/lib/libopencv_core.4.10.0.dylib
bin/img_server: CMakeFiles/img_server.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir="/Users/ibleminen/Library/CloudStorage/OneDrive-AalborgUniversitet/VISUAL STUDIO/img_server/c++/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable bin/img_server"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/img_server.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/img_server.dir/build: bin/img_server
.PHONY : CMakeFiles/img_server.dir/build

CMakeFiles/img_server.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/img_server.dir/cmake_clean.cmake
.PHONY : CMakeFiles/img_server.dir/clean

CMakeFiles/img_server.dir/depend:
	cd "/Users/ibleminen/Library/CloudStorage/OneDrive-AalborgUniversitet/VISUAL STUDIO/img_server/c++" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/Users/ibleminen/Library/CloudStorage/OneDrive-AalborgUniversitet/VISUAL STUDIO/img_server/c++" "/Users/ibleminen/Library/CloudStorage/OneDrive-AalborgUniversitet/VISUAL STUDIO/img_server/c++" "/Users/ibleminen/Library/CloudStorage/OneDrive-AalborgUniversitet/VISUAL STUDIO/img_server/c++" "/Users/ibleminen/Library/CloudStorage/OneDrive-AalborgUniversitet/VISUAL STUDIO/img_server/c++" "/Users/ibleminen/Library/CloudStorage/OneDrive-AalborgUniversitet/VISUAL STUDIO/img_server/c++/CMakeFiles/img_server.dir/DependInfo.cmake" "--color=$(COLOR)"
.PHONY : CMakeFiles/img_server.dir/depend

