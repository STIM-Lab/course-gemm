#Specify the version being used as well as the language
cmake_minimum_required(VERSION 3.30)

#Name your project here
project(matrix LANGUAGES CXX CUDA)

#set the module directory
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")
set(CMAKE_CXX_STANDARD 20)

find_package(TIRA REQUIRED)
find_package(CUDAToolkit REQUIRED)

# I usually set a few parameters when building in Visual Studio since the defaults
#	can make things difficult. What this mostly does is set the Debug and Release
#	directories to the build directory. That way the executable is always in the same
#	location when testing the application from the command line.
#
# Another thing this does is enable the ability to use functions like printf() which
#	are deprecated and may cause warnings.
if ( MSVC )
	SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${OUTPUT_DIRECTORY}")
	SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIRECTORY}")
	SET( LIBRARY_OUTPUT_DIRECTORY_DEBUG "${OUTPUT_DIRECTORY}")
	SET( LIBRARY_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIRECTORY}")
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
	add_definitions(-D_SCL_SECURE_NO_WARNINGS)
	add_definitions(-D_USE_MATH_DEFINES)
else()
# These settings are used for GCC and other compilers (usually on Linux systems). It sets
#	agressive optimization for Release builds

	set(CMAKE_CXX_FLAGS "-Wall -Wextra")
	set(CMAKE_CXX_FLAGS_RELEASE "-O3")
	set(CMAKE_CXX_FLAGS_DEBUG "-g")
endif ( MSVC )


# Set all of the include directories so that you can include the necessary header files
include_directories(
			${CMAKE_CURRENT_BINARY_DIR}
			${CMAKE_CURRENT_SOURCE_DIR}
			${TIRA_INCLUDE_DIRS}
)

# Set the files required to build the executable
add_executable(matrix
		main.cpp
		matrix.h
		cpufunc.h
		gpufunc.cu		
)


# Set all of the libraries so that the linker knows where to find them
target_link_libraries(matrix
		CUDA::cublas
)