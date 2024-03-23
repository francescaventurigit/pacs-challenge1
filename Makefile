# get all files *.cpp
SRCS=$(wildcard *.cpp)
# get the corresponding object file
OBJS = $(SRCS:.cpp=.o)
# get all headers in the working directory
HEADERS=$(wildcard *.hpp)

exe_sources=$(filter main%.cpp,$(SRCS))
EXEC=$(exe_sources:.cpp=)

CXXFLAGS = -std=c++17
CPPFLAGS = -DNDEBUG -I${mkEigenInc} 
# -I/home/jammy/Examples/include

# Link all object files to create the executable
main: $(OBJS)
	$(CXX) $(LDFLAGS) $(OBJS) -o main $(LIBS)

# # Compile each source file into an object file
# %.o: %.cpp $(HEADERS)
# 	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

clean:
	$(RM) *.o 

distclean: clean
	$(RM) main