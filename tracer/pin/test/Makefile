CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -O -fopenmp
LDFLAGS := -pthread -fopenmp

TARGET := thread_names
SRCS := thread_names.cpp

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

.PHONY: clean
clean:
	$(RM) $(TARGET)
