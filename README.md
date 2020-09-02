# Cache-Benchmark
Tests the cache on Jetson boards.
## Usage
Simply compile by running the compile script (currently configured for Volta/Xavier) on a Jetson Board:
```
./compile
```
and then run with
```
./as
```
## Configuration
To modify the configuration change the variables contained at the start of "AS.cu":
```
// Set PRINT to 1 for debug output
#define PRINT 0
#define FROM_debug 0
#define TO_debug 16

// Set ZEROCOPY to 1 to use Zero Copy Memory Mode, UNIFIED to 1 to use Unified Memory, COPY to 1 to use Copy
#define ZEROCOPY 0
#define UNIFIED 0
#define COPY 0

// Set CPU to 1 to use the CPU concurrently
#define CPU 1
// Set OPENMP to 1 to use more than 1 thread for the CPU
#define OPENMP 1

unsigned int N = 2;
// Array size
const int POW = 12;
const int RUNS = 50;
// Side inversions in the same run
const int SUMS = 8;
```
