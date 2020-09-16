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
#define ZEROCOPY 1
#define UNIFIED 0
#define COPY 0

// Set RESULTCHECK to 1 to verify the result with a single CPU thread
#define RESULTCHECK 1

// Set CPU to 1 to use the CPU concurrently, otherwise a slightly different version of the benchmark is used
#define CPU 1
// Set OPENMP to 1 to use more than 1 thread for the CPU (does nothing if CPU is set to 0)
#define OPENMP 1

// N is later overwritten as N = N^POW, making N the size of the input array
unsigned int N = 2;
const int POW = 18;

const int SUMS = 8; // As CPU and GPU work on either the left side or right side, this number indicates how many "side swaps" there will be
const int RUNS = 5; // How many times the benchmark is run
const int BLOCK_SIZE_X = 1024;
const int BLOCK_SIZE_Y = 1;
```
