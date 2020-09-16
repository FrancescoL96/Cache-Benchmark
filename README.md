# Cache-Benchmark
Tests the cache on Jetson boards: both the CPU and the GPU cycle through the input array to perform mathematical operations, the CPU accesses one half the array and the GPU access the other half. The CPU does a mostly sequential access on its side on even/odd positions, while the GPU performs a sequential access on the CPU side on odd/even positions. (Odd or even depends on if it is right side or left side of the array, as CPU and GPU do swap around)

As Zero Copy allows the usage of the cache on the Xavier board and concurrent execution, the CPU moves through the data at the same time as the GPU (avoiding race conditions as CPU and GPU never read/write on the same locations), the CPU should cache some data inside the L2/L3. After the "side swap" (the number of side swaps is indicates by the config var "SUMS") the GPU should read data which is possibly cached on the CPU, making use of the Hardware I/O coherency to avoid inconsisent data/flushing the CPU cache.
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
# Result
The CPU, when using Unified Memory, Copy or Zero Copy, takes similar amounts of time, meanwhile the GPU is always significantly slower on Zero Copy, even if the average GPU kernel time is about 5x slower compared to the 80x slower when run on a TX2. While it is still a lot faster compared to a TX2, it is still much slower when compared to Unified Memory/Copy: the benefit of concurrent access and removing the copy times are heavily outweighed by the slower GPU computation times.
