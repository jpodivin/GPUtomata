
#include "GPUtomata.cuh"
#define MAX_SIZE 1000

int main();

cudaError_t iterateWithCuda(std::vector<int>& newState, std::vector<int>& currentState, unsigned int size, std::vector<int> ruleset);

std::vector<int> ruleTranslation(std::vector<std::vector<int>> transitionRules);
int checkRuleset(std::vector<int> transitionRules);

void setGliderTest(int* currentState, int size);
void initField(std::vector<int>& currentState, std::vector<int>& newState, int size, bool gliderTest);

void automatonSetup(int& size, int& runTime, bool& randomInit, std::vector<int>& ruleSet);

void printDelimiter(std::string delimiter, int outputSize);
void printState(int* currentState, int size);

__global__ void iterateRule(int* newState, int* currentState, int size, int* ruleset) {

    /*
        Using block number, block size and thread id to derive grid coordinates. 
        The same principle used in array to grid translation in following loops.
    */
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int orow = int(index / size);
    int ocol = index % size;
    int row;
    int col;
    int sum = 0;

    /*
      This can be done with one statement since the neighborhood is defined by known offsets. 
      However the compiler will probably end up with equivalent result anyway, plus this way it's more intelligeble.    
    */
    for (int x = -1; x < 2; x++)
    {
        row = (orow + x)%size;

        if (row < 0) {
            row = size + row;
        }
        for (int y = -1; y < 2; y++)
        {
            col = (ocol + y)%size;

            if (col == -1) {
                col = size + col;
            }

            sum += currentState[col + (row * size)];
        }
    }

    /*
        Extendable to more states. If I wanted to build a library this would be replaced by a loop.
    */
    if (currentState[index] == 0) {
        newState[index] = ruleset[sum];
    }
    else {
        newState[index] = ruleset[sum+9];
    }
}

int main()
{

    int size = 0;
    std::vector<int> currentState;
    std::vector<int> newState;
    std::vector<int> transitionRules;  
    const std::string delimiter = "-";

    /*
        N*M vector
        N stands for the number of possible states of a cell
        M for the number of transition rules A -> B, where B is any state.
    */

    int runTime = 0;
    cudaError_t cudaStatus;
    bool random = true;
           
    automatonSetup(size, runTime, random, transitionRules);

    newState.resize(size * size);
    currentState.resize(size * size);

    initField(currentState, newState, size, !random);

    printState(currentState.data(), size);
    printDelimiter(delimiter, size);

    while (runTime > 0) {
        //Iterate in parallel
        cudaStatus = iterateWithCuda(newState, currentState, (size * size), transitionRules);

        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "iterateWithCuda failed! Status: %i", cudaStatus);
            return 1;
        }        
        //Print result once each 100 iterations.
        if (runTime % 100 == 0) {
            
            printState(newState.data(), size);
            printDelimiter(delimiter, size);
        }
        //Copy new state to current state, clean up new state
        for (int i = 0; i < size * size; i++) {
            currentState[i] = newState[i];
            newState[i] = 0;
        }       
        runTime--;        
    }
  
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed! Status: %i", cudaStatus);
        return 1;
    }

    return 0;
}

std::vector<int> ruleTranslation(std::vector<std::vector<int>> transitionRules) {

    /*
        Sets up ruleset vector of size N*M. 
    */

    std::vector<int> ruleset(transitionRules.size()*10);
    int ruleShift = 0;
    //rule setup
    for each (std::vector<int> state in transitionRules)
    {
        for each (int rule in state)
        {          
            ruleset[rule + ruleShift] = 1;
        }
        ruleShift += 10;
    }
    
    return ruleset;
}

int checkRuleset(std::vector<int> transitionRules)
{
    /*
        Checks that transition rules submitted don't require impossible states or contain redundant transitions.
    */

    if (transitionRules.size() > 9) {
        return 1;
    }

    for each (int state in transitionRules)
    {
        if (state > 9) {
            return 1;
        }
    }

    return 0;
}

void setGliderTest(int* currentState, int size)
{
    /*
        Standard minimal GOL glider set roughly in center of the field. 
        Won't work in all possible rulesets but it is a good way to test behavior.
    */

    int yStart = int(((size * size) / 2) / size);
    int xStart = (int((size * size) / 2) % size) + 2;

    currentState[xStart + yStart * size] = 1;
    currentState[xStart + (yStart + 1) * size] = 1;
    currentState[xStart + (yStart + 2) * size] = 1;
    currentState[(xStart + 2) + (yStart + 1) * size] = 1;
    currentState[(xStart + 1) + (yStart + 2) * size] = 1;
}

void printState(int* currentState, int size)
{
    /*
        Self explanatory iteration over all elements printing their states.
        Since the states of elements are integers the lines might get a bit unenven.
    */
    for (int i = 0; i < size * size; i++)
    {
        if (i % size == 0) {
            printf("\n");
        }
        printf("%d ", currentState[i]);             
    }
}

void initField(std::vector<int> &currentState, std::vector<int> &newState, int size, bool gliderTest)
{
    /*
        Initializes field with either random values or preset pattern. 
    */

    for (int i = 0; i < size * size; i++)
    {
        if (rand() % 10 > 6 && !gliderTest) {
            currentState[i] = 1;
        }
        else {
            currentState[i] = 0;
        }
        newState[i] = 0;
    }

    if (gliderTest) {
        setGliderTest(currentState.data(), size);
    }

}

void automatonSetup(int& size, int& runTime, bool& randomInit, std::vector<int>& ruleSet)
{
    std::vector<std::vector<int>> allConfs = { };
    std::vector<int> surviveStates = { 2, 3 };
    std::vector<int> birthStates = { 3 };
    std::string arg;
    std::string substring;

    while (true)
    {
        printf("Enter field size: ");
        std::getline(std::cin, arg);
        size = std::stoi(arg);

        arg.clear();

        if (size > MAX_SIZE) {
            printf("Too large field size entered. \nPlease enter number < %d!", MAX_SIZE);
            size = 0;
        }
        else if (size <= 0) {
            printf("Too small field size entered. \nPlease enter number > %d!", 0);
            size = 0;
        }
        else {
            break;
        }
    }

    while (true)
    {
        printf(" \nEnter run time: ");
        std::getline(std::cin, arg);
        runTime = std::stoi(arg);

        arg.clear();

        if (runTime < 0) {
            printf("Negative run time entered. \nPlease enter runtime >= 0. ");
            runTime = 0;
        }
        else {
            break;
        }
    }

    while (true)
    {
        printf(" \nInitialize field with standard glider? N will result in random initial state. Y/N: ");
        std::getline(std::cin, arg);
        
        if (arg.compare("Y") == 0 || arg.compare("y") == 0) {
            randomInit = false;
            arg.clear();
            break;
        }
        else if (arg.compare("N") == 0 || arg.compare("n") == 0) {
            randomInit = true;
            arg.clear();
            break;
        }
        else {
            printf("Invalid string entered. Please enter Y or N.");
        } 
    }

    while (true) {

        printf(" \nEnter list of all states for 0 -> 1 transition, as number of active cells separated by space char: ");
        std::getline(std::cin, arg);
        std::stringstream argStream (arg);

        while (std::getline(argStream, substring, ' '))
        {
            birthStates.push_back(std::stoi(substring));
        }

        arg.clear();
        argStream.clear();

        if (checkRuleset(birthStates)!=0) {
            printf("Invalid list of states provided. Please make sure you enter at most 9 entries with each element <= 9. ");
        }
        else {
            break;
        }
    }

    while (true) {
        printf(" \nEnter list of all states for 1 -> 1 transition, as number of active cells separated by space char: ");
        std::getline(std::cin, arg);
        std::stringstream argStream(arg);

        while (std::getline(argStream, substring, ' '))
        {
            birthStates.push_back(std::stoi(substring));
        }

        arg.clear();
        argStream.clear();

        if (checkRuleset(surviveStates) != 0) {
            printf("Invalid list of states provided. Please make sure you enter at most 9 entries with each element <= 9. ");
        }
        else {
            break;
        }
    }

    allConfs.push_back(birthStates);
    allConfs.push_back(surviveStates);
    ruleSet = ruleTranslation(allConfs);
}

void printDelimiter(std::string delimiter, int outputSize)
{
    /*
        Prints selected delimiter string to separate two states of field in the output. 
    */

    printf("\n");

    for (int delimiterWidth = 0; delimiterWidth < outputSize; delimiterWidth++) {
        printf(delimiter.c_str());
    }

    printf("\n");
}

// Helper function for using CUDA to update state of the field.
cudaError_t iterateWithCuda(std::vector<int> &newState, std::vector<int> &currentState, unsigned int size, std::vector<int> ruleset)
{
    int* dev_currentState = 0;
    int* dev_newState = 0;
    int* dev_ruleset = 0;
    int* newStateArr = newState.data();    
    
    cudaError_t cudaStatus;
    
    int blockSize = 32;
    int numBlocks = (size + blockSize - 1) / blockSize;

    /*
        Most of this huge condition block is lifted straight from the tutorial.
        I just got rid of the goto. Arguably I replaced it with also ugly equivalent,
        repeating memory dealocation block, but it was a solution requiring the least work.
    */

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return cudaStatus;
    }

    // Allocate GPU buffers for three vectors (two for states, one for ruleset).
    cudaStatus = cudaMalloc((void**)&dev_currentState, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");

        cudaFree(dev_currentState);


        return cudaStatus;
    }

    cudaStatus = cudaMalloc((void**)&dev_newState, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");

        cudaFree(dev_currentState);
        cudaFree(dev_newState);

        return cudaStatus;
    }

    cudaStatus = cudaMalloc((void**)&dev_ruleset, ruleset.size() * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");

        cudaFree(dev_currentState);
        cudaFree(dev_newState);
        cudaFree(dev_ruleset);

        return cudaStatus;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_currentState, currentState.data(), size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");

        cudaFree(dev_currentState);
        cudaFree(dev_newState);
        cudaFree(dev_ruleset);

        return cudaStatus;
    }

    cudaStatus = cudaMemcpy(dev_ruleset, ruleset.data(), ruleset.size() * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");

        cudaFree(dev_currentState);
        cudaFree(dev_newState);
        cudaFree(dev_ruleset);

        return cudaStatus;
    }

    
    /*
        This essentially devides field into smaller pieces that are then processed in parallel fashion.
        The assumption is that field CAN be divided in one go.
    */
    iterateRule << <numBlocks, blockSize >> > (dev_newState, dev_currentState, int(sqrt(size)), dev_ruleset);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "iterateWithCuda launch failed: %s\n", cudaGetErrorString(cudaStatus));

        cudaFree(dev_currentState);
        cudaFree(dev_newState);
        cudaFree(dev_ruleset);

        return cudaStatus;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching iterateRule!\n", cudaStatus);

        cudaFree(dev_currentState);
        cudaFree(dev_newState);
        cudaFree(dev_ruleset);

        return cudaStatus;
    }

    // Copy new state of the field from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(newStateArr, dev_newState, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

    cudaFree(dev_currentState);
    cudaFree(dev_newState);
    cudaFree(dev_ruleset);

    return cudaStatus;
}